import argparse
import asyncio
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from aiohttp import web

from container_inference.backends import DETRTensorRTBackend, InferenceBackend


def build_backend(args: argparse.Namespace) -> InferenceBackend:
    if args.backend == "detr-trt":
        return DETRTensorRTBackend(
            engine_path=args.engine,
            detr_version=args.detr_version,
            threshold=args.threshold,
            engine_height=args.engine_height,
            engine_width=args.engine_width,
            device=args.device,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("No image bytes were provided.")
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes.")
    return image


async def parse_request_payload(request: web.Request) -> Tuple[bytes, Optional[str], Optional[str]]:
    content_type = request.content_type.lower()
    if content_type.startswith("multipart/"):
        reader = await request.multipart()
        image_bytes = b""
        frame_id = None
        timestamp = None

        async for part in reader:
            if part.name == "image":
                image_bytes = await part.read(decode=False)
            elif part.name == "frame_id":
                frame_id = (await part.text()).strip() or None
            elif part.name == "timestamp":
                timestamp = (await part.text()).strip() or None

        return image_bytes, frame_id, timestamp

    image_bytes = await request.read()
    frame_id = request.headers.get("x-frame-id")
    timestamp = request.headers.get("x-timestamp")
    return image_bytes, frame_id, timestamp


async def health(request: web.Request) -> web.Response:
    backend: InferenceBackend = request.app["backend"]
    return web.json_response(
        {
            "status": "ok",
            "backend": backend.metadata(),
        }
    )


@web.middleware
async def json_error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def infer(request: web.Request) -> web.Response:
    backend: InferenceBackend = request.app["backend"]
    lock: asyncio.Lock = request.app["inference_lock"]

    image_bytes, frame_id, timestamp = await parse_request_payload(request)
    image_bgr = decode_image_bytes(image_bytes)

    async with lock:
        result = backend.infer(image_bgr=image_bgr, frame_id=frame_id, timestamp=timestamp)

    return web.json_response(result.to_dict())


def build_parser(default_engine: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Container-friendly realtime inference server.")
    parser.add_argument("--backend", type=str, default="detr-trt", choices=["detr-trt"], help="Backend/model implementation.")
    parser.add_argument("--engine", type=Path, default=default_engine, help="Path to TensorRT engine (.ts/.ep).")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before serving requests.")
    parser.add_argument("--detr-version", type=str, default="ems", choices=["ems", "base"], help="DETR class set.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Detection confidence threshold.")
    parser.add_argument("--engine-height", type=int, default=480, help="Compile-time engine input height.")
    parser.add_argument("--engine-width", type=int, default=640, help="Compile-time engine input width.")
    return parser


def create_app(args: argparse.Namespace) -> web.Application:
    backend = build_backend(args)
    backend.warmup(args.warmup)

    app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[json_error_middleware])
    app["backend"] = backend
    app["inference_lock"] = asyncio.Lock()
    app.add_routes(
        [
            web.get("/health", health),
            web.post("/infer", infer),
        ]
    )
    return app


def main() -> None:
    inference_root = Path(__file__).resolve().parents[2]
    default_engine = inference_root / "checkpoints" / "ems_finetuned_detr_trt.ts"
    parser = build_parser(default_engine=default_engine)
    args = parser.parse_args()

    app = create_app(args)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
