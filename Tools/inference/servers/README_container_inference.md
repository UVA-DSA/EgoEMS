# Container Inference Server

This document explains how to build and run the DETR TensorRT inference server in Docker so that:

- the required Python packages are already installed in the image
- the server starts automatically when the container starts
- your host Python program can simply send frames to `localhost:8000`

## What You Have Now

The container runtime files live here:

- [Dockerfile](/mnt/f/repos/EgoEMS/Tools/inference/servers/Dockerfile)
- [start_inference_server.sh](/mnt/f/repos/EgoEMS/Tools/inference/servers/start_inference_server.sh)
- [container_inference_server.py](/mnt/f/repos/EgoEMS/Tools/inference/servers/container_inference_server.py)

The test client lives here:

- [test_video_client.py](/mnt/f/repos/EgoEMS/Tools/inference/tests/test_video_client.py)

## Why You Had To Reinstall Packages Before

When you ran:

- a base NVIDIA image
- then `pip install ...` inside that running container

those installed packages only lived inside that temporary container.

If you started the container with `--rm`, Docker deleted that container when it exited, so the installed packages disappeared too.

That is why you had to install them again every time.

## What This New Setup Changes

Now you will:

1. build your own Docker image once
2. that image already contains the server dependencies
3. the image automatically starts the inference server
4. you run one Docker command
5. your host program sends frames to `http://localhost:8000/infer`

## Architecture

What runs where:

- host machine
  - your normal Python program
  - optional test client
  - sends frames over HTTP
- Docker container
  - loads the TensorRT model once
  - keeps it in GPU memory
  - serves `/health` and `/infer`

The repo is mounted into the container at runtime, so:

- your server code is read from the repo
- your TensorRT engine file is read from the repo
- code edits on the host are visible inside the container

## Prerequisites

Before starting:

1. Docker is installed in WSL.
2. NVIDIA Container Toolkit is working.
3. `docker run --gpus all ...` works.
4. You can access the repo at `/mnt/f/repos/EgoEMS`.
5. Your engine exists at:
   - `/mnt/f/repos/EgoEMS/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts`

## Step 1: Build The Docker Image

Run this from WSL:

```bash
docker build \
  -t egoems-detr-server:latest \
  /mnt/f/repos/EgoEMS/Tools/inference/servers
```

What this does:

- starts from `nvcr.io/nvidia/pytorch:26.02-py3`
- installs `aiohttp` and `opencv-python-headless`
- copies in the startup script
- makes the container auto-start the server

You only need to rebuild when:

- you change the `Dockerfile`
- you change the startup script
- you want to add more packages to the image

You do not need to rebuild just because you changed Python code in the repo, since the repo is mounted at runtime.

## Step 2: Start The Container

Run this from WSL:

```bash
docker run --gpus all --rm -d \
  --name egoems-detr-server \
  -p 8000:8000 \
  -v /mnt/f/repos/EgoEMS:/workspace/EgoEMS \
  egoems-detr-server:latest
```

What this does:

- starts the container in the background with `-d`
- gives it GPU access
- mounts your repo into `/workspace/EgoEMS`
- exposes the server on `localhost:8000`
- automatically launches the inference server

You do not need to manually run Python inside the container anymore.

## Step 3: Watch The Logs

To confirm the server started:

```bash
docker logs -f egoems-detr-server
```

You should see startup lines showing:

- app home
- backend
- engine path
- host and port

Stop following logs with `Ctrl+C`.

## Step 4: Check That The Server Is Healthy

From the host:

```bash
curl http://localhost:8000/health
```

Expected result:

```json
{
  "status": "ok",
  "backend": {
    "model_name": "detr_tensorrt",
    "engine_path": "/workspace/EgoEMS/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts",
    "detr_version": "ems",
    "threshold": 0.7,
    "engine_height": 480,
    "engine_width": 640,
    "device": "cuda"
  }
}
```

## Step 5: Send Frames To It

Once the container is running, your host Python code can send frames to:

```text
http://localhost:8000/infer
```

The server accepts:

- raw image bytes in the request body
- or `multipart/form-data`

It returns:

- detections
- image size
- preprocess timing
- inference timing
- postprocess timing

## Step 6: Run The Included Test Client

The simplest end-to-end test is:

1. start the Docker container
2. run the host-side test client

The test client lives at:

- [test_video_client.py](/mnt/f/repos/EgoEMS/Tools/inference/tests/test_video_client.py)

Run it from the host:

```bash
python /mnt/f/repos/EgoEMS/Tools/inference/tests/test_video_client.py
```

Inside that file, you can choose:

- `USE_WEBCAM = False` to read a video file
- `USE_WEBCAM = True` to use a live webcam

Other variables you can edit in the file:

- `VIDEO_PATH`
- `WEBCAM_INDEX`
- `SERVER_URL`
- `MAX_FRAMES`
- `SEND_EVERY_NTH_FRAME`
- `JPEG_QUALITY`
- `SHOW_PREVIEW`

Example output:

```text
[frame 1] pre=2.10 ms infer=11.85 ms post=0.72 ms | bvm(0.94) [101.2, 119.8, 240.6, 300.1]
```

## Common Commands

Start the server:

```bash
docker run --gpus all --rm -d \
  --name egoems-detr-server \
  -p 8000:8000 \
  -v /mnt/f/repos/EgoEMS:/workspace/EgoEMS \
  egoems-detr-server:latest
```

See logs:

```bash
docker logs -f egoems-detr-server
```

Stop the container:

```bash
docker stop egoems-detr-server
```

Remove the image later if you want:

```bash
docker image rm egoems-detr-server:latest
```

## Customizing The Server Without Editing Dockerfile

The startup script reads environment variables, so you can override settings at `docker run` time.

Available environment variables:

- `APP_HOME`
- `BACKEND`
- `ENGINE_PATH`
- `SERVER_HOST`
- `SERVER_PORT`
- `DEVICE`
- `WARMUP`
- `DETR_VERSION`
- `DETECTION_THRESHOLD`
- `ENGINE_HEIGHT`
- `ENGINE_WIDTH`

Example:

```bash
docker run --gpus all --rm -d \
  --name egoems-detr-server \
  -p 8000:8000 \
  -v /mnt/f/repos/EgoEMS:/workspace/EgoEMS \
  -e ENGINE_PATH=/workspace/EgoEMS/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts \
  -e DETECTION_THRESHOLD=0.6 \
  -e WARMUP=5 \
  egoems-detr-server:latest
```

## If You Want To Open A Shell Inside The Container

Sometimes you may still want an interactive shell for debugging:

```bash
docker run --gpus all --rm -it \
  -v /mnt/f/repos/EgoEMS:/workspace/EgoEMS \
  egoems-detr-server:latest \
  bash
```

Because the startup script uses:

- the default server when no extra command is passed
- your command when one is passed

that `bash` command opens a shell instead of starting the server.

## API Summary

Endpoints:

- `GET /health`
- `POST /infer`

`POST /infer` accepts either:

- `multipart/form-data`
  - field `image`
  - optional field `frame_id`
  - optional field `timestamp`
- raw request body
  - optional header `x-frame-id`
  - optional header `x-timestamp`

Example raw request with `curl`:

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/octet-stream" \
  -H "x-frame-id: 1" \
  --data-binary @/mnt/f/repos/EgoEMS/Tools/inference/test_frame.jpg
```

Example response shape:

```json
{
  "model_name": "detr_tensorrt",
  "frame_id": "1",
  "timestamp": "1710200000.123456",
  "image_width": 1280,
  "image_height": 720,
  "inference_ms": 12.3,
  "preprocess_ms": 2.1,
  "postprocess_ms": 0.8,
  "detections": []
}
```

## Troubleshooting

### The container exits immediately

Check the logs:

```bash
docker logs egoems-detr-server
```

Common causes:

- repo volume was not mounted
- engine path is wrong
- engine file does not exist

### `curl http://localhost:8000/health` fails

Check:

- container is running
- port mapping includes `-p 8000:8000`
- the server did not crash during startup

Check running containers:

```bash
docker ps
```

### I changed Python code in the repo

You usually do not need to rebuild the Docker image.

Just:

1. stop the running container
2. start it again

Because the repo is mounted from the host, the container sees the updated code.

### I changed the Dockerfile or startup script

Then rebuild:

```bash
docker build \
  -t egoems-detr-server:latest \
  /mnt/f/repos/EgoEMS/Tools/inference/servers
```

### I want the container to stay around after stop

Remove `--rm` from `docker run`.

For development, `--rm` is usually simpler.

## Recommended First Workflow

Use this exact order:

1. build the image once
2. start the container
3. watch the logs
4. call `/health`
5. run the test client
6. then connect your real host Python program

That gives you the easiest debugging path.
