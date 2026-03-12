FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/workspace/EgoEMS
ENV BACKEND=detr-trt
ENV ENGINE_PATH=/workspace/EgoEMS/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV DEVICE=cuda
ENV WARMUP=3
ENV DETR_VERSION=ems
ENV DETECTION_THRESHOLD=0.7
ENV ENGINE_HEIGHT=480
ENV ENGINE_WIDTH=640

WORKDIR /opt/inference-server

COPY start_inference_server.sh /usr/local/bin/start-inference-server.sh

RUN chmod +x /usr/local/bin/start-inference-server.sh \
    && python -m pip install --no-cache-dir aiohttp opencv-python-headless

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/start-inference-server.sh"]
