FROM quay.io/jupyter/scipy-notebook:latest

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

USER jovyan
