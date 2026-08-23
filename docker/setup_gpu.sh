#!/usr/bin/env bash
# Install the NVIDIA Container Toolkit on the host so Docker can pass the GPU
# into the container. Run on the host, not inside the container.
set -euo pipefail

KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list
PROBE=ubuntu:24.04

say() { printf '\n== %s\n' "$*"; }

gpu_works() { docker run --rm --gpus all "$PROBE" true >/dev/null 2>&1; }

restart_docker() {
    if command -v systemctl >/dev/null && systemctl is-system-running >/dev/null 2>&1; then
        sudo systemctl restart docker
    else
        sudo service docker restart
    fi
    for _ in $(seq 30); do
        docker info >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "docker did not come back up" >&2
    return 1
}

if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found: no NVIDIA driver visible to this host/WSL." >&2
    echo "On WSL, install the driver on Windows -- never inside WSL." >&2
    exit 1
fi
nvidia-smi -L

if ! docker info >/dev/null 2>&1; then
    echo "cannot talk to the docker daemon" >&2
    exit 1
fi

if gpu_works; then
    say "GPU already works in docker; nothing to do"
    exit 0
fi

if ! command -v nvidia-ctk >/dev/null; then
    say "installing nvidia-container-toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor --yes -o "$KEYRING"
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" \
        | sudo tee "$LIST" >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
fi

say "registering the runtime with docker"
sudo nvidia-ctk runtime configure --runtime=docker
restart_docker

if ! gpu_works; then
    say "falling back to a generated CDI spec"
    sudo mkdir -p /etc/cdi
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    restart_docker
fi

if ! gpu_works; then
    echo "GPU still unavailable to docker; run 'docker run --rm --gpus all $PROBE true' to see the error" >&2
    exit 1
fi

say "GPU is available to docker"
docker run --rm --gpus all "$PROBE" nvidia-smi -L
say "next: python3 docker/container.py build --gpu && python3 docker/container.py start"
