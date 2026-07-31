#!/usr/bin/env bash
set -euo pipefail

image="${IMPERIUM_RUNTIME_IMAGE:-imperium-mlebench-runtime:dev-20260715}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Docker is installed as a confined snap on the source server. Streaming the
# context avoids its BuildKit bind-mount restriction on /mnt/raid0.
tar --exclude='./__pycache__' --exclude='./.pytest_cache' --exclude='./.ruff_cache' \
  --exclude='./*.pyc' -C "$root" -cf - . \
  | docker build --pull --tag "$image" -
docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,exec,size=8g \
  "$image" python /opt/imperium/smoke_runtime.py

for gpu in 0 1; do
  docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,exec,size=8g \
    --runtime nvidia --env "NVIDIA_VISIBLE_DEVICES=${gpu}" --env CUDA_VISIBLE_DEVICES=0 \
    --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
    "$image" \
    python /opt/imperium/smoke_runtime.py --gpu --dali
done

docker image inspect "$image" --format '{{.Id}}'
