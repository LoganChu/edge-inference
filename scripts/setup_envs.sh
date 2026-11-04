#!/usr/bin/env bash
# Build the two Python environments used by this comparison.
#
#   .venvs/trt   torch 2.14.1+cu132 · Torch-TensorRT 2.14.0 · TensorRT 11.1 · ExecuTorch 1.4.1
#                · torch-tensorrt-executorch-runtime 0.1.0  (direct TRT, ET+TRT, ET+CUDA on the ET 1.4.1 runtime)
#   .venvs/et15  torch 2.15.0+cu132 · ExecuTorch 1.5.1+cu132  (ET+CUDA on the newest ET runtime)
#
# LeRobot 0.6.1 pins torch<2.12, so it is installed with --no-deps and its real
# runtime dependencies are listed explicitly.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PT=https://download.pytorch.org/whl/cu132
LEROBOT_DEPS=(opencv-python-headless sentencepiece protobuf einops draccus "huggingface-hub>=1.6" gymnasium safetensors "numpy>=2.0,<2.3" Pillow scipy "transformers>=5.4,<5.6" rich tqdm)

make_env () {  # name, then pip specs
  local name=$1; shift
  uv venv --python 3.12 -q ".venvs/$name"
  uv pip install --python ".venvs/$name/bin/python" -q \
     --index-url "$PT" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "$@"
  uv pip install --python ".venvs/$name/bin/python" -q --no-deps lerobot==0.6.1
  uv pip install --python ".venvs/$name/bin/python" -q "${LEROBOT_DEPS[@]}"
}

case "${1:-all}" in
  trt|all)
    make_env trt "torch==2.14.1+cu132" "torchvision" "torch_tensorrt==2.14.0+cu132" "nvidia-modelopt[torch]==0.48.0" \
                 "tensorrt-cu13==11.1.0.106" "executorch==1.4.1" "torch-tensorrt-executorch-runtime==0.1.0" ;;&
  et15|all)
    make_env et15 "torch==2.15.0+cu132" "torchvision" "executorch==1.5.1+cu132" ;;
esac
echo "SETUP DONE: ${1:-all}"
