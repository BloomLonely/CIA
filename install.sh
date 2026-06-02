#!/bin/bash
set -e

# 1. PyTorch (CUDA 12.4 빌드)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. torch-scatter / torch-sparse (PyG 전용 휠 — PyPI 접근 차단)
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
    --find-links https://data.pyg.org/whl/torch-2.5.1+cu124.html \
    --no-index

# 3. torch-geometric (PyPI)
pip install torch-geometric==2.7.0

# 4. 나머지 패키지 (PyPI)
pip install -r requirements.txt
