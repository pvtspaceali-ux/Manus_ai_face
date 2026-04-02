#!/bin/bash
# ============================================================
#  Private FaceSwap Engine — Automated Setup (Linux / macOS)
# ============================================================
# Usage: chmod +x setup.sh && ./setup.sh

set -e

echo "============================================"
echo "  Private FaceSwap Engine — Setup Script"
echo "============================================"
echo ""

# ── Step 1: Check for Conda ──
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found! Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "[OK] Conda found"

# ── Step 2: Check for FFmpeg ──
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARN] FFmpeg not found. Video processing will not work."
    echo "  Install: sudo apt install ffmpeg  (Ubuntu/Debian)"
    echo "  Install: brew install ffmpeg       (macOS)"
else
    echo "[OK] FFmpeg found"
fi

# ── Step 3: Create Conda Environment ──
echo ""
echo "[SETUP] Creating Conda environment 'faceswap' with Python 3.10..."
conda create -n faceswap python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate faceswap

# ── Step 4: Install PyTorch ──
echo ""
echo "[SETUP] Installing PyTorch..."

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU] NVIDIA GPU detected. Installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "[CPU] No NVIDIA GPU detected. Installing CPU version..."
    pip install torch torchvision torchaudio
fi

# ── Step 5: Install Dependencies ──
echo ""
echo "[SETUP] Installing Python dependencies..."
pip install insightface==0.7.3
pip install opencv-python
pip install gradio==4.44.0
pip install Pillow numpy tqdm
pip install gfpgan basicsr facexlib realesrgan

# Install appropriate ONNX Runtime
if command -v nvidia-smi &> /dev/null; then
    pip install onnxruntime-gpu
else
    pip install onnxruntime
fi

# ── Step 6: Create Directory Structure ──
echo ""
echo "[SETUP] Creating directory structure..."
mkdir -p models input/source input/target output temp

# ── Step 7: Download Models ──
echo ""
echo "[SETUP] Downloading AI models..."
echo "  This may take a few minutes depending on your connection."
echo ""

# Model 1: inswapper_128.onnx
if [ ! -f "./models/inswapper_128.onnx" ]; then
    echo "  Downloading inswapper_128.onnx (~554 MB)..."
    wget -q --show-progress -O ./models/inswapper_128.onnx \
        "https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx"
else
    echo "  [SKIP] inswapper_128.onnx already exists"
fi

# Model 2: GFPGANv1.4.pth
if [ ! -f "./models/GFPGANv1.4.pth" ]; then
    echo "  Downloading GFPGANv1.4.pth (~300 MB)..."
    wget -q --show-progress -O ./models/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
else
    echo "  [SKIP] GFPGANv1.4.pth already exists"
fi

# Model 3: RealESRGAN_x4plus.pth (optional)
if [ ! -f "./models/RealESRGAN_x4plus.pth" ]; then
    echo "  Downloading RealESRGAN_x4plus.pth (~65 MB) [optional]..."
    wget -q --show-progress -O ./models/RealESRGAN_x4plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
else
    echo "  [SKIP] RealESRGAN_x4plus.pth already exists"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  To start the tool:"
echo "    conda activate faceswap"
echo "    python app.py"
echo ""
echo "  Then open: http://127.0.0.1:7860"
echo "============================================"
