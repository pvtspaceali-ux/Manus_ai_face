@echo off
REM ============================================================
REM  Private FaceSwap Engine — Automated Setup (Windows)
REM ============================================================
REM  Run this from Anaconda Prompt (NOT regular cmd/PowerShell)
REM  Usage: setup.bat

echo ============================================
echo   Private FaceSwap Engine — Setup Script
echo ============================================
echo.

REM ── Step 1: Create Conda Environment ──
echo [SETUP] Creating Conda environment 'faceswap' with Python 3.10...
conda create -n faceswap python=3.10 -y
call conda activate faceswap

REM ── Step 2: Install PyTorch (CUDA) ──
echo.
echo [SETUP] Installing PyTorch (CUDA 11.8)...
echo   If you don't have an NVIDIA GPU, cancel and run:
echo   pip install torch torchvision torchaudio
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM ── Step 3: Install Dependencies ──
echo.
echo [SETUP] Installing Python dependencies...
pip install insightface==0.7.3
pip install onnxruntime-gpu
pip install opencv-python
pip install gradio==4.44.0
pip install Pillow numpy tqdm
pip install gfpgan basicsr facexlib realesrgan

REM ── Step 4: Create Directories ──
echo.
echo [SETUP] Creating directory structure...
if not exist "models" mkdir models
if not exist "input\source" mkdir input\source
if not exist "input\target" mkdir input\target
if not exist "output" mkdir output
if not exist "temp" mkdir temp

REM ── Step 5: Download Models ──
echo.
echo [SETUP] Downloading AI models...
echo   You can also download these manually from the URLs below.
echo.

if not exist "models\inswapper_128.onnx" (
    echo   Downloading inswapper_128.onnx (~554 MB)...
    curl -L -o models\inswapper_128.onnx "https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx"
) else (
    echo   [SKIP] inswapper_128.onnx already exists
)

if not exist "models\GFPGANv1.4.pth" (
    echo   Downloading GFPGANv1.4.pth (~300 MB)...
    curl -L -o models\GFPGANv1.4.pth "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
) else (
    echo   [SKIP] GFPGANv1.4.pth already exists
)

if not exist "models\RealESRGAN_x4plus.pth" (
    echo   Downloading RealESRGAN_x4plus.pth (~65 MB)...
    curl -L -o models\RealESRGAN_x4plus.pth "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
) else (
    echo   [SKIP] RealESRGAN_x4plus.pth already exists
)

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo   To start the tool:
echo     conda activate faceswap
echo     python app.py
echo.
echo   Then open: http://127.0.0.1:7860
echo ============================================
pause
