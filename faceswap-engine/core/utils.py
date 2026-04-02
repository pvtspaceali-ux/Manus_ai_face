# core/utils.py — Helper Functions
# Private FaceSwap Engine — Utility Module

import cv2
import numpy as np
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── File & Directory Helpers ──────────────────────────────

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def get_output_path(input_path, output_dir="./output", prefix="swapped_"):
    """Generate an output file path based on input filename."""
    ensure_dir(output_dir)
    filename = os.path.basename(input_path)
    return os.path.join(output_dir, f"{prefix}{filename}")


def cleanup_temp(temp_dir="./temp"):
    """Remove all temporary files (privacy: ephemeral processing)."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[CLEAN] Temp directory wiped: {temp_dir}")


def get_supported_image_extensions():
    """Return set of supported image file extensions."""
    return {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def get_supported_video_extensions():
    """Return set of supported video file extensions."""
    return {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


def is_image(path):
    """Check if a file is a supported image format."""
    return Path(path).suffix.lower() in get_supported_image_extensions()


def is_video(path):
    """Check if a file is a supported video format."""
    return Path(path).suffix.lower() in get_supported_video_extensions()


def is_gif(path):
    """Check if a file is a GIF."""
    return Path(path).suffix.lower() == '.gif'


# ── Image Helpers ─────────────────────────────────────────

def load_image(path):
    """Load an image as BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def save_image(img, path):
    """Save a BGR numpy array as an image file."""
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, img)
    return path


def bgr_to_rgb(img):
    """Convert BGR (OpenCV) to RGB (PIL/Gradio)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    """Convert RGB (PIL/Gradio) to BGR (OpenCV)."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def pil_to_cv2(pil_img):
    """Convert PIL Image to OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img):
    """Convert OpenCV BGR numpy array to PIL Image."""
    from PIL import Image
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_image(img, max_size=1920):
    """
    Resize image if larger than max_size while preserving aspect ratio.

    Args:
        img: BGR numpy array
        max_size: Maximum dimension (width or height)

    Returns:
        Resized image (or original if already within limits)
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── System Checks ─────────────────────────────────────────

def check_ffmpeg():
    """Check if FFmpeg is available in PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        version_line = result.stdout.split('\n')[0]
        print(f"[OK] FFmpeg found: {version_line}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[ERROR] FFmpeg not found! Install it and add to PATH.")
        return False


def check_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"[OK] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print("[INFO] No CUDA GPU detected. Running on CPU.")
            return False
    except ImportError:
        print("[INFO] PyTorch not installed. GPU check skipped.")
        return False


def check_models(models_dir="./models"):
    """Check if all required model files are present."""
    required = {
        "inswapper_128.onnx": "https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx",
        "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    }

    optional = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }

    all_found = True

    print("\n--- Model Check ---")
    for filename, url in required.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [OK] {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {filename}")
            print(f"           Download: {url}")
            all_found = False

    # Check buffalo_l directory
    buffalo_dir = os.path.join(models_dir, "buffalo_l")
    if os.path.isdir(buffalo_dir):
        onnx_files = [f for f in os.listdir(buffalo_dir) if f.endswith('.onnx')]
        print(f"  [OK] buffalo_l/ ({len(onnx_files)} model files)")
    else:
        print(f"  [INFO] buffalo_l/ not found (will auto-download on first run)")

    for filename, url in optional.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [OK] {filename} ({size_mb:.1f} MB) [optional]")
        else:
            print(f"  [SKIP] {filename} (optional — for background upscaling)")

    print("-------------------\n")
    return all_found


def run_system_check():
    """Run all system checks and print a summary."""
    print("=" * 50)
    print("  Private FaceSwap Engine — System Check")
    print("=" * 50)

    gpu_ok = check_gpu()
    ffmpeg_ok = check_ffmpeg()
    models_ok = check_models()

    print("=" * 50)
    if gpu_ok and ffmpeg_ok and models_ok:
        print("  STATUS: ALL CHECKS PASSED")
    else:
        print("  STATUS: Some checks failed — see above")
    print("=" * 50)

    return gpu_ok, ffmpeg_ok, models_ok


if __name__ == "__main__":
    run_system_check()
