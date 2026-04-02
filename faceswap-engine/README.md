# Private FaceSwap Engine v1.0

**100% local, zero-cloud face swap tool for personal use.** No accounts, no uploads, no telemetry. Everything runs on your machine.

---

## Features

| Feature | Description |
|---|---|
| **Photo Swap** | Single-image face swap with identity preservation |
| **Video Swap** | Frame-by-frame processing with FFmpeg audio re-muxing |
| **GIF Swap** | Decompose, swap, and reconstruct GIFs with original timing |
| **Multi-Face** | Detect all faces, select specific ones to swap |
| **Batch Processing** | Process entire folders of images automatically |
| **Face Enhancement** | GFPGAN v1.4 restores skin texture, sharpens eyes |
| **Upscaling** | 1x, 2x, or 4x output resolution via Real-ESRGAN |
| **GPU Acceleration** | CUDA support for NVIDIA GPUs (CPU fallback available) |
| **Web UI** | Gradio browser interface at `localhost:7860` |
| **CLI Tool** | Command-line interface for scripting and automation |

---

## Quick Start

### Automated Setup

**Linux / macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows (Anaconda Prompt):**
```cmd
setup.bat
```

### Manual Setup

```bash
# 1. Create environment
conda create -n faceswap python=3.10 -y
conda activate faceswap

# 2. Install PyTorch
# GPU (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU only:
# pip install torch torchvision torchaudio

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (see Model Downloads section below)

# 5. Launch
python app.py
# Open http://127.0.0.1:7860
```

---

## Project Structure

```
faceswap-engine/
├── models/                    # AI model files (download separately)
│   ├── inswapper_128.onnx     # Core swap model (~554 MB)
│   ├── buffalo_l/             # Face detection pack (~200 MB)
│   ├── GFPGANv1.4.pth        # Face enhancer (~300 MB)
│   └── RealESRGAN_x4plus.pth # Background upscaler (~65 MB)
├── input/
│   ├── source/                # Your source face photos
│   └── target/                # Target photos/videos
├── output/                    # All results saved here
├── temp/                      # Auto-wiped after each job
├── core/
│   ├── __init__.py            # Module exports
│   ├── swapper.py             # Face swap engine (InsightFace + inswapper)
│   ├── enhancer.py            # GFPGAN face restoration
│   ├── video.py               # FFmpeg video/GIF pipeline
│   └── utils.py               # Helper functions & system checks
├── app.py                     # Gradio Web UI (main entry point)
├── swap.py                    # CLI tool
├── requirements.txt           # Python dependencies
├── setup.sh                   # Linux/Mac setup script
└── setup.bat                  # Windows setup script
```

---

## Model Downloads

Download these files and place them in the `models/` directory:

| Model | Size | Download |
|---|---|---|
| **inswapper_128.onnx** | ~554 MB | [HuggingFace](https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx) |
| **GFPGANv1.4.pth** | ~300 MB | [GitHub Release](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth) |
| **RealESRGAN_x4plus.pth** | ~65 MB | [GitHub Release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| **buffalo_l** | ~200 MB | Auto-downloads on first run, or [GitHub](https://github.com/deepinsight/insightface/releases/tag/v0.7) |

**Quick download (Linux/Mac):**
```bash
# inswapper_128.onnx
wget -O ./models/inswapper_128.onnx \
  "https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx"

# GFPGANv1.4.pth
wget -O ./models/GFPGANv1.4.pth \
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

# RealESRGAN_x4plus.pth
wget -O ./models/RealESRGAN_x4plus.pth \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
```

---

## Usage

### Web UI (Recommended)

```bash
conda activate faceswap
python app.py
# Open http://127.0.0.1:7860 in your browser
```

The Web UI has five tabs:

1. **Photo Swap** — Upload source face + target photo, click Swap
2. **Multi-Face** — Detect faces in target, select specific face index to swap
3. **Video Swap** — Upload source face + target video, process frame-by-frame
4. **GIF Swap** — Upload source face + target GIF
5. **Batch Process** — Upload source face + multiple target images

### CLI Tool

```bash
# Basic photo swap
python swap.py -s face.jpg -t target.jpg

# With GFPGAN enhancement
python swap.py -s face.jpg -t target.jpg -e

# With 2x upscale
python swap.py -s face.jpg -t target.jpg -e --upscale 2

# Video swap
python swap.py -s face.jpg -t video.mp4

# Video swap with enhancement (slower)
python swap.py -s face.jpg -t video.mp4 -e

# GIF swap
python swap.py -s face.jpg -t animation.gif

# Batch mode (all images in a folder)
python swap.py -s face.jpg --batch ./input/target/

# Swap specific face (multi-face)
python swap.py -s face.jpg -t group_photo.jpg --face-index 2

# Force CPU mode
python swap.py -s face.jpg -t target.jpg --cpu

# System check
python swap.py --check
```

---

## Hardware Requirements

| Tier | GPU | RAM | Photo Speed | Video Speed |
|---|---|---|---|---|
| **Minimum** (CPU) | None | 16 GB | 45-120 sec | ~5 min/min |
| **Recommended** | RTX 3060/3070 (8GB+) | 32 GB | 1-3 sec | ~45 sec/min |
| **Optimal** | RTX 4090 (24GB) | 64 GB | < 0.5 sec | Real-time |

---

## Technology Stack

| Component | Technology |
|---|---|
| Face Detection | InsightFace buffalo_l (RetinaFace + ArcFace) |
| Face Swap | inswapper_128.onnx |
| Face Enhancement | GFPGAN v1.4 |
| Upscaling | Real-ESRGAN x4plus |
| Video Engine | FFmpeg + OpenCV |
| UI Framework | Gradio 4.x |
| Inference Runtime | ONNX Runtime (CUDA) |
| Environment | Python 3.10 / Conda |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No face detected` | Face too small / blurry / side profile | Use a clear, front-facing photo. Min face size ~100px |
| `CUDA out of memory` | GPU VRAM exhausted | Set `use_gpu=False` or reduce `det_size` to (320,320) |
| `onnxruntime not found` | Wrong package installed | `pip install onnxruntime-gpu` (GPU) or `onnxruntime` (CPU) |
| `ffmpeg not recognized` | FFmpeg not in PATH | Add FFmpeg `bin/` folder to system PATH |
| Swap result looks blurry | Enhancement not enabled | Enable GFPGAN checkbox; ensure model is downloaded |
| Video has no audio | FFmpeg audio mux failed | Check FFmpeg version >= 5.0 |

---

## Privacy and Security

This tool is designed with privacy as the top priority:

- **100% Local Processing** — No internet required after model download
- **Zero Telemetry** — No analytics, no "phone home" requests
- **Data Sovereignty** — All files remain on your local machine
- **Ephemeral Processing** — Temp frames are auto-wiped after encoding
- **Inference Only** — Does not train on your photos

For maximum privacy, disconnect from the internet while processing sensitive media.

---

## Pro Tips

1. **Source Photo Quality** — Use a high-res, well-lit, front-facing photo (minimum 512x512px)
2. **Always Enable Enhancement** — Raw inswapper output is 128px; GFPGAN makes it look like 1024px
3. **Upscale for Print** — Set `upscale=2` or `upscale=4` for high-resolution output
4. **Video Speed** — Skip per-frame enhancement for long videos; enhance key frames only

---

*Private FaceSwap Engine v1.0 — For personal research use only.*
*Models referenced (InsightFace, GFPGAN) are subject to their respective non-commercial research licenses.*
