# app.py — Gradio Web UI (Main Entry Point)
# Private FaceSwap Engine — 100% Local, Zero Cloud
# Run: python app.py → Open http://127.0.0.1:7860

import gradio as gr
import cv2
import numpy as np
import os
import tempfile
import time

from core.swapper import FaceSwapper
from core.enhancer import FaceEnhancer
from core.video import VideoProcessor
from core.utils import (
    ensure_dir, cleanup_temp, check_ffmpeg, check_gpu,
    check_models, pil_to_cv2, bgr_to_rgb
)


# ── Configuration ─────────────────────────────────────────
MODELS_DIR = "./models"
OUTPUT_DIR = "./output"
TEMP_DIR = "./temp"
SWAP_MODEL = os.path.join(MODELS_DIR, "inswapper_128.onnx")
GFPGAN_MODEL = os.path.join(MODELS_DIR, "GFPGANv1.4.pth")
REALESRGAN_MODEL = os.path.join(MODELS_DIR, "RealESRGAN_x4plus.pth")

# Auto-detect GPU
USE_GPU = False
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    pass


# ── Load Models at Startup ────────────────────────────────
print("\n" + "=" * 55)
print("  Private FaceSwap Engine — Starting Up...")
print("=" * 55)

swapper = FaceSwapper(model_path=SWAP_MODEL, use_gpu=USE_GPU)
enhancer = FaceEnhancer(
    model_path=GFPGAN_MODEL,
    upscale=1,
    bg_upsampler_path=REALESRGAN_MODEL if os.path.exists(REALESRGAN_MODEL) else None
)
video_proc = VideoProcessor(swapper, enhancer)

ensure_dir(OUTPUT_DIR)
ensure_dir(TEMP_DIR)

print("=" * 55)
print(f"  GPU: {'Enabled' if USE_GPU else 'CPU Mode'}")
print("  Ready! Launching Gradio UI...")
print("=" * 55 + "\n")


# ── Photo Swap Function ──────────────────────────────────
def swap_photo(source_pil, target_pil, use_enhance, upscale_factor):
    """Swap faces in a single photo."""
    if source_pil is None:
        raise gr.Error("Please upload a source face image!")
    if target_pil is None:
        raise gr.Error("Please upload a target photo!")

    start = time.time()

    # Convert PIL to OpenCV BGR
    src = pil_to_cv2(source_pil)
    tgt = pil_to_cv2(target_pil)

    # Perform face swap
    result = swapper.swap_faces(src, tgt)

    # Apply enhancement if enabled
    if use_enhance:
        # Adjust upscale if needed
        if upscale_factor != enhancer.upscale:
            enhancer.set_upscale(int(upscale_factor))
        result = enhancer.enhance(result)

    elapsed = time.time() - start

    # Convert back to RGB for Gradio display
    result_rgb = bgr_to_rgb(result)

    # Save to output directory
    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"photo_{timestamp}.png")
    cv2.imwrite(output_path, result)

    status = f"Done in {elapsed:.1f}s | Saved: {output_path}"
    return result_rgb, status


# ── Multi-Face Swap Function ─────────────────────────────
def detect_faces_preview(target_pil):
    """Detect faces in target and return annotated preview."""
    if target_pil is None:
        return None, "Upload a target image first"

    tgt = pil_to_cv2(target_pil)
    try:
        faces = swapper.get_faces(tgt)
    except ValueError:
        return None, "No faces detected in target image"

    # Draw bounding boxes with face indices
    preview = tgt.copy()
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            preview, f"Face {i}",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    return bgr_to_rgb(preview), f"Detected {len(faces)} face(s). Select index 0-{len(faces)-1}."


def swap_specific_face(source_pil, target_pil, face_index, use_enhance):
    """Swap a specific face by index."""
    if source_pil is None or target_pil is None:
        raise gr.Error("Please upload both source and target images!")

    src = pil_to_cv2(source_pil)
    tgt = pil_to_cv2(target_pil)

    try:
        idx = int(face_index)
    except (ValueError, TypeError):
        idx = None

    result = swapper.swap_faces(src, tgt, face_index=idx)

    if use_enhance:
        result = enhancer.enhance(result)

    result_rgb = bgr_to_rgb(result)

    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"multiface_{timestamp}.png")
    cv2.imwrite(output_path, result)

    return result_rgb


# ── Video Swap Function ──────────────────────────────────
def swap_video(source_pil, video_path, use_enhance, progress=gr.Progress()):
    """Swap faces in a video file."""
    if source_pil is None:
        raise gr.Error("Please upload a source face image!")
    if video_path is None:
        raise gr.Error("Please upload a target video!")

    # Save source face to temp
    ensure_dir(TEMP_DIR)
    src_path = os.path.join(TEMP_DIR, "source_face.jpg")
    source_pil.save(src_path)

    # Generate output path
    basename = os.path.basename(video_path)
    name, ext = os.path.splitext(basename)
    timestamp = int(time.time())
    out_path = os.path.join(OUTPUT_DIR, f"video_{name}_{timestamp}.mp4")

    # Configure enhancer for video
    video_proc.enhancer = enhancer if use_enhance else None

    # Progress callback
    def update_progress(current, total):
        progress(current / total, desc=f"Frame {current}/{total}")

    # Process video
    video_proc.process_video(
        src_path, video_path, out_path,
        temp_dir=TEMP_DIR,
        progress_callback=update_progress
    )

    return out_path, f"Video saved: {out_path}"


# ── GIF Swap Function ────────────────────────────────────
def swap_gif(source_pil, gif_path, use_enhance):
    """Swap faces in a GIF file."""
    if source_pil is None:
        raise gr.Error("Please upload a source face image!")
    if gif_path is None:
        raise gr.Error("Please upload a target GIF!")

    ensure_dir(TEMP_DIR)
    src_path = os.path.join(TEMP_DIR, "source_face.jpg")
    source_pil.save(src_path)

    basename = os.path.basename(gif_path)
    name, _ = os.path.splitext(basename)
    timestamp = int(time.time())
    out_path = os.path.join(OUTPUT_DIR, f"gif_{name}_{timestamp}.gif")

    video_proc.enhancer = enhancer if use_enhance else None
    video_proc.process_gif(src_path, gif_path, out_path, temp_dir=TEMP_DIR)

    return out_path, f"GIF saved: {out_path}"


# ── Batch Processing Function ────────────────────────────
def batch_process(source_pil, target_files, use_enhance, progress=gr.Progress()):
    """Batch process multiple target images."""
    if source_pil is None:
        raise gr.Error("Please upload a source face image!")
    if not target_files:
        raise gr.Error("Please upload target images!")

    ensure_dir(TEMP_DIR)
    src_path = os.path.join(TEMP_DIR, "source_face.jpg")
    source_pil.save(src_path)
    src = cv2.imread(src_path)

    timestamp = int(time.time())
    batch_dir = os.path.join(OUTPUT_DIR, f"batch_{timestamp}")
    ensure_dir(batch_dir)

    results = []
    total = len(target_files)

    for i, target_file in enumerate(target_files):
        progress((i + 1) / total, desc=f"Processing {i+1}/{total}")
        try:
            tgt = cv2.imread(target_file.name)
            if tgt is None:
                continue

            result = swapper.swap_faces(src, tgt)
            if use_enhance:
                result = enhancer.enhance(result)

            basename = os.path.basename(target_file.name)
            out_path = os.path.join(batch_dir, f"swapped_{basename}")
            cv2.imwrite(out_path, result)
            results.append(out_path)
        except Exception as e:
            print(f"[SKIP] {target_file.name}: {e}")

    gallery = [bgr_to_rgb(cv2.imread(p)) for p in results]
    status = f"Batch complete: {len(results)}/{total} processed. Saved to: {batch_dir}"
    return gallery, status


# ── Build Gradio UI ──────────────────────────────────────
custom_css = """
.gradio-container { max-width: 1200px !important; }
.main-title { text-align: center; margin-bottom: 5px; }
.subtitle { text-align: center; color: #888; font-size: 0.9em; margin-bottom: 20px; }
"""

with gr.Blocks(
    theme=gr.themes.Default(primary_hue="blue"),
    title="Private FaceSwap Engine",
    css=custom_css
) as demo:

    gr.Markdown(
        "# Private FaceSwap Engine\n"
        "*100% local processing. Zero cloud uploads. Your machine, your rules.*",
        elem_classes="main-title"
    )

    with gr.Tabs():

        # ── TAB 1: PHOTO MODE ──
        with gr.Tab("Photo Swap", id="photo"):
            gr.Markdown("### Upload a source face and a target photo to swap faces.")
            with gr.Row():
                with gr.Column():
                    src_img = gr.Image(
                        label="Source Face (YOUR face)",
                        type="pil",
                        height=300
                    )
                with gr.Column():
                    tgt_img = gr.Image(
                        label="Target Photo (face to replace)",
                        type="pil",
                        height=300
                    )
                with gr.Column():
                    out_img = gr.Image(
                        label="Result",
                        type="numpy",
                        height=300
                    )

            with gr.Row():
                enhance_cb = gr.Checkbox(
                    label="Apply GFPGAN Enhancement",
                    value=True
                )
                upscale_slider = gr.Slider(
                    minimum=1, maximum=4, step=1, value=1,
                    label="Upscale Factor (1x = no upscale)"
                )

            swap_btn = gr.Button("SWAP FACES", variant="primary", size="lg")
            photo_status = gr.Textbox(label="Status", interactive=False)

            swap_btn.click(
                swap_photo,
                inputs=[src_img, tgt_img, enhance_cb, upscale_slider],
                outputs=[out_img, photo_status]
            )

        # ── TAB 2: MULTI-FACE MODE ──
        with gr.Tab("Multi-Face", id="multiface"):
            gr.Markdown(
                "### Detect multiple faces and swap a specific one.\n"
                "Upload a target image, detect faces, then choose which face to swap."
            )
            with gr.Row():
                with gr.Column():
                    mf_src = gr.Image(label="Source Face", type="pil", height=250)
                    mf_tgt = gr.Image(label="Target Photo", type="pil", height=250)
                with gr.Column():
                    mf_preview = gr.Image(label="Detected Faces Preview", type="numpy", height=300)
                    mf_status = gr.Textbox(label="Detection Status", interactive=False)
                with gr.Column():
                    mf_result = gr.Image(label="Result", type="numpy", height=300)

            with gr.Row():
                mf_detect_btn = gr.Button("Detect Faces", variant="secondary")
                mf_index = gr.Number(label="Face Index to Swap", value=0, precision=0)
                mf_enhance = gr.Checkbox(label="Enhance", value=True)
                mf_swap_btn = gr.Button("Swap Selected Face", variant="primary")

            mf_detect_btn.click(
                detect_faces_preview,
                inputs=[mf_tgt],
                outputs=[mf_preview, mf_status]
            )
            mf_swap_btn.click(
                swap_specific_face,
                inputs=[mf_src, mf_tgt, mf_index, mf_enhance],
                outputs=[mf_result]
            )

        # ── TAB 3: VIDEO MODE ──
        with gr.Tab("Video Swap", id="video"):
            gr.Markdown(
                "### Swap faces in a video file.\n"
                "Frame-by-frame processing with FFmpeg audio re-muxing."
            )
            with gr.Row():
                with gr.Column():
                    vsrc_img = gr.Image(label="Source Face", type="pil", height=250)
                with gr.Column():
                    vtgt_vid = gr.Video(label="Target Video")

            venh_cb = gr.Checkbox(
                label="Enhance Each Frame (slower but higher quality)",
                value=False
            )
            vswap_btn = gr.Button("START VIDEO SWAP", variant="primary", size="lg")

            with gr.Row():
                vout = gr.Video(label="Output Video")
                vstatus = gr.Textbox(label="Status", interactive=False)

            vswap_btn.click(
                swap_video,
                inputs=[vsrc_img, vtgt_vid, venh_cb],
                outputs=[vout, vstatus]
            )

        # ── TAB 4: GIF MODE ──
        with gr.Tab("GIF Swap", id="gif"):
            gr.Markdown("### Swap faces in a GIF. Preserves loop and timing.")
            with gr.Row():
                with gr.Column():
                    gsrc_img = gr.Image(label="Source Face", type="pil", height=250)
                with gr.Column():
                    gtgt_gif = gr.File(label="Target GIF", file_types=[".gif"])

            genh_cb = gr.Checkbox(label="Enhance Frames", value=False)
            gswap_btn = gr.Button("SWAP GIF", variant="primary")

            with gr.Row():
                gout = gr.File(label="Output GIF")
                gstatus = gr.Textbox(label="Status", interactive=False)

            gswap_btn.click(
                swap_gif,
                inputs=[gsrc_img, gtgt_gif, genh_cb],
                outputs=[gout, gstatus]
            )

        # ── TAB 5: BATCH MODE ──
        with gr.Tab("Batch Process", id="batch"):
            gr.Markdown(
                "### Process multiple target images at once.\n"
                "Upload a source face and multiple target images."
            )
            with gr.Row():
                with gr.Column():
                    bsrc_img = gr.Image(label="Source Face", type="pil", height=250)
                with gr.Column():
                    btgt_files = gr.File(
                        label="Target Images (select multiple)",
                        file_count="multiple",
                        file_types=["image"]
                    )

            benh_cb = gr.Checkbox(label="Enhance All Results", value=True)
            bswap_btn = gr.Button("START BATCH", variant="primary", size="lg")

            bgallery = gr.Gallery(label="Results", columns=4, height=400)
            bstatus = gr.Textbox(label="Status", interactive=False)

            bswap_btn.click(
                batch_process,
                inputs=[bsrc_img, btgt_files, benh_cb],
                outputs=[bgallery, bstatus]
            )

    # ── Footer ──
    gr.Markdown(
        "---\n"
        "*Private FaceSwap Engine v1.0 — For personal research use only. "
        "All processing is 100% local. No data leaves your machine.*"
    )


# ── Launch ────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",    # localhost ONLY — not exposed to network
        server_port=7860,
        share=False,                 # NEVER set True for privacy
        show_api=False
    )
