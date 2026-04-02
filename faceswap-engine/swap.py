# swap.py — Command-Line Face Swap Tool
# Private FaceSwap Engine — CLI Interface
#
# Usage:
#   python swap.py -s source.jpg -t target.jpg                    # basic swap
#   python swap.py -s source.jpg -t target.jpg -e                 # with enhancement
#   python swap.py -s source.jpg -t target.jpg -e --upscale 2     # with 2x upscale
#   python swap.py -s source.jpg -t target_video.mp4              # video swap
#   python swap.py -s source.jpg --batch ./input/target/          # batch mode

import argparse
import os
import sys
import time

from core.swapper import FaceSwapper
from core.enhancer import FaceEnhancer
from core.video import VideoProcessor
from core.utils import (
    is_image, is_video, is_gif, ensure_dir,
    get_output_path, run_system_check
)


def main():
    parser = argparse.ArgumentParser(
        description="Private FaceSwap Engine — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python swap.py -s face.jpg -t target.jpg
  python swap.py -s face.jpg -t target.jpg -e --upscale 2
  python swap.py -s face.jpg -t video.mp4 -e
  python swap.py -s face.jpg --batch ./input/target/
  python swap.py --check
        """
    )

    parser.add_argument("-s", "--source", type=str, help="Path to source face image")
    parser.add_argument("-t", "--target", type=str, help="Path to target image/video/GIF")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    parser.add_argument("-e", "--enhance", action="store_true", help="Apply GFPGAN enhancement")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2, 4], help="Upscale factor (1/2/4)")
    parser.add_argument("--face-index", type=int, default=None, help="Specific face index to swap")
    parser.add_argument("--batch", type=str, default=None, help="Batch mode: path to target directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (no GPU)")
    parser.add_argument("--check", action="store_true", help="Run system check and exit")
    parser.add_argument("--model-dir", type=str, default="./models", help="Models directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")

    args = parser.parse_args()

    # System check mode
    if args.check:
        run_system_check()
        return

    # Validate inputs
    if not args.source:
        parser.error("--source (-s) is required")

    if not args.target and not args.batch:
        parser.error("--target (-t) or --batch is required")

    if not os.path.exists(args.source):
        print(f"[ERROR] Source file not found: {args.source}")
        sys.exit(1)

    # Initialize models
    use_gpu = not args.cpu
    swap_model = os.path.join(args.model_dir, "inswapper_128.onnx")
    gfpgan_model = os.path.join(args.model_dir, "GFPGANv1.4.pth")
    realesrgan_model = os.path.join(args.model_dir, "RealESRGAN_x4plus.pth")

    print("\n[INIT] Loading models...")
    swapper = FaceSwapper(model_path=swap_model, use_gpu=use_gpu)

    enhancer = None
    if args.enhance:
        bg_path = realesrgan_model if os.path.exists(realesrgan_model) else None
        enhancer = FaceEnhancer(
            model_path=gfpgan_model,
            upscale=args.upscale,
            bg_upsampler_path=bg_path
        )

    ensure_dir(args.output_dir)

    # ── Batch Mode ──
    if args.batch:
        if not os.path.isdir(args.batch):
            print(f"[ERROR] Batch directory not found: {args.batch}")
            sys.exit(1)

        print(f"\n[BATCH] Processing all images in: {args.batch}")
        start = time.time()
        results = swapper.batch_swap(
            args.source, args.batch, args.output_dir,
            face_index=args.face_index
        )

        # Enhance batch results if requested
        if enhancer and results:
            print("\n[ENHANCE] Enhancing batch results...")
            for path in results:
                enhancer.enhance_from_path(path, path)

        elapsed = time.time() - start
        print(f"\n[DONE] Batch complete in {elapsed:.1f}s")
        return

    # ── Single File Mode ──
    target = args.target
    if not os.path.exists(target):
        print(f"[ERROR] Target file not found: {target}")
        sys.exit(1)

    start = time.time()

    if is_image(target):
        # Image swap
        output = args.output or get_output_path(target, args.output_dir)
        print(f"\n[SWAP] Photo: {target}")
        result = swapper.swap_from_paths(
            args.source, target, output,
            face_index=args.face_index
        )
        if enhancer:
            print("[ENHANCE] Applying GFPGAN...")
            enhancer.enhance_from_path(output, output)

    elif is_video(target):
        # Video swap
        output = args.output or get_output_path(target, args.output_dir, prefix="swapped_")
        if not output.endswith('.mp4'):
            output = os.path.splitext(output)[0] + '.mp4'

        print(f"\n[SWAP] Video: {target}")
        video_proc = VideoProcessor(swapper, enhancer)
        video_proc.process_video(
            args.source, target, output,
            face_index=args.face_index
        )

    elif is_gif(target):
        # GIF swap
        output = args.output or get_output_path(target, args.output_dir)
        print(f"\n[SWAP] GIF: {target}")
        video_proc = VideoProcessor(swapper, enhancer)
        video_proc.process_gif(args.source, target, output)

    else:
        print(f"[ERROR] Unsupported file format: {target}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n[DONE] Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
