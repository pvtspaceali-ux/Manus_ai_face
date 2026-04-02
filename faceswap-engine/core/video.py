# core/video.py — FFmpeg + Frame-by-Frame Video Processing
# Private FaceSwap Engine — Video Pipeline Module
# Extracts frames, applies face swap per frame, re-encodes with audio

import cv2
import os
import shutil
import subprocess
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, swapper, enhancer=None):
        """
        Initialize the video processor.

        Args:
            swapper: FaceSwapper instance
            enhancer: Optional FaceEnhancer instance for per-frame enhancement
        """
        self.swapper = swapper
        self.enhancer = enhancer

    def get_video_info(self, video_path):
        """
        Get video metadata (fps, frame count, resolution).

        Args:
            video_path: Path to video file

        Returns:
            Dict with fps, total_frames, width, height
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return info

    def extract_first_frame(self, video_path):
        """
        Extract the first frame of a video for preview/analysis.

        Args:
            video_path: Path to video file

        Returns:
            First frame as BGR numpy array
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read first frame from: {video_path}")
        return frame

    def has_audio(self, video_path):
        """Check if video has an audio stream."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    video_path
                ],
                capture_output=True, text=True, timeout=10
            )
            return "audio" in result.stdout
        except Exception:
            return False

    def process_video(self, source_path, video_path, output_path,
                      temp_dir="./temp", face_index=None,
                      progress_callback=None):
        """
        Process a video: swap faces frame-by-frame and re-encode.

        Args:
            source_path: Path to source face image
            video_path: Path to target video
            output_path: Path to save output video
            temp_dir: Temporary directory for frame extraction
            face_index: Optional specific face index to swap
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Path to output video
        """
        # Setup temp directories
        os.makedirs(temp_dir, exist_ok=True)
        swapped_dir = os.path.join(temp_dir, "swapped_frames")
        os.makedirs(swapped_dir, exist_ok=True)

        # Read source face image once
        source_img = cv2.imread(source_path)
        if source_img is None:
            raise FileNotFoundError(f"Source face not found: {source_path}")

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[VIDEO] {total} frames @ {fps:.1f}fps, {w}x{h}")

        # Process each frame
        frame_idx = 0
        swapped_count = 0
        skipped_count = 0

        pbar = tqdm(total=total, desc="Swapping frames", unit="frame")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                swapped = self.swapper.swap_faces(
                    source_img, frame, face_index=face_index
                )
                if self.enhancer:
                    swapped = self.enhancer.enhance(swapped)
                swapped_count += 1
            except (ValueError, Exception):
                # No face detected in this frame — keep original
                swapped = frame
                skipped_count += 1

            # Save frame
            frame_path = os.path.join(swapped_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(frame_path, swapped)
            frame_idx += 1
            pbar.update(1)

            # Progress callback for UI
            if progress_callback:
                progress_callback(frame_idx, total)

        cap.release()
        pbar.close()

        print(f"[VIDEO] Swapped: {swapped_count} | Skipped (no face): {skipped_count}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Re-encode video frames to video using FFmpeg
        temp_video = os.path.join(temp_dir, "temp_noaudio.mp4")

        encode_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(swapped_dir, "%06d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            temp_video
        ]

        print("[VIDEO] Encoding frames to video...")
        subprocess.run(encode_cmd, capture_output=True, check=True)

        # Mux audio from original video if it exists
        if self.has_audio(video_path):
            print("[VIDEO] Muxing original audio...")
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_path
            ]
            subprocess.run(mux_cmd, capture_output=True, check=True)
        else:
            # No audio — just rename
            shutil.move(temp_video, output_path)

        # Cleanup temp frames (ephemeral processing — privacy requirement)
        self._cleanup_temp(swapped_dir, temp_dir)

        print(f"[OK] Video saved: {output_path}")
        return output_path

    def process_gif(self, source_path, gif_path, output_path,
                    temp_dir="./temp", face_index=None):
        """
        Process a GIF: decompose frames, swap faces, reconstruct GIF.

        Args:
            source_path: Path to source face image
            gif_path: Path to target GIF
            output_path: Path to save output GIF
            temp_dir: Temporary directory
            face_index: Optional specific face index to swap

        Returns:
            Path to output GIF
        """
        os.makedirs(temp_dir, exist_ok=True)
        frames_dir = os.path.join(temp_dir, "gif_frames")
        swapped_dir = os.path.join(temp_dir, "gif_swapped")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(swapped_dir, exist_ok=True)

        # Extract GIF frames using FFmpeg (preserves timing)
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", gif_path,
            os.path.join(frames_dir, "%06d.png")
        ]
        subprocess.run(extract_cmd, capture_output=True, check=True)

        # Get GIF frame rate
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            gif_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        try:
            num, den = result.stdout.strip().split('/')
            gif_fps = float(num) / float(den)
        except Exception:
            gif_fps = 10  # default GIF fps

        # Read source face
        source_img = cv2.imread(source_path)
        if source_img is None:
            raise FileNotFoundError(f"Source face not found: {source_path}")

        # Process each frame
        frame_files = sorted([
            f for f in os.listdir(frames_dir) if f.endswith('.png')
        ])

        for filename in tqdm(frame_files, desc="Swapping GIF frames"):
            frame_path = os.path.join(frames_dir, filename)
            frame = cv2.imread(frame_path)

            try:
                swapped = self.swapper.swap_faces(
                    source_img, frame, face_index=face_index
                )
                if self.enhancer:
                    swapped = self.enhancer.enhance(swapped)
            except Exception:
                swapped = frame

            cv2.imwrite(os.path.join(swapped_dir, filename), swapped)

        # Reconstruct GIF
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        reconstruct_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(gif_fps),
            "-i", os.path.join(swapped_dir, "%06d.png"),
            "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0",
            output_path
        ]
        subprocess.run(reconstruct_cmd, capture_output=True, check=True)

        # Cleanup
        self._cleanup_temp(frames_dir, swapped_dir)

        print(f"[OK] GIF saved: {output_path}")
        return output_path

    def _cleanup_temp(self, *dirs):
        """
        Remove temporary directories (ephemeral processing).
        Privacy requirement: intermediate frames are auto-wiped.
        """
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
