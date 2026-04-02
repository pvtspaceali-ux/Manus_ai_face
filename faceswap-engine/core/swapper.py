# core/swapper.py — The Heart of the Tool
# Private FaceSwap Engine — Face Swap Core Module
# Uses InsightFace buffalo_l for detection + inswapper_128 for swapping

import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import os


class FaceSwapper:
    def __init__(self, model_path="./models/inswapper_128.onnx", use_gpu=True):
        """
        Initialize the FaceSwapper with face analysis and swap models.

        Args:
            model_path: Path to inswapper_128.onnx model file
            use_gpu: True for CUDA GPU acceleration, False for CPU
        """
        # ctx_id: 0 = GPU, -1 = CPU
        ctx_id = 0 if use_gpu else -1

        # Determine execution providers based on GPU availability
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load face analysis model (buffalo_l)
        # buffalo_l provides: RetinaFace detection + ArcFace recognition
        self.app = FaceAnalysis(
            name='buffalo_l',
            root='./models',
            providers=providers
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # Load the inswapper model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Swap model not found: {model_path}\n"
                f"Download from: https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx"
            )

        self.swapper = insightface.model_zoo.get_model(
            model_path,
            download=False,
            download_zip=False
        )
        self.use_gpu = use_gpu
        print(f"[OK] FaceSwapper loaded. GPU: {use_gpu}")

    def get_faces(self, img):
        """
        Detect all faces in an image.

        Args:
            img: BGR numpy array (OpenCV format)

        Returns:
            List of face objects sorted left-to-right by bounding box x-coordinate
        """
        faces = self.app.get(img)
        if not faces:
            raise ValueError("No face detected in image!")
        return sorted(faces, key=lambda x: x.bbox[0])  # sort left-to-right

    def get_face_thumbnails(self, img, size=112):
        """
        Get cropped face thumbnails for multi-face selection UI.

        Args:
            img: BGR numpy array
            size: Thumbnail size in pixels

        Returns:
            List of (face_object, thumbnail_bgr) tuples
        """
        faces = self.get_faces(img)
        thumbnails = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            # Add padding
            pad = int((x2 - x1) * 0.2)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.shape[1], x2 + pad)
            y2 = min(img.shape[0], y2 + pad)
            crop = img[y1:y2, x1:x2]
            thumb = cv2.resize(crop, (size, size))
            thumbnails.append((face, thumb))
        return thumbnails

    def swap_faces(self, source_img, target_img, face_index=None):
        """
        Swap face from source_img onto target_img.

        Args:
            source_img: BGR numpy array containing the source face
            target_img: BGR numpy array containing the target face(s)
            face_index: None = swap all faces, int = swap specific face index

        Returns:
            Swapped image (numpy array, BGR)
        """
        source_faces = self.get_faces(source_img)
        source_face = source_faces[0]  # always use first detected face as source

        target_faces = self.get_faces(target_img)
        result = target_img.copy()

        if face_index is not None:
            if face_index >= len(target_faces):
                raise IndexError(
                    f"Face index {face_index} out of range. "
                    f"Only {len(target_faces)} face(s) detected."
                )
            target_faces = [target_faces[face_index]]

        for face in target_faces:
            result = self.swapper.get(result, face, source_face, paste_back=True)

        return result

    def swap_from_paths(self, source_path, target_path, output_path, face_index=None):
        """
        High-level: read image files, swap faces, save result.

        Args:
            source_path: Path to source face image
            target_path: Path to target image
            output_path: Path to save the result
            face_index: Optional specific face index to swap

        Returns:
            Swapped image (numpy array, BGR)
        """
        src = cv2.imread(source_path)
        tgt = cv2.imread(target_path)

        if src is None:
            raise FileNotFoundError(f"Source not found: {source_path}")
        if tgt is None:
            raise FileNotFoundError(f"Target not found: {target_path}")

        result = self.swap_faces(src, tgt, face_index=face_index)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"[OK] Saved: {output_path}")
        return result

    def batch_swap(self, source_path, target_dir, output_dir, face_index=None):
        """
        Batch process: swap faces in all images in a directory.

        Args:
            source_path: Path to source face image
            target_dir: Directory containing target images
            output_dir: Directory to save results
            face_index: Optional specific face index to swap
        """
        os.makedirs(output_dir, exist_ok=True)
        src = cv2.imread(source_path)
        if src is None:
            raise FileNotFoundError(f"Source not found: {source_path}")

        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        files = [
            f for f in os.listdir(target_dir)
            if os.path.splitext(f)[1].lower() in supported_ext
        ]

        results = []
        for filename in sorted(files):
            target_path = os.path.join(target_dir, filename)
            output_path = os.path.join(output_dir, f"swapped_{filename}")
            try:
                tgt = cv2.imread(target_path)
                if tgt is None:
                    print(f"[SKIP] Cannot read: {filename}")
                    continue
                result = self.swap_faces(src, tgt, face_index=face_index)
                cv2.imwrite(output_path, result)
                results.append(output_path)
                print(f"[OK] {filename} -> swapped_{filename}")
            except ValueError as e:
                print(f"[SKIP] {filename}: {e}")
            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

        print(f"\n[DONE] Batch complete: {len(results)}/{len(files)} processed")
        return results
