# core/enhancer.py — GFPGAN Face Restoration & Enhancement
# Private FaceSwap Engine — Post-Processing Module
# Restores skin texture, sharpens eyes, fixes low-res artifacts from 128px swaps

import os
import cv2
import numpy as np
import torch


class FaceEnhancer:
    def __init__(self, model_path="./models/GFPGANv1.4.pth", upscale=1,
                 bg_upsampler_path=None):
        """
        Initialize the GFPGAN-based face enhancer.

        Args:
            model_path: Path to GFPGANv1.4.pth model file
            upscale: Upscale factor (1 = no upscale, 2 = 2x, 4 = 4x)
            bg_upsampler_path: Optional path to RealESRGAN model for background upscaling
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GFPGAN model not found: {model_path}\n"
                f"Download from: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
            )

        self.upscale = upscale
        self.bg_upsampler = None

        # Optionally load Real-ESRGAN for background upscaling
        if bg_upsampler_path and os.path.exists(bg_upsampler_path):
            self.bg_upsampler = self._load_realesrgan(bg_upsampler_path, upscale)

        # Import and initialize GFPGAN
        from gfpgan import GFPGANer

        self.enhancer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.bg_upsampler
        )
        print(f"[OK] FaceEnhancer loaded. Upscale: {upscale}x | BG Upsampler: {bg_upsampler_path is not None}")

    def _load_realesrgan(self, model_path, upscale):
        """Load Real-ESRGAN model for background upscaling."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            bg_upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if torch.cuda.is_available() else False
            )
            print(f"[OK] Real-ESRGAN background upsampler loaded")
            return bg_upsampler
        except Exception as e:
            print(f"[WARN] Could not load Real-ESRGAN: {e}")
            return None

    def enhance(self, img_bgr):
        """
        Enhance a face-swapped image using GFPGAN.

        Input: BGR numpy array (output of swapper)
        Output: Enhanced BGR numpy array

        GFPGAN restores:
        - Skin texture and pores
        - Eye sharpness and detail
        - Overall face clarity (128px -> appears 1024px)
        """
        try:
            _, _, enhanced = self.enhancer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            return enhanced if enhanced is not None else img_bgr
        except Exception as e:
            print(f"[WARN] Enhancement failed: {e}. Returning original image.")
            return img_bgr

    def enhance_from_path(self, input_path, output_path):
        """
        Enhance an image file and save the result.

        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image
        """
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_path}")

        enhanced = self.enhance(img)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, enhanced)
        print(f"[OK] Enhanced image saved: {output_path}")
        return enhanced

    def set_upscale(self, upscale):
        """
        Change the upscale factor dynamically.
        Note: Requires re-initialization of the enhancer.

        Args:
            upscale: New upscale factor (1, 2, or 4)
        """
        if upscale == self.upscale:
            return

        from gfpgan import GFPGANer

        self.upscale = upscale
        self.enhancer = GFPGANer(
            model_path=self.enhancer.model_path if hasattr(self.enhancer, 'model_path') else "./models/GFPGANv1.4.pth",
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.bg_upsampler
        )
        print(f"[OK] Upscale factor changed to {upscale}x")
