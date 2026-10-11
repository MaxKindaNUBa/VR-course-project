"""
RGB Edge Detection Module for Visual Odometry.

Implements:
1. Canny edge detector with bilateral/Gaussian filtering.
2. Sobel / Scharr gradient magnitude & orientation.
3. Laplacian of Gaussian (LoG) zero-crossing edge detector.
4. Illumination-Normalized Gradient (Yan et al.) - Robust to illumination changes.
"""

from typing import Tuple
import numpy as np
import cv2


class RGBEdgeDetectionModule:
    """
    RGB Edge Detection Suite:
    Extracts photometric discontinuities and gradients from RGB/grayscale imagery.
    """

    @staticmethod
    def canny_detector(
        rgb_image: np.ndarray,
        low_threshold: int = 40,
        high_threshold: int = 120,
        use_bilateral: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Canny edge detection on RGB image.
        Returns binary edge map and gradient orientation map [-pi, pi].
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image.copy()

        if use_bilateral:
            filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=5)
        else:
            filtered = cv2.GaussianBlur(gray, (5, 5), 1.2)

        edges = cv2.Canny(filtered, low_threshold, high_threshold)

        # Gradient orientation for angular consistency checks
        gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
        orientation = np.arctan2(gy, gx)

        return edges, orientation

    @staticmethod
    def sobel_scharr_gradients(
        rgb_image: np.ndarray,
        ksize: int = -1,  # -1 triggers Scharr operator
        magnitude_threshold: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gradient magnitude and thresholded edges using Sobel/Scharr operators.
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image.copy()

        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)

        if ksize == -1:
            gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
        else:
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=ksize)

        mag = cv2.magnitude(gx, gy)
        binary_edges = np.zeros_like(gray, dtype=np.uint8)
        binary_edges[mag > magnitude_threshold] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        thinned = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, kernel)

        return thinned, mag, np.arctan2(gy, gx)

    @staticmethod
    def laplacian_of_gaussian(
        rgb_image: np.ndarray,
        kernel_size: int = 5,
        threshold: float = 4.0
    ) -> np.ndarray:
        """
        LoG (Laplacian of Gaussian) edge detector based on second-derivative zero crossings.
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image.copy()

        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1.4)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)

        h, w = gray.shape
        zero_crossings = np.zeros((h, w), dtype=np.uint8)
        lap_p = lap[:-1, :-1]
        lap_right = lap[:-1, 1:]
        lap_down = lap[1:, :-1]

        cond_h = (lap_p * lap_right < 0) & (np.abs(lap_p - lap_right) > threshold)
        cond_v = (lap_p * lap_down < 0) & (np.abs(lap_p - lap_down) > threshold)
        zero_crossings[:-1, :-1][cond_h | cond_v] = 255

        return zero_crossings

    @staticmethod
    def illumination_normalized_edges(
        rgb_image: np.ndarray,
        gradient_quantile: float = 0.85
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Illumination-normalized edge detector (Yan et al., Section 4.1).
        g_norm(p) = ||grad I(p)|| / (mean_local(I(p)) + epsilon)
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = rgb_image.astype(np.float32)

        local_mean = cv2.blur(gray, (15, 15)) + 1e-4
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)

        norm_grad = mag / local_mean
        thresh = float(np.quantile(norm_grad, gradient_quantile))
        edges = np.zeros_like(norm_grad, dtype=np.uint8)
        edges[norm_grad > thresh] = 255

        return edges, norm_grad
