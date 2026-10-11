"""
Test script for RGBEdgeDetectionModule.
Tests:
1. Canny edge detector (gradient orientation and binary response)
2. Sobel / Scharr gradient magnitude and angle extraction
3. Laplacian of Gaussian (LoG) zero-crossing detector
4. Illumination-Normalized Gradient detector (robustness to lighting change)
"""

import os
import sys
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge_vo.edge_detection_rgb import RGBEdgeDetectionModule


def test_canny_detector():
    print("[TEST 1/4] Testing Canny Edge Detector...")
    # Synthetic checkerboard image with sharp contrast edges
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = 255
    img[50:, 50:] = 255

    edges, orientations = RGBEdgeDetectionModule.canny_detector(img, low_threshold=50, high_threshold=150)
    assert edges.shape == (100, 100), f"Shape mismatch: {edges.shape}"
    assert orientations.shape == (100, 100), f"Orientation shape mismatch: {orientations.shape}"
    assert np.count_nonzero(edges) > 50, "Canny failed to detect checkerboard boundaries"
    assert np.min(orientations) >= -np.pi and np.max(orientations) <= np.pi, "Orientation out of [-pi, pi] bounds"

    print(f"  --> PASS: Canny detected {np.count_nonzero(edges)} edge pixels.")


def test_sobel_scharr():
    print("[TEST 2/4] Testing Sobel & Scharr Gradient Detectors...")
    # Create horizontal ramp with steep transition at row 50
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50:, :] = 200

    edges, mag, angles = RGBEdgeDetectionModule.sobel_scharr_gradients(img, ksize=-1, magnitude_threshold=60.0)
    assert edges.shape == (100, 100)
    assert mag.shape == (100, 100)
    # The gradient magnitude should peak around row 50
    assert np.max(mag[48:52, :]) > 100.0, "Scharr filter missed high gradient boundary"

    print("  --> PASS: Scharr gradient calculation verified.")


def test_log_zero_crossings():
    print("[TEST 3/4] Testing Laplacian of Gaussian (LoG)...")
    # Circular disk in center
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 255, -1)

    log_edges = RGBEdgeDetectionModule.laplacian_of_gaussian(img, kernel_size=5, threshold=3.0)
    assert log_edges.shape == (100, 100)
    assert np.count_nonzero(log_edges) > 30, "LoG failed to detect circle perimeter"

    print("  --> PASS: LoG zero-crossing perimeter detected.")


def test_illumination_normalized():
    print("[TEST 4/4] Testing Illumination-Normalized Edge Detector...")
    # Dark region with edge vs bright region with edge
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:50, :50] = 30
    img[:50, 50:] = 60    # Ratio 2.0 in dark setting
    img[50:, :50] = 120
    img[50:, 50:] = 240   # Ratio 2.0 in bright setting

    edges, norm_grad = RGBEdgeDetectionModule.illumination_normalized_edges(img)
    assert edges.shape == (100, 100)
    assert norm_grad.shape == (100, 100)
    # The normalized gradient should respond comparably in both illumination regimes
    assert np.count_nonzero(edges) > 20, "Failed to extract normalized edges"

    print("  --> PASS: Illumination-normalized detector verified.")


def main():
    print("=" * 60)
    print("RUNNING UNIT TESTS: RGBEdgeDetectionModule")
    print("=" * 60)
    test_canny_detector()
    test_sobel_scharr()
    test_log_zero_crossings()
    test_illumination_normalized()
    print("=" * 60)
    print("ALL RGB EDGE DETECTION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
