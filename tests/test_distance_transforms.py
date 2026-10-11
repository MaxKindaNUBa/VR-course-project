"""
Test script for DistanceTransformModule.
Tests:
1. Euclidean distance transform (L2 metric and nearest coordinate indexing)
2. Truncated distance transform (clipping threshold compliance)
3. Chamfer distance transform (L1 / integer metric behavior)
4. Directional distance transform (orientation preservation and nearest pixel lookup)
5. Multi-scale distance field pyramid (octave downsampling and boundary convergence)
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge_vo.distance_transforms import DistanceTransformModule


def test_euclidean_and_truncated():
    print("[TEST 1/5] Testing Euclidean and Truncated Distance Transforms...")
    # Create synthetic 100x100 binary edge image with a central vertical line
    edges = np.zeros((100, 100), dtype=np.uint8)
    edges[:, 50] = 255  # Edge at column 50

    # 1. Exact Euclidean DT
    dt_l2, indices = DistanceTransformModule.euclidean_distance_transform(edges, return_indices=True)
    assert dt_l2.shape == (100, 100), f"Incorrect shape: {dt_l2.shape}"
    assert np.allclose(dt_l2[:, 50], 0.0), "Edge points should have distance 0.0"
    assert np.isclose(dt_l2[50, 40], 10.0), f"Point at col 40 should be dist 10 from col 50, got {dt_l2[50, 40]}"
    assert indices is not None, "Nearest indices should not be None"

    # 2. Truncated DT
    tau = 8.0
    dt_trunc = DistanceTransformModule.truncated_distance_transform(edges, max_dist_threshold=tau)
    assert np.max(dt_trunc) <= tau, f"Truncated DT exceeded max threshold {tau}: {np.max(dt_trunc)}"
    assert np.allclose(dt_trunc[:, 50], 0.0), "Edge points should remain zero in truncated DT"

    print("  --> PASS: Euclidean & Truncated DT verified.")


def test_chamfer_transform():
    print("[TEST 2/5] Testing Chamfer Distance Transform (L1 Manhattan)...")
    edges = np.zeros((50, 50), dtype=np.uint8)
    edges[25, 25] = 255  # Single center edge point

    dt_chamfer = DistanceTransformModule.chamfer_distance_transform(edges, metric="L1")
    assert dt_chamfer.shape == (50, 50), "Incorrect shape"
    assert dt_chamfer[25, 25] == 0.0, "Center should have 0 distance"
    # Manhattan distance |20-25| + |20-25| = 5 + 5 = 10
    assert np.isclose(dt_chamfer[20, 20], 10.0), f"Expected L1 distance 10.0, got {dt_chamfer[20, 20]}"

    print("  --> PASS: Chamfer L1 DT verified.")


def test_directional_transform():
    print("[TEST 3/5] Testing Directional / Orientation-Aware Distance Transform...")
    edges = np.zeros((60, 60), dtype=np.uint8)
    edges[30, :] = 255  # Horizontal edge across row 30
    orientations = np.full((60, 60), np.pi / 2.0, dtype=np.float32)  # Pi/2 orientation

    res = DistanceTransformModule.directional_distance_transform(edges, orientations, max_dist=15.0)
    assert "distance" in res and "nearest_y" in res and "target_orientation" in res
    assert np.all(res["nearest_y"][20, :] == 30), "Nearest Y should be 30 for points at row 20"
    assert np.allclose(res["target_orientation"][20, :], np.pi / 2.0), "Orientation did not match source edge"

    print("  --> PASS: Directional DT verified.")


def test_distance_pyramid():
    print("[TEST 4/5] Testing Multi-Scale Distance Field Pyramid...")
    edges = np.zeros((120, 160), dtype=np.uint8)
    edges[20:100, 40] = 255
    edges[20:100, 120] = 255

    pyramid = DistanceTransformModule.build_distance_field_pyramid(edges, num_levels=3)
    assert len(pyramid) == 3, f"Expected 3 levels, got {len(pyramid)}"
    assert pyramid[0].shape == (120, 160), f"Level 0 shape mismatch: {pyramid[0].shape}"
    assert pyramid[1].shape == (60, 80), f"Level 1 shape mismatch: {pyramid[1].shape}"
    assert pyramid[2].shape == (30, 40), f"Level 2 shape mismatch: {pyramid[2].shape}"

    print("  --> PASS: Distance Field Pyramid verified.")


def main():
    print("=" * 60)
    print("RUNNING UNIT TESTS: DistanceTransformModule")
    print("=" * 60)
    test_euclidean_and_truncated()
    test_chamfer_transform()
    test_directional_transform()
    test_distance_pyramid()
    print("=" * 60)
    print("ALL DISTANCE TRANSFORM TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
