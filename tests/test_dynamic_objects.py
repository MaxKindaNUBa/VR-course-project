"""
Test script for LargeMovingObjectDetector.
Tests:
1. Distance transform reprojection residual calculation
2. Inter-frame depth inconsistency detection
3. Morphological spatial clustering and bounding box extraction
4. Static inlier edge filtering
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge_vo.distance_transforms import DistanceTransformModule
from edge_vo.dynamic_objects import LargeMovingObjectDetector, DynamicObjectDetectionResult


def test_dynamic_detector():
    print("[TEST 1/2] Testing Large Moving Object Detection & Bounding Boxes...")
    # Frame 1: Background wall (x=20) and a moving person (box at x=60..100, y=40..80)
    ref_edges = np.zeros((120, 160), dtype=np.uint8)
    ref_edges[:, 20] = 255  # Static background edge
    ref_edges[40:80, 60:100] = 255  # Dynamic person at t=0

    # Frame 2: Static background edge stays at x=20, but person moved to x=80..120
    curr_edges = np.zeros((120, 160), dtype=np.uint8)
    curr_edges[:, 20] = 255  # Static background edge matches
    curr_edges[40:80, 80:120] = 255  # Person has translated

    # Current Distance Transform
    curr_dt = DistanceTransformModule.truncated_distance_transform(curr_edges, max_dist_threshold=20.0)

    detector = LargeMovingObjectDetector(
        dt_residual_threshold=5.0,
        min_cluster_area=500,
        cluster_dilation_kernel=11
    )

    result = detector.detect_large_moving_objects(
        ref_edges=ref_edges,
        curr_edges=curr_edges,
        curr_dt_map=curr_dt
    )

    assert isinstance(result, DynamicObjectDetectionResult)
    assert result.num_dynamic_clusters >= 1, "Failed to cluster moving dynamic object"
    assert len(result.bounding_boxes) >= 1, "Failed to extract bounding box for dynamic object"

    # Static edges should preserve background at x=20
    assert np.count_nonzero(result.static_edge_mask[:, 20]) > 80, "Static background edge was wrongly masked out"

    print(f"  --> PASS: Extracted {result.num_dynamic_clusters} dynamic cluster(s), BBoxes: {result.bounding_boxes}")


def test_depth_inconsistency():
    print("[TEST 2/2] Testing Depth Motion Inconsistency...")
    prev_d = np.full((100, 100), 2000, dtype=np.uint16)
    curr_d = np.full((100, 100), 2000, dtype=np.uint16)

    # Dynamic intrusion entering frame at depth 1.2m
    curr_d[30:70, 30:70] = 1200

    detector = LargeMovingObjectDetector()
    inconsistency = detector.compute_depth_motion_inconsistency(prev_d, curr_d, depth_threshold_meters=0.3)

    assert inconsistency.shape == (100, 100)
    assert np.count_nonzero(inconsistency[30:70, 30:70]) > 1000, "Failed to flag depth difference"

    print("  --> PASS: Depth motion inconsistency correctly flagged dynamic intrusion.")


def main():
    print("=" * 60)
    print("RUNNING UNIT TESTS: LargeMovingObjectDetector")
    print("=" * 60)
    test_dynamic_detector()
    test_depth_inconsistency()
    print("=" * 60)
    print("ALL DYNAMIC OBJECT DETECTION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
