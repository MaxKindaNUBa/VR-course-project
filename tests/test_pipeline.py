"""
Test script for UnifiedEdgeVOAnalysisPipeline.
Tests:
1. End-to-end execution of the complete pipeline on synthetic RGB-D frame pairs
2. Integration across edge detection, distance transforms, scene study, and dynamic segmentation
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge_vo.pipeline import UnifiedEdgeVOAnalysisPipeline


def test_pipeline_synthetic_frames():
    print("[TEST 1/1] Testing Unified Pipeline on Synthetic RGB-D Frame Pair...")
    h, w = 240, 320

    # Frame 0 (reference frame)
    rgb_0 = np.full((h, w, 3), 180, dtype=np.uint8)
    # Draw geometric structure
    rgb_0[60:180, 80:240] = 50
    depth_0 = np.full((h, w), 2500, dtype=np.uint16)
    depth_0[60:180, 80:240] = 1200

    # Frame 1 (current frame) with slight translation and dynamic object movement
    rgb_1 = np.full((h, w, 3), 180, dtype=np.uint8)
    rgb_1[60:180, 85:245] = 50
    # Dynamic foreground block moving across
    rgb_1[120:170, 200:260] = 20
    depth_1 = np.full((h, w), 2500, dtype=np.uint16)
    depth_1[60:180, 85:245] = 1200
    depth_1[120:170, 200:260] = 900

    pipeline = UnifiedEdgeVOAnalysisPipeline(
        camera_intrinsics=(320.0, 320.0, 160.0, 120.0),
        dt_truncation=12.0,
        dynamic_cluster_min_area=300
    )

    outputs = pipeline.process_rgbd_pair(
        rgb_frame=rgb_1,
        depth_frame=depth_1,
        prev_rgb_frame=rgb_0,
        prev_depth_frame=depth_0
    )

    # Validate output dictionary contracts
    assert "rgb_edges" in outputs
    assert "depth_edges" in outputs
    assert "gated_edges" in outputs
    assert "distance_transforms" in outputs
    assert "scene_study" in outputs
    assert "dynamic_detection" in outputs

    # Check structural components
    assert outputs["rgb_edges"]["canny"].shape == (h, w)
    assert outputs["depth_edges"]["step"].shape == (h, w)
    assert outputs["distance_transforms"]["truncated"].shape == (h, w)
    assert outputs["dynamic_detection"] is not None

    print(f"  --> Scene Classification: {outputs['scene_study'].scene_classification}")
    print(f"  --> Dynamic Clusters Detected: {outputs['dynamic_detection'].num_dynamic_clusters}")
    print("  --> PASS: Full pipeline processed RGB-D input pair successfully.")


def main():
    print("=" * 60)
    print("RUNNING UNIT TESTS: UnifiedEdgeVOAnalysisPipeline")
    print("=" * 60)
    test_pipeline_synthetic_frames()
    print("=" * 60)
    print("ALL PIPELINE INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
