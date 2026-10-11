"""
Test script for SceneStudyModule.
Tests:
1. Shannon entropy calculation on textured vs homogeneous images
2. Motion degeneracy & structure tensor condition ratio (aperture problem)
3. EdgeVO spatial partition matroid uniformity score
4. Comprehensive scene classification pipeline
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge_vo.scene_study import SceneStudyModule, SceneAnalysisMetrics


def test_entropy_and_texture():
    print("[TEST 1/3] Testing Texture Entropy & Information Measure...")
    # 1. Homogeneous blank image (low entropy)
    blank = np.full((100, 100), 128, dtype=np.uint8)
    entropy_blank = SceneStudyModule.calculate_texture_entropy(blank)
    assert np.isclose(entropy_blank, 0.0), f"Blank image entropy should be 0.0, got {entropy_blank}"

    # 2. Rich noisy/textured image (high entropy)
    textured = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    entropy_rich = SceneStudyModule.calculate_texture_entropy(textured)
    assert entropy_rich > 6.5, f"Expected high entropy > 6.5, got {entropy_rich}"

    print(f"  --> PASS: Blank entropy = {entropy_blank:.2f}, Rich entropy = {entropy_rich:.2f}")


def test_motion_degeneracy():
    print("[TEST 2/3] Testing Observability & Structure Tensor Degeneracy...")
    # Degenerate case: strictly vertical parallel edges (corridor effect)
    deg_edges = np.zeros((100, 100), dtype=np.uint8)
    deg_edges[:, 20:80:10] = 255
    deg_orientations = np.zeros((100, 100), dtype=np.float32)  # All gradients horizontal (0 rad)

    ratio_deg, _ = SceneStudyModule.analyze_motion_degeneracy(deg_edges, deg_orientations)
    assert ratio_deg < 0.05, f"Single direction edges should be degenerate (<0.05), got {ratio_deg}"

    # Multi-directional well-conditioned edges (box / grid)
    box_edges = np.zeros((100, 100), dtype=np.uint8)
    box_edges[20, 20:80] = 255
    box_edges[80, 20:80] = 255
    box_edges[20:80, 20] = 255
    box_edges[20:80, 80] = 255

    box_orientations = np.zeros((100, 100), dtype=np.float32)
    box_orientations[20, 20:80] = np.pi / 2.0  # Horizontal lines have vertical gradient
    box_orientations[80, 20:80] = np.pi / 2.0
    box_orientations[20:80, 20] = 0.0          # Vertical lines have horizontal gradient
    box_orientations[20:80, 80] = 0.0

    ratio_box, _ = SceneStudyModule.analyze_motion_degeneracy(box_edges, box_orientations)
    assert ratio_box > 0.5, f"Grid box should have well-conditioned ratio (>0.5), got {ratio_box}"

    print(f"  --> PASS: Degenerate ratio = {ratio_deg:.4f}, Well-conditioned ratio = {ratio_box:.4f}")


def test_full_scene_study():
    print("[TEST 3/3] Testing Full Automated Scene Study Classification...")
    rgb = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)
    depth = np.full((120, 160), 2000, dtype=np.uint16)

    edges = np.zeros((120, 160), dtype=np.uint8)
    edges[30:90, 40:120] = 255
    structural = edges.copy()
    orientations = np.random.uniform(-np.pi, np.pi, (120, 160)).astype(np.float32)

    report = SceneStudyModule.perform_full_scene_study(
        rgb_image=rgb,
        depth_image=depth,
        rgb_edges=edges,
        structural_edges=structural,
        orientations=orientations
    )

    assert isinstance(report, SceneAnalysisMetrics)
    assert report.depth_valid_percentage == 1.0
    assert report.spatial_uniformity_score > 0.0
    assert len(report.scene_classification) > 0

    print(f"  --> PASS: Classified Scene as '{report.scene_classification}'")


def main():
    print("=" * 60)
    print("RUNNING UNIT TESTS: SceneStudyModule")
    print("=" * 60)
    test_entropy_and_texture()
    test_motion_degeneracy()
    test_full_scene_study()
    print("=" * 60)
    print("ALL SCENE STUDY TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
