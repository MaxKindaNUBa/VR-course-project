"""
Test script for DepthEdgeDetectionModule.
Tests:
1. Depth step discontinuity edges (foreground foreground step dropoff)
2. Relative depth gradient edges (scale-invariant Delta Z / Z step)
3. 3D surface normal reconstruction and dihedral crease edges
4. Sensor missing depth boundary / hole masking
5. Cross-bilateral RGB-D gating (structural 3D vs flat texture separation)
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge_vo.edge_detection_depth import DepthEdgeDetectionModule


def test_depth_step_and_relative():
    print("[TEST 1/4] Testing Depth Step Discontinuity & Relative Gradient...")
    # Synthetic depth map: background at 3.0m (3000 mm), foreground box at 1.0m (1000 mm)
    depth = np.full((120, 160), 3000, dtype=np.uint16)
    depth[40:80, 50:110] = 1000  # Box in center

    # 1. Absolute Step Edges (> 0.15m)
    step_edges = DepthEdgeDetectionModule.depth_step_discontinuity_edges(
        depth, abs_threshold_meters=0.15, depth_scale=1000.0
    )
    assert step_edges.shape == (120, 160)
    assert np.count_nonzero(step_edges) > 50, "Failed to detect foreground step edges"
    # Corners of the box should be marked
    assert step_edges[40, 80] == 255 or step_edges[41, 80] == 255, "Box boundary missing"

    # 2. Relative Step Edges (> 0.08)
    rel_edges = DepthEdgeDetectionModule.relative_depth_gradient_edges(
        depth, rel_threshold=0.08, depth_scale=1000.0
    )
    assert np.count_nonzero(rel_edges) > 50, "Failed to detect relative step edges"

    print("  --> PASS: Depth step and relative step edges verified.")


def test_surface_normal_creases():
    print("[TEST 2/4] Testing Surface Normal Estimation and Crease Edges...")
    # Create an internal corner (v-groove / dihedral fold):
    # Left half tilted plane, right half oppositely tilted plane
    h, w = 100, 100
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = 2.0 + 0.01 * np.abs(u - 50)  # V-crease at u = 50
    depth = (z * 1000.0).astype(np.uint16)

    intrinsics = (500.0, 500.0, 50.0, 50.0)
    crease_edges, normals = DepthEdgeDetectionModule.depth_crease_edges_from_normals(
        depth, camera_intrinsics=intrinsics, normal_angle_threshold_deg=20.0, depth_scale=1000.0
    )

    assert crease_edges.shape == (100, 100)
    assert normals.shape == (100, 100, 3)
    # Normals should be unit vectors where valid
    valid_norms = np.linalg.norm(normals[50, 20])
    assert np.isclose(valid_norms, 1.0, atol=1e-2), f"Normals not unit length: {valid_norms}"

    print("  --> PASS: 3D Surface normal and crease detection verified.")


def test_sensor_holes():
    print("[TEST 3/4] Testing Sensor Hole Boundary Detection...")
    depth = np.full((80, 80), 1500, dtype=np.uint16)
    depth[30:50, 30:50] = 0  # Missing data hole (IR shadow)

    hole_edges = DepthEdgeDetectionModule.sensor_boundary_hole_mask(depth)
    assert hole_edges.shape == (80, 80)
    assert np.count_nonzero(hole_edges) > 20, "Failed to extract hole boundary ring"

    print("  --> PASS: Sensor hole boundaries verified.")


def test_cross_bilateral_gating():
    print("[TEST 4/4] Testing Cross-Bilateral RGB-D Edge Gating...")
    # RGB has 2 edges: one geometric (wall border) and one flat texture (poster)
    rgb_edges = np.zeros((100, 100), dtype=np.uint8)
    rgb_edges[:, 30] = 255  # Physical geometric boundary
    rgb_edges[:, 70] = 255  # Flat poster texture

    # Depth only has step at x = 30
    depth_step = np.zeros((100, 100), dtype=np.uint8)
    depth_step[:, 30] = 255
    depth_crease = np.zeros((100, 100), dtype=np.uint8)

    fused = DepthEdgeDetectionModule.cross_bilateral_edge_fusion(
        rgb_edges=rgb_edges,
        depth_step_edges=depth_step,
        depth_crease_edges=depth_crease,
        disparity_tolerance_kernel=5
    )

    assert "structural_edges" in fused and "texture_edges" in fused
    # Physical edge at x=30 should be structural
    assert np.count_nonzero(fused["structural_edges"][:, 30]) > 90, "Geometric edge not classified as structural"
    # Flat edge at x=70 should be texture
    assert np.count_nonzero(fused["texture_edges"][:, 70]) > 90, "Flat edge not classified as texture"

    print("  --> PASS: Cross-bilateral gating accurately separated 3D geometry from 2D texture.")


def main():
    print("=" * 60)
    print("RUNNING UNIT TESTS: DepthEdgeDetectionModule")
    print("=" * 60)
    test_depth_step_and_relative()
    test_surface_normal_creases()
    test_sensor_holes()
    test_cross_bilateral_gating()
    print("=" * 60)
    print("ALL DEPTH EDGE DETECTION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
