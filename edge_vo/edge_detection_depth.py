"""
Depth (D) Edge Detection Module for Visual Odometry.

Implements:
1. Depth Discontinuity (Step) Edges - Physical depth dropoffs (foreground/background separation).
2. Relative Depth Step Edges - Compensating for depth-dependent sensor noise (Delta Z / Z).
3. Depth Crease / Fold Edges - High normal angle change across adjacent 3D surfaces.
4. Sensor Missing Boundary / IR Hole Masker.
5. Cross-Bilateral RGB-D Gated Edge Fusion - Classifying 3D structural vs 2D flat texture edges.
"""

from typing import Dict, Tuple
import numpy as np
import cv2


class DepthEdgeDetectionModule:
    """
    Depth Edge Detection Suite:
    Isolates physical 3D scene geometry from optical surface texture.
    """

    @staticmethod
    def depth_step_discontinuity_edges(
        depth_map: np.ndarray,
        abs_threshold_meters: float = 0.15,
        depth_scale: float = 1000.0
    ) -> np.ndarray:
        """
        Detects step/jump discontinuities where physical depth changes sharply.
        """
        if depth_map.dtype in [np.uint16, np.int32]:
            z_m = depth_map.astype(np.float32) / depth_scale
        else:
            z_m = depth_map.copy().astype(np.float32)

        valid_mask = z_m > 0.05

        gx = np.zeros_like(z_m)
        gy = np.zeros_like(z_m)
        gx[:, 1:] = np.abs(z_m[:, 1:] - z_m[:, :-1])
        gy[1:, :] = np.abs(z_m[1:, :] - z_m[:-1, :])

        valid_pairs_x = valid_mask[:, 1:] & valid_mask[:, :-1]
        valid_pairs_y = valid_mask[1:, :] & valid_mask[:-1, :]
        gx[:, 1:][~valid_pairs_x] = 0
        gy[1:, :][~valid_pairs_y] = 0

        step_mag = np.maximum(gx, gy)
        step_edges = np.zeros_like(depth_map, dtype=np.uint8)
        step_edges[step_mag > abs_threshold_meters] = 255

        return step_edges

    @staticmethod
    def relative_depth_gradient_edges(
        depth_map: np.ndarray,
        rel_threshold: float = 0.08,
        depth_scale: float = 1000.0
    ) -> np.ndarray:
        """
        Detects relative depth step edges: Delta Z / Z > threshold.
        Compensates for sensor noise scaling quadratically with distance (sigma_z ~ Z^2).
        """
        z_m = depth_map.astype(np.float32) / depth_scale if depth_map.dtype == np.uint16 else depth_map.astype(np.float32)
        valid = z_m > 0.1

        gx = np.zeros_like(z_m)
        gy = np.zeros_like(z_m)
        gx[:, 1:] = np.abs(z_m[:, 1:] - z_m[:, :-1]) / np.maximum(z_m[:, 1:], 0.1)
        gy[1:, :] = np.abs(z_m[1:, :] - z_m[:-1, :]) / np.maximum(z_m[1:, :], 0.1)

        rel_mag = np.maximum(gx, gy)
        rel_mag[~valid] = 0

        edges = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        edges[rel_mag > rel_threshold] = 255
        return edges

    @staticmethod
    def depth_crease_edges_from_normals(
        depth_map: np.ndarray,
        camera_intrinsics: Tuple[float, float, float, float],  # fx, fy, cx, cy
        normal_angle_threshold_deg: float = 35.0,
        depth_scale: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects crease edges (internal corners / plane intersections) by computing
        surface normal differences between adjacent pixels in 3D.
        """
        fx, fy, cx, cy = camera_intrinsics
        z = depth_map.astype(np.float32) / depth_scale if depth_map.dtype == np.uint16 else depth_map.astype(np.float32)
        h, w = z.shape

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x_3d = (u - cx) * z / fx
        y_3d = (v - cy) * z / fy
        points_3d = np.stack([x_3d, y_3d, z], axis=-1)

        dx = np.zeros_like(points_3d)
        dy = np.zeros_like(points_3d)
        dx[:, 1:-1] = points_3d[:, 2:] - points_3d[:, :-2]
        dy[1:-1, :] = points_3d[2:, :] - points_3d[:-2, :]

        normals = np.cross(dx, dy)
        norm_len = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
        normals = normals / norm_len

        valid = (z > 0.1)
        normals[~valid] = 0.0

        dot_h = np.sum(normals[:, 1:] * normals[:, :-1], axis=-1)
        dot_v = np.sum(normals[1:, :] * normals[:-1, :], axis=-1)

        cos_thresh = np.cos(np.radians(normal_angle_threshold_deg))
        crease_mask = np.zeros((h, w), dtype=np.uint8)

        crease_h = (dot_h < cos_thresh) & (dot_h > -0.99) & valid[:, 1:] & valid[:, :-1]
        crease_v = (dot_v < cos_thresh) & (dot_v > -0.99) & valid[1:, :] & valid[:-1, :]

        crease_mask[:, 1:][crease_h] = 255
        crease_mask[1:, :][crease_v] = 255

        return crease_mask, normals

    @staticmethod
    def sensor_boundary_hole_mask(depth_map: np.ndarray) -> np.ndarray:
        """
        Extracts structural boundaries caused by missing depth / sensor holes.
        """
        missing = (depth_map == 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        boundary = cv2.morphologyEx(missing, cv2.MORPH_GRADIENT, kernel)
        return boundary

    @staticmethod
    def cross_bilateral_edge_fusion(
        rgb_edges: np.ndarray,
        depth_step_edges: np.ndarray,
        depth_crease_edges: np.ndarray,
        disparity_tolerance_kernel: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Fuses RGB and Depth edges:
        - Structural Edges: RGB edges verified by a physical depth step or crease.
        - Flat Texture Edges: RGB edges without physical depth jump (e.g., posters, floor patterns).
        - Geometric Depth-Only Edges: Discontinuities with weak photometric contrast.
        """
        combined_depth_edges = cv2.bitwise_or(depth_step_edges, depth_crease_edges)

        kernel = np.ones((disparity_tolerance_kernel, disparity_tolerance_kernel), np.uint8)
        depth_dilated = cv2.dilate(combined_depth_edges, kernel)

        structural_edges = cv2.bitwise_and(rgb_edges, depth_dilated)
        texture_edges = cv2.bitwise_and(rgb_edges, cv2.bitwise_not(depth_dilated))
        geometric_only = cv2.bitwise_and(combined_depth_edges, cv2.bitwise_not(cv2.dilate(rgb_edges, kernel)))

        return {
            "structural_edges": structural_edges,
            "texture_edges": texture_edges,
            "geometric_only_edges": geometric_only,
            "combined_geometry": combined_depth_edges,
        }
