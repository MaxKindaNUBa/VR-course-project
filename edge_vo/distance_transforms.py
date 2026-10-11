"""
Distance Transform Module for Edge-Based Visual Odometry.

Implements:
1. Euclidean Distance Transform (L2) - Exact Euclidean metric.
2. Truncated Distance Transform (TDT) - Limits maximum distance to avoid outlier bias.
3. Chamfer Distance Transform (L1 / Quasi-Euclidean) - Linear/integer approximations.
4. Directional / Orientation-Aware Distance Transform - Associates distance with
   edge gradient orientations to enforce directional consistency (EdgeVO).
5. Multi-Scale Distance Field Pyramid - Multi-octave representation for coarse-to-fine optimization.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2


class DistanceTransformModule:
    """
    Implements multiple distance transform formulations utilized in edge-based VO.
    """

    @staticmethod
    def euclidean_distance_transform(
        binary_edges: np.ndarray,
        return_indices: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Calculates standard Euclidean Distance Transform (L2) where zero indicates edge.
        Delta(p) = min_{q in E} ||p - q||_2.

        Args:
            binary_edges: uint8 binary image (255 for edge, 0 for background).
            return_indices: Whether to return nearest edge coordinate indices (Voronoi sites).

        Returns:
            dist_map: float32 Euclidean distance map.
            nearest_idx: Optional int32 array (H, W, 2) storing (y, x) of closest edge pixel.
        """
        inverted = cv2.bitwise_not(binary_edges)

        if return_indices:
            dist_map, labels = cv2.distanceTransformWithLabels(
                inverted,
                distanceType=cv2.DIST_L2,
                maskSize=cv2.DIST_MASK_5,
                labelType=cv2.DIST_LABEL_PIXEL
            )
            return dist_map.astype(np.float32), labels
        else:
            dist_map = cv2.distanceTransform(
                inverted,
                distanceType=cv2.DIST_L2,
                maskSize=cv2.DIST_MASK_5
            )
            return dist_map.astype(np.float32), None

    @staticmethod
    def truncated_distance_transform(
        binary_edges: np.ndarray,
        max_dist_threshold: float = 15.0
    ) -> np.ndarray:
        """
        Truncated Distance Transform (TDT).
        Caps distance residual at a threshold tau to prevent distant outlier edges
        from corrupting the non-linear optimization basin (EEDVO / Yan et al.).

        Phi_tau(p) = min(Delta(p), tau)
        """
        dist_map, _ = DistanceTransformModule.euclidean_distance_transform(binary_edges)
        truncated = np.clip(dist_map, 0.0, max_dist_threshold)
        return truncated.astype(np.float32)

    @staticmethod
    def chamfer_distance_transform(
        binary_edges: np.ndarray,
        metric: str = "L1"
    ) -> np.ndarray:
        """
        Chamfer Distance Transform (L1 Manhattan or C-metric approximation).
        Provides efficient integer/linear approximations for embedded or high-speed VO.
        """
        inverted = cv2.bitwise_not(binary_edges)
        if metric.upper() == "L1":
            dist_type = cv2.DIST_L1
        elif metric.upper() == "C":
            dist_type = cv2.DIST_C
        else:
            dist_type = cv2.DIST_FAIR

        chamfer_map = cv2.distanceTransform(inverted, distanceType=dist_type, maskSize=3)
        return chamfer_map.astype(np.float32)

    @staticmethod
    def directional_distance_transform(
        binary_edges: np.ndarray,
        edge_orientations_rad: np.ndarray,
        max_dist: float = 20.0
    ) -> Dict[str, np.ndarray]:
        """
        Directional / Orientation-Aware Distance Field (EdgeVO Eq. 2-3).
        Associates each pixel's distance with the orientation of its nearest edge pixel.
        Enables angular consistency filtering: |dot(g_ref, g_curr)| > eta.

        Args:
            binary_edges: uint8 edge mask (255 = edge).
            edge_orientations_rad: float32 orientation in radians for edge pixels [-pi, pi].
            max_dist: Truncation threshold.

        Returns:
            Dict containing distance, nearest coordinate grids, and target orientation.
        """
        h, w = binary_edges.shape
        dist_map, labels = DistanceTransformModule.euclidean_distance_transform(
            binary_edges, return_indices=True
        )

        edge_ys, edge_xs = np.where(binary_edges > 0)
        if len(edge_ys) == 0:
            return {
                "distance": np.full((h, w), max_dist, dtype=np.float32),
                "nearest_y": np.zeros((h, w), dtype=np.int32),
                "nearest_x": np.zeros((h, w), dtype=np.int32),
                "target_orientation": np.zeros((h, w), dtype=np.float32),
            }

        label_to_y = np.zeros(labels.max() + 1, dtype=np.int32)
        label_to_x = np.zeros(labels.max() + 1, dtype=np.int32)
        label_to_ori = np.zeros(labels.max() + 1, dtype=np.float32)

        for y, x in zip(edge_ys, edge_xs):
            lbl = labels[y, x]
            label_to_y[lbl] = y
            label_to_x[lbl] = x
            label_to_ori[lbl] = edge_orientations_rad[y, x]

        nearest_y = label_to_y[labels]
        nearest_x = label_to_x[labels]
        target_ori = label_to_ori[labels]
        truncated_dist = np.clip(dist_map, 0.0, max_dist)

        return {
            "distance": truncated_dist.astype(np.float32),
            "nearest_y": nearest_y,
            "nearest_x": nearest_x,
            "target_orientation": target_ori,
        }

    @staticmethod
    def build_distance_field_pyramid(
        binary_edges: np.ndarray,
        num_levels: int = 3,
        scale_factor: float = 0.5,
        truncation_thresholds: Optional[List[float]] = None
    ) -> List[np.ndarray]:
        """
        Constructs a coarse-to-fine Distance Field Pyramid (EdgeVO & Yan et al.).
        At coarse levels, larger truncation bounds provide wide convergence basins;
        at fine levels, tighter bounds refine sub-pixel camera pose.
        """
        if truncation_thresholds is None:
            truncation_thresholds = [25.0, 15.0, 8.0]

        pyramid = []
        curr_edges = binary_edges.copy()

        for lvl in range(num_levels):
            tau = truncation_thresholds[min(lvl, len(truncation_thresholds) - 1)]
            dt_level = DistanceTransformModule.truncated_distance_transform(curr_edges, tau)
            pyramid.append(dt_level)

            if lvl < num_levels - 1:
                dilated = cv2.dilate(curr_edges, np.ones((2, 2), np.uint8))
                new_w = int(curr_edges.shape[1] * scale_factor)
                new_h = int(curr_edges.shape[0] * scale_factor)
                curr_edges = cv2.resize(dilated, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return pyramid
