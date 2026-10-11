"""
Large Moving Object Detection and Dynamic Masking Module.

Implements:
1. Distance Transform Edge Residuals - Detecting moving edges violating camera motion.
2. Depth Motion Inconsistency - Inter-frame depth difference checks.
3. Spatio-Temporal Morphological Clustering - Aggregating dynamic edge points into contiguous bodies.
4. Volumetric Filtering & Bounding Box Extraction - Isolating large dynamic objects (e.g., walking humans).
5. Static Inlier Edge Masking - Masking out dynamic foreground for visual odometry pose optimization.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class DynamicObjectDetectionResult:
    """Data container for dynamic object segmentation and static inlier filtering."""
    dynamic_mask: np.ndarray  # uint8 mask: 255 = dynamic foreground, 0 = static background
    static_edge_mask: np.ndarray  # Inlier edges eligible for visual odometry tracking
    bounding_boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h) of detected dynamic objects
    residual_map: np.ndarray  # Reprojection / distance transform residual error map
    num_dynamic_clusters: int


class LargeMovingObjectDetector:
    """
    Detects and segments large moving dynamic objects in RGB-D streams.
    """

    def __init__(
        self,
        dt_residual_threshold: float = 6.0,
        min_cluster_area: int = 1200,
        cluster_dilation_kernel: int = 15
    ):
        self.dt_residual_threshold = dt_residual_threshold
        self.min_cluster_area = min_cluster_area
        self.cluster_dilation_kernel = cluster_dilation_kernel

    def compute_edge_reprojection_residuals(
        self,
        ref_edges: np.ndarray,
        curr_dt_map: np.ndarray,
        estimated_flow_u: Optional[np.ndarray] = None,
        estimated_flow_v: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes edge distance residual map r_e(p) = Delta_c(omega(p, d, xi)).
        """
        h, w = ref_edges.shape
        residual_map = np.zeros((h, w), dtype=np.float32)

        edge_ys, edge_xs = np.where(ref_edges > 0)
        if len(edge_ys) == 0:
            return residual_map

        if estimated_flow_u is not None and estimated_flow_v is not None:
            warped_xs = np.clip(np.round(edge_xs + estimated_flow_u[edge_ys, edge_xs]).astype(np.int32), 0, w - 1)
            warped_ys = np.clip(np.round(edge_ys + estimated_flow_v[edge_ys, edge_xs]).astype(np.int32), 0, h - 1)
            residuals = curr_dt_map[warped_ys, warped_xs]
        else:
            residuals = curr_dt_map[edge_ys, edge_xs]

        residual_map[edge_ys, edge_xs] = residuals
        return residual_map

    def compute_depth_motion_inconsistency(
        self,
        prev_depth: np.ndarray,
        curr_depth: np.ndarray,
        depth_threshold_meters: float = 0.25,
        depth_scale: float = 1000.0
    ) -> np.ndarray:
        """
        Detects dynamic motion via depth inconsistency |Z_curr - Z_prev| > epsilon_z.
        """
        z_prev = prev_depth.astype(np.float32) / depth_scale
        z_curr = curr_depth.astype(np.float32) / depth_scale

        valid = (z_prev > 0.1) & (z_curr > 0.1)
        depth_diff = np.abs(z_curr - z_prev)
        depth_diff[~valid] = 0.0

        dynamic_depth = np.zeros_like(depth_diff, dtype=np.uint8)
        dynamic_depth[depth_diff > depth_threshold_meters] = 255
        return dynamic_depth

    def detect_large_moving_objects(
        self,
        ref_edges: np.ndarray,
        curr_edges: np.ndarray,
        curr_dt_map: np.ndarray,
        prev_depth: Optional[np.ndarray] = None,
        curr_depth: Optional[np.ndarray] = None,
        estimated_flow_u: Optional[np.ndarray] = None,
        estimated_flow_v: Optional[np.ndarray] = None
    ) -> DynamicObjectDetectionResult:
        """
        Full dynamic object detection pipeline:
        Aggregates residuals, clusters high-residual regions, filters large components,
        and generates a clean static inlier edge set for visual odometry.
        """
        h, w = ref_edges.shape

        dt_residuals = self.compute_edge_reprojection_residuals(
            ref_edges, curr_dt_map, estimated_flow_u, estimated_flow_v
        )

        high_residual_edges = np.zeros((h, w), dtype=np.uint8)
        high_residual_edges[dt_residuals > self.dt_residual_threshold] = 255

        if prev_depth is not None and curr_depth is not None:
            depth_inconsistency = self.compute_depth_motion_inconsistency(prev_depth, curr_depth)
            combined_dynamic_seeds = cv2.bitwise_or(high_residual_edges, depth_inconsistency)
        else:
            combined_dynamic_seeds = high_residual_edges

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cluster_dilation_kernel, self.cluster_dilation_kernel)
        )
        dilated_dynamic = cv2.morphologyEx(combined_dynamic_seeds, cv2.MORPH_CLOSE, kernel)
        dilated_dynamic = cv2.dilate(dilated_dynamic, kernel, iterations=2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_dynamic)

        dynamic_mask = np.zeros((h, w), dtype=np.uint8)
        bounding_boxes = []

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= self.min_cluster_area:
                bx = stats[label_id, cv2.CC_STAT_LEFT]
                by = stats[label_id, cv2.CC_STAT_TOP]
                bw = stats[label_id, cv2.CC_STAT_WIDTH]
                bh = stats[label_id, cv2.CC_STAT_HEIGHT]

                dynamic_mask[labels == label_id] = 255
                bounding_boxes.append((bx, by, bw, bh))

        static_edges = cv2.bitwise_and(curr_edges, cv2.bitwise_not(dynamic_mask))

        return DynamicObjectDetectionResult(
            dynamic_mask=dynamic_mask,
            static_edge_mask=static_edges,
            bounding_boxes=bounding_boxes,
            residual_map=dt_residuals,
            num_dynamic_clusters=len(bounding_boxes)
        )
