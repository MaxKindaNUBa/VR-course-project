"""
Unified Pipeline Orchestrator for Edge-Based Visual Odometry Analysis.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from .distance_transforms import DistanceTransformModule
from .edge_detection_rgb import RGBEdgeDetectionModule
from .edge_detection_depth import DepthEdgeDetectionModule
from .scene_study import SceneStudyModule
from .dynamic_objects import LargeMovingObjectDetector


class UnifiedEdgeVOAnalysisPipeline:
    """
    Unified Orchestrator combining:
    - RGB & Depth Edge Detection
    - Cross-Bilateral Geometric Gating
    - Distance Transform & Orientation Maps
    - Scene Complexity & Observability Study
    - Large Dynamic Object Segmentation
    """

    def __init__(
        self,
        camera_intrinsics: Tuple[float, float, float, float] = (525.0, 525.0, 319.5, 239.5),  # TUM Default (fx, fy, cx, cy)
        dt_truncation: float = 15.0,
        dynamic_cluster_min_area: int = 1500
    ):
        self.intrinsics = camera_intrinsics
        self.dt_truncation = dt_truncation
        self.dynamic_detector = LargeMovingObjectDetector(
            dt_residual_threshold=6.0,
            min_cluster_area=dynamic_cluster_min_area
        )

    def process_rgbd_pair(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        prev_rgb_frame: Optional[np.ndarray] = None,
        prev_depth_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes complete edge, distance transform, scene, and dynamic object pipeline.
        """
        # 1. RGB Edge Extraction
        rgb_canny, orientations = RGBEdgeDetectionModule.canny_detector(rgb_frame, 35, 110)
        rgb_scharr, _, _ = RGBEdgeDetectionModule.sobel_scharr_gradients(rgb_frame)
        rgb_log = RGBEdgeDetectionModule.laplacian_of_gaussian(rgb_frame)
        rgb_norm, _ = RGBEdgeDetectionModule.illumination_normalized_edges(rgb_frame)

        # 2. Depth Edge Extraction
        d_step = DepthEdgeDetectionModule.depth_step_discontinuity_edges(depth_frame, 0.15)
        d_rel = DepthEdgeDetectionModule.relative_depth_gradient_edges(depth_frame, 0.08)
        d_crease, normals = DepthEdgeDetectionModule.depth_crease_edges_from_normals(
            depth_frame, self.intrinsics, normal_angle_threshold_deg=35.0
        )
        d_holes = DepthEdgeDetectionModule.sensor_boundary_hole_mask(depth_frame)

        # 3. Cross-Bilateral RGB-D Gating
        gated_edges = DepthEdgeDetectionModule.cross_bilateral_edge_fusion(
            rgb_edges=rgb_canny,
            depth_step_edges=d_step,
            depth_crease_edges=d_crease
        )

        # 4. Distance Transforms
        dt_euclid, _ = DistanceTransformModule.euclidean_distance_transform(rgb_canny)
        dt_truncated = DistanceTransformModule.truncated_distance_transform(rgb_canny, self.dt_truncation)
        dt_chamfer = DistanceTransformModule.chamfer_distance_transform(rgb_canny, "L1")
        dt_directional = DistanceTransformModule.directional_distance_transform(
            rgb_canny, orientations, self.dt_truncation
        )
        dt_pyramid = DistanceTransformModule.build_distance_field_pyramid(rgb_canny, num_levels=3)

        # 5. Scene Study & Observability
        scene_metrics = SceneStudyModule.perform_full_scene_study(
            rgb_image=rgb_frame,
            depth_image=depth_frame,
            rgb_edges=rgb_canny,
            structural_edges=gated_edges["structural_edges"],
            orientations=orientations
        )

        # 6. Large Moving Object Detection
        dynamic_results = None
        if prev_rgb_frame is not None:
            prev_canny, _ = RGBEdgeDetectionModule.canny_detector(prev_rgb_frame, 35, 110)
            dynamic_results = self.dynamic_detector.detect_large_moving_objects(
                ref_edges=prev_canny,
                curr_edges=rgb_canny,
                curr_dt_map=dt_truncated,
                prev_depth=prev_depth_frame,
                curr_depth=depth_frame
            )

        return {
            "rgb_edges": {
                "canny": rgb_canny,
                "scharr": rgb_scharr,
                "log": rgb_log,
                "illum_norm": rgb_norm,
                "orientations": orientations
            },
            "depth_edges": {
                "step": d_step,
                "relative_step": d_rel,
                "crease": d_crease,
                "hole_boundaries": d_holes,
                "normals": normals
            },
            "gated_edges": gated_edges,
            "distance_transforms": {
                "euclidean": dt_euclid,
                "truncated": dt_truncated,
                "chamfer": dt_chamfer,
                "directional": dt_directional,
                "pyramid": dt_pyramid
            },
            "scene_study": scene_metrics,
            "dynamic_detection": dynamic_results
        }
