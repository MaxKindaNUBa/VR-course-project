"""
Edge-Based Visual Odometry (EdgeVO) and Hybrid RGB-D Analysis Framework.

Modular implementation based on:
- Yan et al. (2022) "Robust and efficient edge-based visual odometry"
- Wang et al. "Edge Enhanced Direct Visual Odometry" (EEDVO)
- Zhao et al. (2023) "EdgeVO: An Efficient and Accurate Edge-based Visual Odometry"
- TUM Dynamic Benchmark & Literature on Dynamic Moving Object Rejection
"""

from .distance_transforms import DistanceTransformModule
from .edge_detection_rgb import RGBEdgeDetectionModule
from .edge_detection_depth import DepthEdgeDetectionModule
from .scene_study import SceneStudyModule, SceneAnalysisMetrics
from .dynamic_objects import LargeMovingObjectDetector, DynamicObjectDetectionResult
from .pipeline import UnifiedEdgeVOAnalysisPipeline

__all__ = [
    "DistanceTransformModule",
    "RGBEdgeDetectionModule",
    "DepthEdgeDetectionModule",
    "SceneStudyModule",
    "SceneAnalysisMetrics",
    "LargeMovingObjectDetector",
    "DynamicObjectDetectionResult",
    "UnifiedEdgeVOAnalysisPipeline",
]
