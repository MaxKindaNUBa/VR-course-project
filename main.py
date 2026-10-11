"""
Main Entry Point for Edge-Based Visual Odometry & Scene Analysis.
"""

from edge_vo import (
    DistanceTransformModule,
    RGBEdgeDetectionModule,
    DepthEdgeDetectionModule,
    SceneStudyModule,
    LargeMovingObjectDetector,
    UnifiedEdgeVOAnalysisPipeline,
)


def main():
    print("=" * 70)
    print("EdgeVO Modular System Initialized Successfully")
    print("=" * 70)
    print("Available Modules:")
    print("  1. DistanceTransformModule (Euclidean, Truncated, Chamfer, Directional, Pyramid)")
    print("  2. RGBEdgeDetectionModule (Canny, Scharr/Sobel, LoG, Illum-Normalized)")
    print("  3. DepthEdgeDetectionModule (Step Discontinuity, Crease Normals, Rel-Step, Gating)")
    print("  4. SceneStudyModule (Entropy, Motion Degeneracy, Spatial Matroid Uniformity)")
    print("  5. LargeMovingObjectDetector (DT Residuals, Depth Inconsistency, Clustering)")
    print("  6. UnifiedEdgeVOAnalysisPipeline (End-to-End Orchestrator)")


if __name__ == "__main__":
    main()
