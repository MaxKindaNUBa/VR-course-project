"""
Scene Study and Observability Analysis Module.

Implements:
1. Texture Density & Shannon Entropy Analysis - Distinguishes low-texture from textured scenes.
2. Motion Degeneracy & Structure Tensor Analysis - Detects aperture problems / ill-conditioned directions.
3. Stochastic Partition Matroid Metric (EdgeVO) - Evaluates spatial distribution uniformity.
4. Comprehensive Automated Scene Categorization.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2


@dataclass
class SceneAnalysisMetrics:
    """Structured report of scene complexity, texture density, and motion observability."""
    texture_entropy: float
    rgb_edge_density: float
    depth_valid_percentage: float
    structural_to_texture_ratio: float
    eigenvalue_ratio_degeneracy: float  # lambda_min / lambda_max of structure tensor
    is_textureless_scene: bool
    is_motion_degenerate: bool
    spatial_uniformity_score: float  # Matroid partition coverage score (EdgeVO)
    scene_classification: str


class SceneStudyModule:
    """
    Evaluates visual scene characteristics for visual odometry robustness.
    """

    @staticmethod
    def calculate_texture_entropy(gray_image: np.ndarray) -> float:
        """
        Computes 1D intensity Shannon entropy as a measure of visual information richness.
        """
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)

    @staticmethod
    def analyze_motion_degeneracy(
        edge_mask: np.ndarray,
        orientations: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Observability analysis for edge-based VO:
        Computes the 2x2 edge orientation structure tensor S = sum [cos(theta), sin(theta)]^T [cos(theta), sin(theta)].
        If condition ratio lambda_min / lambda_max << 0.1, camera motion orthogonal to the dominant
        edge direction cannot be observed (Aperture problem / Degeneracy).
        """
        valid_ys, valid_xs = np.where(edge_mask > 0)
        if len(valid_ys) < 50:
            return 0.0, np.array([0.0, 0.0])

        thetas = orientations[valid_ys, valid_xs]
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        c2 = np.sum(cos_t ** 2)
        s2 = np.sum(sin_t ** 2)
        cs = np.sum(cos_t * sin_t)

        tensor = np.array([[c2, cs], [cs, s2]], dtype=np.float64)
        eigvals = np.linalg.eigvalsh(tensor)
        lambda_min, lambda_max = float(eigvals[0]), float(eigvals[1])

        ratio = lambda_min / (lambda_max + 1e-8)
        return ratio, np.array([lambda_min, lambda_max])

    @staticmethod
    def evaluate_spatial_partition_uniformity(
        edge_mask: np.ndarray,
        grid_rows: int = 8,
        grid_cols: int = 8
    ) -> float:
        """
        Spatial partition matroid evaluation (EdgeVO Section III-B, Fig. 4).
        Divides the image into a grid of partitions {P_1, ..., P_k} and counts partition occupancy.
        Returns fraction of active grid cells (1.0 = perfectly well-distributed edges).
        """
        h, w = edge_mask.shape
        cell_h = h // grid_rows
        cell_w = w // grid_cols

        active_cells = 0
        total_cells = grid_rows * grid_cols

        for r in range(grid_rows):
            for c in range(grid_cols):
                patch = edge_mask[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                if np.count_nonzero(patch) > 10:
                    active_cells += 1

        return active_cells / total_cells

    @staticmethod
    def perform_full_scene_study(
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        rgb_edges: np.ndarray,
        structural_edges: np.ndarray,
        orientations: np.ndarray
    ) -> SceneAnalysisMetrics:
        """
        Integrates all scene metrics and returns a comprehensive classification.
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) if len(rgb_image.shape) == 3 else rgb_image
        total_pixels = float(gray.shape[0] * gray.shape[1])

        entropy = SceneStudyModule.calculate_texture_entropy(gray)
        edge_density = float(np.count_nonzero(rgb_edges)) / total_pixels

        valid_depth_count = float(np.count_nonzero(depth_image > 0))
        depth_valid_pct = valid_depth_count / total_pixels

        num_structural = float(np.count_nonzero(structural_edges))
        num_texture = float(np.count_nonzero(rgb_edges) - num_structural)
        struct_ratio = num_structural / (num_texture + 1e-5)

        deg_ratio, _ = SceneStudyModule.analyze_motion_degeneracy(rgb_edges, orientations)
        uniformity = SceneStudyModule.evaluate_spatial_partition_uniformity(rgb_edges)

        is_textureless = (entropy < 5.5) or (edge_density < 0.015)
        is_degenerate = (deg_ratio < 0.15)

        if is_textureless and is_degenerate:
            classification = "Challenging Texture-Less & Degenerate Corridor"
        elif is_textureless:
            classification = "Texture-Less Geometric Environment (Wall/Plane)"
        elif is_degenerate:
            classification = "Well-Textured Single-Directional (Aperture Degenerate)"
        else:
            classification = "Rich Multi-Directional 3D Environment (Optimal for EdgeVO)"

        return SceneAnalysisMetrics(
            texture_entropy=entropy,
            rgb_edge_density=edge_density,
            depth_valid_percentage=depth_valid_pct,
            structural_to_texture_ratio=struct_ratio,
            eigenvalue_ratio_degeneracy=deg_ratio,
            is_textureless_scene=is_textureless,
            is_motion_degenerate=is_degenerate,
            spatial_uniformity_score=uniformity,
            scene_classification=classification
        )
