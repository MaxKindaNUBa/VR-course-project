"""
Preliminary Edge-Based Visual Odometry, Distance Transforms, Scene Analysis,
and Large Moving Object Detection Pipeline.

Synthesized from research papers:
1. "Robust and efficient edge-based visual odometry" (Yan, Li, Zhou, 2022)
   - 3D-2D edge alignment using geometric reprojection error over Distance Transform (DT) maps.
   - Illumination-invariant edge residuals vs direct photometric error.
   - Coarse-to-fine pyramid registration and dynamic weighting.
2. "Edge Enhanced Direct Visual Odometry" (EEDVO, Wang et al.)
   - Hybrid energy combining direct photometric error and edge distance fields.
   - Distance field Delta(p) = min_{q in E} ||p - q||.
   - Outlier edge rejection based on mean distance error thresholds.
3. "EdgeVO: An Efficient and Accurate Edge-based Visual Odometry" (Zhao et al., 2023)
   - Distance field pyramid and gradient direction consistency: g(p) . g(q) > eta.
   - Stochastic Partition Greedy edge selection over partition matroids for uniform spatial distribution.
   - Least-squares covariance and Hessian conditioning: H = J^T J.
4. TUM Dynamic Benchmark ("walking_xyz") & Literature on Dynamic VO:
   - Segmentation and masking of large moving dynamic objects (e.g., walking humans).
   - Geometric depth discontinuity clustering and photometric/DT residual gating.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2


# ==============================================================================
# SECTION 1: DISTANCE TRANSFORMS (DT)
# ==============================================================================

class DistanceTransformModule:
    """
    Implements multiple distance transform formulations utilized in edge-based VO:
    1. Euclidean Distance Transform (L2) - Exact Euclidean metric.
    2. Truncated Distance Transform (TDT) - Limits maximum distance to avoid outlier bias.
    3. Chamfer Distance Transform (L1 / Quasi-Euclidean) - Linear/integer approximations.
    4. Directional / Orientation-Aware Distance Transform - Associates distance with
       edge gradient orientations to enforce directional consistency (EdgeVO).
    5. Multi-Scale Distance Field Pyramid - Multi-octave representation for coarse-to-fine optimization.
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
        # Invert so edge pixels are 0 (sources) and background is 255 (targets)
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
        from pulling the non-linear optimization basin (as formulated in EEDVO / Yan et al.).

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
        Chamfer Distance Transform (L1 Manhattan or C1/C2 integer approximation).
        Useful for high-speed embedded visual odometry.
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
        In addition to the distance to the nearest edge point, stores the gradient
        orientation of the nearest edge pixel. During visual odometry optimization,
        residuals are rejected if |dot(g_ref, g_curr)| < eta (direction consistency).

        Args:
            binary_edges: uint8 edge mask (255 = edge).
            edge_orientations_rad: float32 orientation in radians for edge pixels [-pi, pi].
            max_dist: Truncation threshold.

        Returns:
            Dict containing:
                'distance': Truncated distance map.
                'nearest_y': Y-coordinate of the closest edge pixel.
                'nearest_x': X-coordinate of the closest edge pixel.
                'target_orientation': Orientation angle of closest edge pixel.
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

        # Build lookup table mapping label ID to original coordinate and orientation
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
                # Downsample edges for coarser pyramid level (morphological dilation preserves edge continuity)
                dilated = cv2.dilate(curr_edges, np.ones((2, 2), np.uint8))
                new_w = int(curr_edges.shape[1] * scale_factor)
                new_h = int(curr_edges.shape[0] * scale_factor)
                curr_edges = cv2.resize(dilated, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return pyramid


# ==============================================================================
# SECTION 2: EDGE DETECTION METHODS (RGB AND DEPTH SEPARATELY)
# ==============================================================================

class RGBEdgeDetectionModule:
    """
    RGB Edge Detection Suite:
    1. Canny edge detection with Gaussian / Bilateral smoothing.
    2. Sobel / Scharr gradient magnitude & orientation.
    3. Laplacian of Gaussian (LoG) zero-crossing edge detector.
    4. Illumination-Normalized Gradient (Yan et al.) - Robust to lighting shifts.
    5. Multi-Scale Multi-Resolution Edge Extractor.
    """

    @staticmethod
    def canny_detector(
        rgb_image: np.ndarray,
        low_threshold: int = 40,
        high_threshold: int = 120,
        use_bilateral: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Canny edge detection on RGB. Returns binary edge map and gradient orientation.
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image.copy()

        if use_bilateral:
            # Bilateral filter preserves sharp geometric borders while eliminating sensor noise
            filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=5)
        else:
            filtered = cv2.GaussianBlur(gray, (5, 5), 1.2)

        edges = cv2.Canny(filtered, low_threshold, high_threshold)

        # Compute gradient orientation for EdgeVO directional checking
        gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
        orientation = np.arctan2(gy, gx)

        return edges, orientation

    @staticmethod
    def sobel_scharr_gradients(
        rgb_image: np.ndarray,
        ksize: int = -1,  # -1 invokes Scharr filter (more rotational symmetry)
        magnitude_threshold: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gradient magnitude and thresholded edges using Sobel/Scharr operators.
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image.copy()

        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)

        if ksize == -1:
            gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
        else:
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=ksize)

        mag = cv2.magnitude(gx, gy)
        binary_edges = np.zeros_like(gray, dtype=np.uint8)
        binary_edges[mag > magnitude_threshold] = 255

        # Thinning via morphological skeleton / non-maximum suppression surrogate
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        thinned = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, kernel)

        return thinned, mag, np.arctan2(gy, gx)

    @staticmethod
    def laplacian_of_gaussian(
        rgb_image: np.ndarray,
        kernel_size: int = 5,
        threshold: float = 4.0
    ) -> np.ndarray:
        """
        LoG (Laplacian of Gaussian) edge detector based on second-derivative zero crossings.
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image.copy()

        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1.4)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)

        # Zero-crossing detection
        h, w = gray.shape
        zero_crossings = np.zeros((h, w), dtype=np.uint8)
        lap_p = lap[:-1, :-1]
        lap_right = lap[:-1, 1:]
        lap_down = lap[1:, :-1]

        cond_h = (lap_p * lap_right < 0) & (np.abs(lap_p - lap_right) > threshold)
        cond_v = (lap_p * lap_down < 0) & (np.abs(lap_p - lap_down) > threshold)
        zero_crossings[:-1, :-1][cond_h | cond_v] = 255

        return zero_crossings

    @staticmethod
    def illumination_normalized_edges(
        rgb_image: np.ndarray,
        gradient_quantile: float = 0.85
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Illumination-normalized edge detector (Yan et al., Section 4.1).
        Divides gradient magnitude by local intensity mean to handle abrupt lighting shifts.
        g_norm(p) = ||grad I(p)|| / (mean_local(I(p)) + epsilon)
        """
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = rgb_image.astype(np.float32)

        local_mean = cv2.blur(gray, (15, 15)) + 1e-4
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)

        norm_grad = mag / local_mean
        thresh = float(np.quantile(norm_grad, gradient_quantile))
        edges = np.zeros_like(norm_grad, dtype=np.uint8)
        edges[norm_grad > thresh] = 255

        return edges, norm_grad


class DepthEdgeDetectionModule:
    """
    Depth Edge Detection Suite:
    Separates 3D geometric boundaries from optical texture:
    1. Depth Discontinuity (Step) Edges - Physical depth dropoffs (foreground/background separation).
    2. Depth Crease / Fold Edges - High curvature / normal change across adjacent surfaces.
    3. Relative Depth Step Edges - Normalized by depth Z (Delta Z / Z).
    4. Depth Missing Boundary / Sensor Hole Masker.
    5. Cross-Bilateral RGB-D Gated Edge Fusion (Gating photometric edges with depth structure).
    """

    @staticmethod
    def depth_step_discontinuity_edges(
        depth_map: np.ndarray,
        abs_threshold_meters: float = 0.15,
        depth_scale: float = 1000.0  # e.g., 5000.0 for TUM or 1000.0 for mm
    ) -> np.ndarray:
        """
        Detects step/jump discontinuities where physical depth changes sharply.
        """
        # Convert depth to meters
        if depth_map.dtype in [np.uint16, np.int32]:
            z_m = depth_map.astype(np.float32) / depth_scale
        else:
            z_m = depth_map.copy().astype(np.float32)

        # Invalidate missing depth (0 values)
        valid_mask = z_m > 0.05

        gx = np.zeros_like(z_m)
        gy = np.zeros_like(z_m)
        gx[:, 1:] = np.abs(z_m[:, 1:] - z_m[:, :-1])
        gy[1:, :] = np.abs(z_m[1:, :] - z_m[:-1, :])

        # Suppress gradient at boundaries with missing/unobserved depth
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

        # Reconstruct 3D Point Cloud X = (x_c, y_c, z)
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x_3d = (u - cx) * z / fx
        y_3d = (v - cy) * z / fy
        points_3d = np.stack([x_3d, y_3d, z], axis=-1)

        # Compute tangent vectors via central differences
        dx = np.zeros_like(points_3d)
        dy = np.zeros_like(points_3d)
        dx[:, 1:-1] = points_3d[:, 2:] - points_3d[:, :-2]
        dy[1:-1, :] = points_3d[2:, :] - points_3d[:-2, :]

        # Surface normal is cross product of tangent vectors: N = dX x dY
        normals = np.cross(dx, dy)
        norm_len = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
        normals = normals / norm_len

        # Invalidate where depth is zero
        valid = (z > 0.1)
        normals[~valid] = 0.0

        # Normal angular difference between neighbors
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
        Extracts structural boundaries caused by sensor depth missing areas (shadows / specular reflection).
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
        Fuses RGB and Depth edges (Wang et al. / Project hybrid formulation):
        - Structural Edges: RGB edges verified by a corresponding 3D depth step or crease.
        - Flat Texture Edges: RGB edges without physical depth jump (e.g. wall poster, floor pattern).
        - Geometric Depth-Only Edges: Depth discontinuities with faint photometric contrast.
        """
        combined_depth_edges = cv2.bitwise_or(depth_step_edges, depth_crease_edges)

        # Dilate depth edges to tolerate cross-sensor baseline disparity & calibration misalignment
        kernel = np.ones((disparity_tolerance_kernel, disparity_tolerance_kernel), np.uint8)
        depth_dilated = cv2.dilate(combined_depth_edges, kernel)

        # Structural 3D geometric edges (coincident in RGB and Depth)
        structural_edges = cv2.bitwise_and(rgb_edges, depth_dilated)

        # Flat texture edges (photometric contrast without depth discontinuity)
        texture_edges = cv2.bitwise_and(rgb_edges, cv2.bitwise_not(depth_dilated))

        # Geometric depth silhouette
        geometric_only = cv2.bitwise_and(combined_depth_edges, cv2.bitwise_not(cv2.dilate(rgb_edges, kernel)))

        return {
            "structural_edges": structural_edges,
            "texture_edges": texture_edges,
            "geometric_only_edges": geometric_only,
            "combined_geometry": combined_depth_edges
        }


# ==============================================================================
# SECTION 3: SCENE STUDY & OBSERVABILITY ANALYSIS
# ==============================================================================

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
    Scene Study and Observability Analysis:
    1. Texture Density & Shannon Entropy Analysis - Classifies scene as rich vs texture-less.
    2. Motion Degeneracy & Structure Tensor Analysis - Detects ill-conditioned VO directions
       (aperture problem, corridor effect where edges align along only one dominant axis).
    3. Stochastic Partition Matroid Metric (EdgeVO) - Evaluates spatial distribution uniformity.
    4. Depth Coverage & Missing Hole Statistics.
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

        Returns:
            ratio: Condition ratio (0.0 to 1.0). Low value indicates degeneracy.
            eigenvalues: [lambda_min, lambda_max]
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
        Returns percentage of active grid cells (1.0 = perfectly well-distributed edges).
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


# ==============================================================================
# SECTION 4: LARGE MOVING OBJECT DETECTION (DYNAMIC OBJECT SEGMENTATION)
# ==============================================================================

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
    Detects and segments large moving dynamic objects (e.g. people walking in TUM walking_xyz):
    1. Distance Transform Reprojection Residual: Reference edges warped to current frame.
       Static edges land on Delta_c approx 0; dynamic object edges land on large Delta_c values.
    2. Photometric & Depth Discontinuity Consistency: Detects moving surfaces violating rigid motion.
    3. Spatio-Temporal Residual Clustering: Connected components / morphological clustering
       to aggregate edge residuals into contiguous volumetric moving blobs.
    4. Size and Aspect Ratio Filtering: Distinguishes large moving human bodies / objects
       from sporadic sensor noise or transient flicker.
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
        If motion warp/flow is provided, edge points are warped to current frame coordinates.
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
            # Under no-motion prior (or small inter-frame displacement)
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

        # 1. Edge Reprojection Residuals on Distance Transform
        dt_residuals = self.compute_edge_reprojection_residuals(
            ref_edges, curr_dt_map, estimated_flow_u, estimated_flow_v
        )

        # High residual edge seeds
        high_residual_edges = np.zeros((h, w), dtype=np.uint8)
        high_residual_edges[dt_residuals > self.dt_residual_threshold] = 255

        # 2. Combine with depth motion inconsistency if depth maps are provided
        if prev_depth is not None and curr_depth is not None:
            depth_inconsistency = self.compute_depth_motion_inconsistency(prev_depth, curr_depth)
            combined_dynamic_seeds = cv2.bitwise_or(high_residual_edges, depth_inconsistency)
        else:
            combined_dynamic_seeds = high_residual_edges

        # 3. Morphological Dilation to bridge edges into cohesive spatial blobs
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cluster_dilation_kernel, self.cluster_dilation_kernel)
        )
        dilated_dynamic = cv2.morphologyEx(combined_dynamic_seeds, cv2.MORPH_CLOSE, kernel)
        dilated_dynamic = cv2.dilate(dilated_dynamic, kernel, iterations=2)

        # 4. Connected Component Analysis to isolate large moving bodies
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

        # 5. Extract static inlier edges (mask out dynamic regions)
        static_edges = cv2.bitwise_and(curr_edges, cv2.bitwise_not(dynamic_mask))

        return DynamicObjectDetectionResult(
            dynamic_mask=dynamic_mask,
            static_edge_mask=static_edges,
            bounding_boxes=bounding_boxes,
            residual_map=dt_residuals,
            num_dynamic_clusters=len(bounding_boxes)
        )


# ==============================================================================
# SECTION 5: UNIFIED PIPELINE ORCHESTRATOR
# ==============================================================================

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
        camera_intrinsics: Tuple[float, float, float, float] = (525.0, 525.0, 319.5, 239.5),  # TUM Default
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
        # 1. RGB Edge Extraction (Canny & Gradients)
        rgb_canny, orientations = RGBEdgeDetectionModule.canny_detector(rgb_frame, 35, 110)
        rgb_scharr, scharr_mag, _ = RGBEdgeDetectionModule.sobel_scharr_gradients(rgb_frame)
        rgb_log = RGBEdgeDetectionModule.laplacian_of_gaussian(rgb_frame)
        rgb_norm, _ = RGBEdgeDetectionModule.illumination_normalized_edges(rgb_frame)

        # 2. Depth Edge Extraction (Step, Crease, Relative)
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


# ==============================================================================
# PIPELINE DEMO / TEST INTERFACE
# ==============================================================================

if __name__ == "__main__":
    print("=== Preliminary Edge-Based VO, Distance Transform & Dynamic Object Pipeline ===")
    print("Pipeline defined successfully with all modular components.")
