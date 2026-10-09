import numpy as np
import cv2 as cv
from typing import Tuple, Optional, Dict, List, Any

class ChessboardSolver:
    def __init__(
            self,
            lattice_points: np.ndarray,
            logging: bool = False
    ):
        self.lattice_points = lattice_points
        self.logging = logging

    # ============================================================================
    # Public API
    # ============================================================================
    def estimate_chess_grid(self, lattice_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Estimate the chessboard grid from the given lattice points.

        Args:
            lattice_points (np.ndarray): The input lattice points.

        Returns:
            Tuple[np.ndarray, np.ndarray, dict]:
            - Estimated chess grid points of shape (N_grid, 2).
            - Basis vectors of shape (2, 2).
            - Additional information as a dictionary.
        """
        if len(lattice_points) < 4:
            return np.zeros_like(lattice_points), np.zeros((2, 2)), {}

        basis_vectors, projected_points, _, info = self._get_lattice_and_reproject(
            lattice_points, num_vectors=2
            )

        return np.round(projected_points), basis_vectors, info

    def estimate_homography(self, lattice_points: np.ndarray, chess_grid_points: np.ndarray) -> np.ndarray:
        """
        Estimate the homography matrix that maps the lattice points to the chess grid points.

        Args:
            lattice_points (np.ndarray): The input lattice points.
            chess_grid_points (np.ndarray): The corresponding chess grid points.

        Returns:
            np.ndarray: The estimated homography matrix of shape (3, 3).
        """
        if len(lattice_points) < 4:
            return np.eye(3)
        H, _ = cv.findHomography(lattice_points.astype(np.float32), chess_grid_points.astype(np.float32), method=cv.RANSAC)
        if H is None:
            return np.eye(3)
        return H


    # ============================================================================
    # Helper methods for ChessboardSolver and Homography
    # ============================================================================
    def _find_lattice_basis_vector(
            self,
            points: np.ndarray,
            num_vectors: int = 8,
            displacements: Optional[np.ndarray] = None,
            bins: int = 100,
    ) -> Tuple[np.ndarray, dict]:
        """
        Find the lattice basis vectors from the given points.

        Args:
            points (np.ndarray): The input points.
            num_vectors (int): The number of basis vectors to find.
            displacements (Optional[np.ndarray]): Optional displacements for the points.
            bins (int): The number of bins for the histogram.

        Returns:
            Tuple[np.ndarray, dict]:
            - np.ndarray: The found lattice basis vectors.
            - dict: Additional information about the lattice basis vectors.
        """
        # Calculate all pairwise displacement vectors between points
        if displacements is None:
            displacements = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        flat_dicts = displacements.reshape((-1, 2))

        # Determining the histogram of displacement vectors with outliner removal
        dx_sorted = np.sort(flat_dicts[:, 0])
        dy_sorted = np.sort(flat_dicts[:, 1])
        n = len(dx_sorted)
        i_lo, i_hi = int(0.20 * (n - 1)), int(0.80 * (n - 1))
        x_min, x_max = dx_sorted[i_lo], dx_sorted[i_hi]
        y_min, y_max = dy_sorted[i_lo], dy_sorted[i_hi]

        # Building the histogram of displacement vectors within the filtered range
        x_scale = bins / (x_max - x_min)
        y_scale = bins / (y_max - y_min)
        xi = ((flat_dicts[:, 0] - x_min) * x_scale).astype(np.intp)
        yi = ((flat_dicts[:, 1] - y_min) * y_scale).astype(np.intp)
        mask = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)
        flat_idx = xi[mask] * bins + yi[mask]
        hist = np.bincount(flat_idx, minlength=bins * bins).reshape(bins, bins)
        x_bin_width = (x_max - x_min) / bins
        y_bin_width = (y_max - y_min) / bins

        # Smooth the histogram using a Gaussian filter
        sigma = 3
        ksize = 2 * int(4 * sigma + 0.5) + 1
        density_map = cv.GaussianBlur(hist.T.astype(np.float32), (ksize, ksize), sigma)

        # find local maxima
        min_distance = 5
        kernel_size = 2 * min_distance + 1
        dilated = cv.dilate(density_map, np.ones((kernel_size, kernel_size), dtype=np.float32))
        threshold = 0.01 * density_map.max()
        peak_mask = (density_map == dilated) & (density_map > threshold)
        coordinates = np.argwhere(peak_mask)

        # convert coordinates to displacement vectors
        if len(coordinates) > 0:
            peak_vectors = np.column_stack([
                x_min + (coordinates[:, 1] + 0.5) * x_bin_width,
                y_min + (coordinates[:, 0] + 0.5) * y_bin_width,
            ])
        else:
            peak_vectors = np.empty((0, 2))
        peak_scores = np.array([density_map[c[0], c[1]] for c in coordinates])

        if len(peak_vectors) > 0:
            origin_idx = np.argmin(np.linalg.norm(peak_vectors, axis=1))
            peak_vectors = np.delete(peak_vectors, origin_idx, axis=0)
            peak_scores = np.delete(peak_scores, origin_idx, axis=0)

        # Score based on solver for best peak vectors
        harmonic_scores = self._calculate_harmonic_scores(peak_vectors, peak_scores)
        best_vectors = self._select_best_vectors(peak_vectors, harmonic_scores, num_vectors=2)

        return self._pad_vectors(best_vectors, num_vectors)

    def _calculate_harmonic_scores(
            self,
            peak_vectors: np.ndarray,
            peak_scores: np.ndarray
        ) -> np.ndarray:
        """
        Calculate harmonic scores for the given peak vectors and their corresponding scores.

        Args:
            peak_vectors (np.ndarray): Array of peak vectors.
            peak_scores (np.ndarray): Array of peak scores.

        Returns:
            np.ndarray: Array of harmonic scores. Shape (N,) for each vector.
        """
        harmonic_scores = peak_scores.copy()
        num_peaks = len(peak_vectors)
        if num_peaks == 0:
            return harmonic_scores

        COS_SIM_THRESH = 0.985
        MAG_RATIO_TOL = 0.15

        # compute pairwise magnitudes
        mags = np.linalg.norm(peak_vectors, axis=1)
        mags_i = mags[:, np.newaxis]
        mags_j = mags[np.newaxis, :]

        # Condition 1: Candidate harmonic to check must be significantly larger
        condition1 =  mags_j > mags_i * 1.5

        # Condition 2: Vectors must point in almost the exact same direction (high cosine similarity)
        dot_products = peak_vectors @ peak_vectors.T
        mag_products = mags_i * mags_j
        cos_sim_matrix = dot_products / (mag_products + 1e-9)  # Add a small epsilon to avoid division by zero
        condition2 = cos_sim_matrix > COS_SIM_THRESH

        # Condition 3: Magnitude ratio must be within tolerance
        mag_ratio_matrix = mags_j / (mags_i + 1e-9)  # Add a small epsilon to avoid division by zero
        k = np.round(mag_ratio_matrix)
        condition3 = np.abs(mag_ratio_matrix - k) < MAG_RATIO_TOL

        # Combine mask for valid harmonic candidates
        identity = np.eye(num_peaks, dtype=bool)
        combined_mask = condition1 & condition2 & condition3 & ~identity

        # Distribute the score of the larger harmonic back to the fundamental vector
        peak_scores_j_broadcast = np.tile(peak_scores[np.newaxis, :], (num_peaks, 1))
        k_safe = np.where(k == 0, 1, k)
        contributions = np.where(combined_mask, peak_scores_j_broadcast / k_safe, 0)
        harmonic_scores += np.sum(contributions, axis=1)

        return harmonic_scores
        

    def _select_best_vectors(
            self,
            peak_vectors: np.ndarray,
            harmonic_scores: np.ndarray,
            num_vectors: int
        ) -> np.ndarray:
        """
        Selects the best non redundant vectors based on their harmonic scores.

        Args:
            peak_vectors (np.ndarray): Array of peak vectors.
            harmonic_scores (np.ndarray): Array of harmonic scores corresponding to the peak vectors.
            num_vectors (int): Number of best vectors to select.

        Returns:
            np.ndarray: Array of selected best vectors. Shape (num_vectors, 2).
        """
        magnitudes = np.linalg.norm(peak_vectors, axis=1)
        final_metric = harmonic_scores / (magnitudes + 1e-6)  # Avoid division by zero
        top_indices = np.argsort(final_metric)[::-1]

        final_vectors: List[np.ndarray] = []
        COS_SIM_THRESH = 0.985

        for idx in top_indices:
            if len(final_vectors) >= num_vectors:
                break
            candidate_vec = peak_vectors[idx]
            if candidate_vec[0] < -1e-6:
                candidate_vec = -candidate_vec
            elif abs(candidate_vec[0]) < 1e-6 and candidate_vec[1] < -1e-6:
                candidate_vec = -candidate_vec
            is_redundant = False
            for existing_vec in final_vectors:
                norm_product = np.linalg.norm(candidate_vec) * np.linalg.norm(existing_vec)
                if norm_product > 1e-6:  # Avoid division by zero
                    cos_sim = np.dot(candidate_vec, existing_vec) / norm_product
                    if cos_sim > COS_SIM_THRESH:
                        is_redundant = True
                        break
            if not is_redundant:
                final_vectors.append(candidate_vec)

        return final_vectors


    def _pad_vectors(self, vectors: List[np.ndarray], target_count: int) -> np.ndarray:
        """Pads the list of vectors to the target count with zero vectors if necessary."""
        output_array = np.array(vectors)
        num_found = len(output_array)
        if num_found < target_count:
            padding = np.zeros((target_count - num_found, 2))
            if num_found == 0:
                return padding
            output_array = np.vstack([output_array, padding])
        return output_array


    def _reproject_points(
            self,
            points: np.ndarray,
            basis_vectors: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects the given points onto the provided basis vectors.

        Args:
            points (np.ndarray): Array of points to be projected. Shape (num_points, 2).
            basis_vectors (np.ndarray): Array of basis vectors. Shape (num_vectors, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - points_lattice (np.ndarray): The coordinates of (N, 2).
                - center (np.ndarray): The center of the points. Shape (2,).
        """
        if np.linalg.det(basis_vectors) == 0:
            return points, np.mean(points, axis=0)

        # Transform points to the new basis
        B_inv = np.linalg.inv(basis_vectors)
        points_lattice = points @ B_inv

        # Remove mean offset
        points_lattice_offset = points_lattice.mean(axis=0)
        points_lattice -= points_lattice_offset

        # Use circular mean to determine the center
        fractional_parts = np.mod(points_lattice, 1)
        angles = 2 * np.pi * fractional_parts
        mean_angles = np.angle(np.mean(np.exp(1j * angles), axis=0))
        points_lattice_frac_offset = np.mod(mean_angles / (2 * np.pi), 1)
        points_lattice -= points_lattice_frac_offset

        # Snap points to the nearest lattice points
        integer_indices = np.round(points_lattice).astype(int)
        integer_offset = np.round(np.mean(integer_indices, axis=0)).astype(int)
        points_lattice -= integer_offset
        integer_indices -= integer_offset

        # Calculate the ideal lattice points and use median difference to find robust center
        lattice_part = integer_indices @ basis_vectors
        origin_estimates = points - lattice_part
        best_center = np.median(origin_estimates, axis=0)

        # Map the best center back into the lattice space
        best_center_lattice = (
            (best_center @ B_inv)
            - points_lattice_offset
            - points_lattice_frac_offset
            - integer_offset
        )
        points_lattice -= best_center_lattice

        return points_lattice, best_center
    

    def _get_lattice_and_reproject(
            self,
            points: np.ndarray,
            num_vectors: int = 2,
            displacements: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Finds the 2D lattice basis vectors and reprojects the points onto this lattice.

        Args:
            points (np.ndarray): Array of points to be reprojected. Shape (num_points, 2).
            num_vectors (int): Number of basis vectors to find. Default is 2.
            displacements (Optional[np.ndarray]): Optional array of displacements for the points. Shape (num_points, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, dict]: A tuple containing:
                - basis_vectors (np.ndarray): The found basis vectors. Shape (2, 2).
                - projected_points (np.ndarray): The points reprojected onto the lattice. Shape (num_points, 2).
                - center (np.ndarray): The estimated center of the lattice. Shape (2,).
                - additional_info (dict): Additional information about the lattice and reprojection.
        """
        assert(len(points.shape) == 2)

        basis_vectors = self._find_lattice_basis_vector(points, num_vectors, displacements=displacements)
        angles = np.arctan2(basis_vectors[:, 1], basis_vectors[:, 0])
        basis_vectors = basis_vectors[np.argsort(angles)]

        projected_points, center = self._reproject_points(points, basis_vectors)

        return basis_vectors, projected_points, center, {}