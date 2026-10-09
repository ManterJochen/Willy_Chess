import cv2 as cv
import numpy as np


class SaddlePointDetector:
    """
    Detect saddle points using the Hessian matrix, non-maximum suppression,
    subpixel refinement, and optional symmetry filtering.

    Attributes:
        image (np.ndarray): Input image.
        max_pts (int): Maximum number of points to return. <= 0 means no limit.
        apply_filter (bool): Whether to apply symmetry and border filtering.
        threshold (float): Minimum saddle response.
        nms_size (int): Window size for non-maximum suppression.
        win_size (int): Border margin used during filtering.
    """

    def __init__(
        self,
        image: np.ndarray,
        max_pts: int = 0,
        apply_filter: bool = True,
        threshold: float = 10000.0,
        nms_size: int = 11,
        win_size: int = 10,
    ):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image must be a valid NumPy array.")

        if image.size == 0:
            raise ValueError("image must not be empty.")

        if image.ndim not in (2, 3):
            raise ValueError("image must be grayscale or a color image.")

        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError("Color image must have 3 or 4 channels.")

        if nms_size < 1 or nms_size % 2 == 0:
            raise ValueError("nms_size must be a positive odd number.")

        if win_size < 5:
            raise ValueError("win_size must be at least 5.")

        self.image = image
        self.max_pts = max_pts
        self.apply_filter = apply_filter
        self.threshold = threshold
        self.nms_size = nms_size
        self.win_size = win_size

    def detect(self) -> np.ndarray:
        """
        Detect saddle points.

        Returns:
            np.ndarray: Points with shape (N, 2), in (x, y) order,
                        using float64 subpixel coordinates.
        """
        # Convert input to grayscale.
        if self.image.ndim == 3:
            if self.image.shape[2] == 4:
                gray_image = cv.cvtColor(
                    self.image, cv.COLOR_BGRA2GRAY
                )
            else:
                gray_image = cv.cvtColor(
                    self.image, cv.COLOR_BGR2GRAY
                )
        else:
            gray_image = self.image.copy()

        # Sobel derivatives work with floating-point image data.
        gray = cv.blur(gray_image, (3, 3))
        gray = gray.astype(np.float32, copy=False)

        h, w = gray.shape

        # The symmetry filter needs a ring with radius 5.
        if h <= 2 * self.win_size + 1 or w <= 2 * self.win_size + 1:
            return np.empty((0, 2), dtype=np.float64)

        saddle, sub_s, sub_t, gx, gy = self._get_saddle(gray)

        # Keep only local maxima.
        self._non_max_suppression(saddle, self.nms_size)

        # Apply the response threshold.
        saddle[saddle < self.threshold] = 0

        # Extract candidate locations.
        ys, xs = np.nonzero(saddle)

        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float64)

        # Convert (y, x) to (x, y).
        points = np.column_stack((xs, ys)).astype(np.float64)

        # Refine locations to subpixel coordinates.
        offsets = np.column_stack((
            sub_s[ys, xs],
            sub_t[ys, xs],
        ))

        points += offsets

        # Sort candidates by saddle response, strongest first.
        strengths = saddle[ys, xs]
        sorted_indices = np.argsort(strengths)[::-1]
        points = points[sorted_indices]

        # Optional border and symmetry filtering.
        if self.apply_filter:
            points = self._filter_saddle_points(
                gray_image=gray,
                gx=gx,
                gy=gy,
                saddle_pts=points,
                win_size=self.win_size,
            )

        # Return only the requested number of points.
        if self.max_pts > 0:
            points = points[:self.max_pts]

        return points

    def _get_saddle(
        self,
        gray_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Hessian determinant response and subpixel offsets.
        """
        gx = cv.Sobel(gray_image, cv.CV_32F, 1, 0)
        gy = cv.Sobel(gray_image, cv.CV_32F, 0, 1)

        gxx = cv.Sobel(gx, cv.CV_32F, 1, 0)
        gyy = cv.Sobel(gy, cv.CV_32F, 0, 1)
        gxy = cv.Sobel(gx, cv.CV_32F, 0, 1)

        # A saddle point has a negative Hessian determinant.
        det = gxx * gyy - gxy * gxy
        saddle = -det

        # Avoid division by zero where the Hessian is singular.
        sub_s = np.divide(
            gy * gxy - gx * gyy,
            det,
            out=np.zeros_like(det),
            where=det != 0,
        )

        sub_t = np.divide(
            gx * gxy - gy * gxx,
            det,
            out=np.zeros_like(det),
            where=det != 0,
        )

        return saddle, sub_s, sub_t, gx, gy

    def _non_max_suppression(
        self,
        image: np.ndarray,
        size: int,
    ) -> None:
        """
        Suppress pixels that are not local maxima.

        Modifies the input image in place.
        """
        kernel = np.ones((size, size), dtype=np.uint8)
        dilated = cv.dilate(image, kernel)

        peaks = cv.compare(image, dilated, cv.CMP_EQ)
        image[peaks == 0] = 0

    def _filter_saddle_points(
        self,
        gray_image: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        saddle_pts: np.ndarray,
        win_size: int,
    ) -> np.ndarray:
        """
        Filter points based on border distance, gradient symmetry,
        and intensity symmetry.

        Input and output coordinates are in (x, y) order.
        """
        if len(saddle_pts) == 0:
            return saddle_pts

        h, w = gray_image.shape

        # Reject points too close to the border.
        margin = max(win_size, 5)

        x = saddle_pts[:, 0]
        y = saddle_pts[:, 1]

        valid = (
            (x >= margin)
            & (y >= margin)
            & (x < w - margin)
            & (y < h - margin)
        )

        saddle_pts = saddle_pts[valid]

        if len(saddle_pts) == 0:
            return saddle_pts

        # Convert remaining subpixel coordinates to integer indices.
        ixs = np.rint(saddle_pts[:, 0]).astype(int)
        iys = np.rint(saddle_pts[:, 1]).astype(int)

        # Magnitude of the image gradient.
        mag = np.sqrt(gx**2 + gy**2)

        # Construct the 40-pixel boundary ring.
        dx = np.concatenate((
            np.arange(-5, 6),
            np.full(9, 5, dtype=int),
            np.arange(5, -6, -1),
            np.full(9, -5, dtype=int),
        ))

        dy = np.concatenate((
            np.full(11, -5, dtype=int),
            np.arange(-4, 5),
            np.full(11, 5, dtype=int),
            np.arange(4, -5, -1),
        ))

        # Gradient magnitude symmetry.
        ring_mags = mag[iys[:, None] + dy, ixs[:, None] + dx]

        row_sums = np.sum(ring_mags, axis=1, keepdims=True)
        norm_mags = ring_mags / (row_sums + 1e-6)

        scores = np.sum(
            norm_mags * np.roll(norm_mags, 20, axis=1),
            axis=1,
        )

        # Intensity symmetry.
        ring_intensities = gray_image[
            iys[:, None] + dy,
            ixs[:, None] + dx,
        ]

        ring_means = np.mean(
            ring_intensities,
            axis=1,
            keepdims=True,
        )

        ring_centered = ring_intensities - ring_means
        ring_rot = np.roll(ring_centered, 20, axis=1)

        numerator = np.sum(ring_centered * ring_rot, axis=1)

        denominator = np.sqrt(
            np.sum(ring_centered**2, axis=1)
            * np.sum(ring_rot**2, axis=1)
        )

        ncc_scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )

        # Apply both symmetry criteria.
        valid_mask = (scores >= 0.02) & (ncc_scores >= 0.2)

        return saddle_pts[valid_mask]