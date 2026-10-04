import cv2 as cv
import numpy as np

class SaddlePointDetector:
    """
    Saddle point detector using the Hessian matrix and non-maximum suppression.

    Attributes:
        image (np.ndarray): Input image.
        max_pts (int): Maximum number of saddle points to detect.
        filter (bool): Threshold for filtering saddle points.
    """
    def __init__(self, image: np.ndarray, max_pts: int, filter: bool):
        self.image = image
        self.max_pts = max_pts
        self.filter = filter


    def detect(self) -> np.ndarray:
        # convert image to grayscale if it is not already
        if len(self.image.shape) == 3:
            gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        else:
            gray_image = self.image.copy()

        win_size = 10

        gray = cv.blur(gray_image, (3,3))
        saddle, sub_s, sub_t, gx, gy = self._get_saddle(gray)
        self._non_max_suppression(saddle)


        saddle[saddle< 10000] = 0
        sub_idxs = np.nonzero(saddle)
        spts = np.argwhere(saddle).astype(np.float64)[:, [1, 0]]

        sub_off = np.array([sub_s[sub_idxs], sub_t[sub_idxs]]).T
        spts = spts + sub_off

        saddle_strengths = saddle[sub_idxs]
        sorted_indicies = np.argsort(saddle_strengths)[::-1]  # sort in descending order
        spts = spts[sorted_indicies]

        spts = self._filter_saddle_points(gray_image=gray, gx=gx, gy=gy, saddle_pts=spts, filter=self.filter, win_size=win_size)
        # Take only the top max_pts
        if self.max_pts > 0 and len(spts) > self.max_pts:
            spts = spts[:self.max_pts]

        return spts


    def _get_saddle(gray_image):
        img = gray_image
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        gxx = cv.Sobel(gx, cv.CV_32F, 1, 0)
        gyy = cv.Sobel(gy, cv.CV_32F, 0, 1)
        gxy = cv.Sobel(gx, cv.CV_32F, 0, 1)

        S = -gxx * gyy + gxy**2

        denom = (gxx*gyy - gxy*gxy)
        sub_s = np.divide(gy*gxy - gx*gyy, denom, out=np.zeros_like(denom), where=denom!=0)
        sub_t = np.divide(gx*gxy - gy*gxx, denom, out=np.zeros_like(denom), where=denom!=0)
        return S, sub_s, sub_t, gx, gy


    def _non_max_suppression(image, size=11):
        element = np.ones([size, size], dtype=np.uint8)
        dilated = cv.dilate(image, element)
        peaks = cv.compare(image, dilated, cv.CMP_EQ)
        image[peaks == 0] = 0


    def _filter_saddle_points(
            self,
            gray_image,
            gx,
            gy,
            saddle_pts,
            filter: bool,
            win_size: int):
        """
        Filter saddle points based on a given criterion.

        Args:
            saddle_pts (np.ndarray): Array of saddle points.
            filter (bool): Whether to apply filtering.
            win_size (int): Window size for filtering.

        Returns:
            np.ndarray: Filtered saddle points.
        """

        if len(saddle_pts) > 0 and filter:
            # 1. edge clipping: remove points too close to the image border
            near_boarder = np.logical_or(
                np.any(saddle_pts <= win_size, axis=1),
                np.any(saddle_pts[:, [1, 0]] >= np.array(gray_image.shape) - win_size - 1, axis=1)
            )

            # 2. rose plot symmetry
            h, w_img = gray_image.shape
            mag = np.sqrt(gx**2 + gy**2)
            ixs = np.round(saddle_pts[:, 0]).astype(int)
            iys = np.round(saddle_pts[:, 1]).astype(int)

            # ensure indices are within image bounds
            ixs = np.clip(ixs, 5, w_img - 6)
            iys = np.clip(iys, 5, h - 6)

            # Precompute relative offsets for 40 square boundary pixels (clockwise)
            dx = np.concatenate([np.arange(-5, 6), np.ones(9, dtype=int)*5, np.arange(5, -6, -1), np.ones(9, dtype=int)*-5])
            dy = np.concatenate([np.ones(11, dtype=int)*-5, np.arange(-4, 5), np.ones(11, dtype=int)*5, np.arange(4, -5, -1)])
            
            # Extract all boundary rings at once: Shape (N, 40)
            ring_mags = mag[iys[:, None] + dy, ixs[:, None] + dx]
            
            # Calculate magnitude symmetry score using 180-degree periodic correlation
            row_sums = np.sum(ring_mags, axis=1, keepdims=True)
            norm_mags = ring_mags / (row_sums + 1e-6)
            scores = np.sum(norm_mags * np.roll(norm_mags, 20, axis=1), axis=1)
            
            # 3. Extract Intensity Ring for Point Symmetry
            ring_intensities = gray_image[iys[:, None] + dy, ixs[:, None] + dx].astype(np.float32)
            
            # Calculate Intensity Symmetry (NCC on the ring)
            # Subtract mean of each ring
            ring_means = np.mean(ring_intensities, axis=1, keepdims=True)
            ring_centered = ring_intensities - ring_means
            
            # 180-degree correlation (periodic shift by 20 in a 40-pixel ring)
            ring_rot = np.roll(ring_centered, 20, axis=1)
            num = np.sum(ring_centered * ring_rot, axis=1)
            den = np.sqrt(np.sum(ring_centered**2, axis=1) * np.sum(ring_rot**2, axis=1))
            # ncc_scores will be near 1.0 for symmetric corners, near -1.0 for anti-symmetric
            ncc_scores = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
            
            # Combined filter: high magnitude symmetry AND high intensity symmetry
            # X-corners have 4 clear peaks (scores) and point symmetry (ncc_scores)
            valid_mask = ~near_boarder & (scores >= 0.02) & (ncc_scores >= 0.2)
                
            spts = saddle_pts[valid_mask]

        return spts