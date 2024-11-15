import numpy as np
import cv2 as cv
from typing import List


class SequentialRANSACHomography:
    """
    Perform homography estimation using the Sequential RANSAC algorithm.
    """

    def __init__(self,
                 min_inliers: int = 500,
                 min_remaining_points: int = 600,
                 max_failures: int = 20,
                 confidence: float = 0.95,
                 lo_method: int = cv.LOCAL_OPTIM_GC,
                 max_iterations: int = 500,
                 sampler: int = cv.SAMPLING_UNIFORM,
                 score: int = cv.SCORE_METHOD_MSAC,
                 inlier_thr: float = 3.,
                 lo_size: int = 20,
                 lo_iter: int = 25,
                 state: int = None):
        """
        Initializes the RANSAC homography estimator.

        Parameters:
        - min_inliers: Minimum number of inliers required for a valid homography.
        - min_remaining_points: Minimum number of remaining unmatched points needed to continue analysis.
        - max_failures: Maximum number consecutive failures before stopping analysis of a template.
        - confidence: Confidence level for the RANSAC algorithm.
        - lo_method: Method used for local optimization during the RANSAC process.
        - max_terations: Maximum number of RANSAC iterations. 
        - sampler: Sampling method used within the RANSAC process.
        - score: Scoring method for evaluating the fit of the model.
        - inlier_thr: Threshold for considering a point an inlier or outlier.
        - lo_size: Sample size used during local optimization.
        - lo_iter: Number of iterations to perform for local optimization.
        - state: Custom state for the random number generator. If not provided, a random state is generated.
        """

        self.min_inliers = min_inliers
        self.min_remaining_points = min_remaining_points
        self.max_failures = max_failures

        self.usac_params = cv.UsacParams()

        self.usac_params.confidence = confidence
        self.usac_params.loMethod = lo_method
        self.usac_params.maxIterations = max_iterations
        self.usac_params.sampler = sampler
        self.usac_params.score = score
        self.usac_params.threshold = inlier_thr

        self.usac_params.isParallel = False
        self.usac_params.loSampleSize = lo_size
        self.usac_params.loIterations = lo_iter
        self.usac_params.neighborsSearch = None

        if state is None:
            rng = np.random.default_rng()
            state = int(rng.choice(np.arange(0, 1e5, 1), 1)[0])

        self.usac_params.randomGeneratorState = state

    def run(self,
            template_points: np.ndarray,
            scene_points: np.ndarray,
            template_height: int,
            template_width: int
            ) -> List[np.ndarray]:
        """
        Performs sequential RANSAC to fit multiple homographies between template and scene points.

        Parameters:
        - template_points: Points in the template image.
        - scene_points: Corresponding points in the scene image.
        - template_height: Height of the template image.
        - template_width: Width of the template image.

        Returns:
        - Hs: List of found homographies.
        """

        assert len(template_points) == len(scene_points)
        remaining_indices = np.arange(len(template_points))
        Hs = []
        failure_count = 0

        while len(remaining_indices) >= self.min_remaining_points:

            if failure_count > self.max_failures:
                break

            # Shuffle to select points randomly
            idxs = np.random.permutation(len(template_points))
            template_points = template_points[idxs]
            scene_points = scene_points[idxs]

            # Fit homography
            H, inliers_mask = cv.findHomography(template_points.reshape(-1, 1, 2),
                                                scene_points.reshape(-1, 1, 2),
                                                self.usac_params)
            num_inliers = np.count_nonzero(inliers_mask)

            # Check for valid homography
            if H is None or num_inliers < self.min_inliers or np.linalg.matrix_rank(H) < 3 or np.abs(np.linalg.det(H)) < 1e-10:
                failure_count += 1
                continue
            else:
                failure_count = 0
                Hs.append(H)

            # Filter out inliers for next round
            inliers_mask = inliers_mask.astype(bool).squeeze()
            outliers_mask = np.logical_not(inliers_mask)
            template_points = template_points[outliers_mask]
            scene_points = scene_points[outliers_mask]
            remaining_indices = remaining_indices[outliers_mask]

            # Remove points within the template corners projected to scene
            template_corners_homog = np.asarray([[0, 0, 1],
                                                 [template_width, 0, 1],
                                                 [template_width,
                                                     template_height, 1],
                                                 [0, template_height, 1]],
                                                dtype='float32')
            projected_template_corners_homog = (H @ template_corners_homog.T).T
            projected_template_corners = projected_template_corners_homog[:,
                                                                          :2] / projected_template_corners_homog[:, 2:]
            outliers_mask = outside_polygon_mask(scene_points,
                                                 projected_template_corners)
            template_points = template_points[outliers_mask]
            scene_points = scene_points[outliers_mask]
            remaining_indices = remaining_indices[outliers_mask]

        return Hs


def outside_polygon_mask(pts_scene: np.ndarray,
                         polygon: np.ndarray
                         ) -> np.ndarray:
    """
    Given a set of points and a polygon, compute a mask where `True` indicates 
    that a point is outside the polygon and `False` indicates that it is inside.
    """
    inside_mask = np.zeros(len(pts_scene), dtype=bool)
    for i, pt in enumerate(pts_scene):
        if cv.pointPolygonTest(polygon.astype(np.float32),
                               (pt[0], pt[1]), False) >= 0:
            inside_mask[i] = True
    outside_mask = np.logical_not(inside_mask)
    return outside_mask
