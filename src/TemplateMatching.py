import os
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from src.SceneMatcher import SceneMatcher
from src.SequentialRANSACHomography import SequentialRANSACHomography
from src.utils import read_rgb_img, log_time

from typing import List, Tuple


class TemplateMatching:

    @staticmethod
    def find_templates(scene_path: str,
                       template_paths: List[str],
                       min_inliers: int = 500,
                       min_remaining_points: int = 600,
                       max_failures: int = 20
                       ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Find multiple templates in the scene image by fitting homographies.

        Parameters:
        - scene_path: Path to the scene image.
        - template_paths: List of paths to the template image.
        - min_inliers: Minimum number of inliers required for a valid homography.
        - min_remaining_points: Minimum number of remaining unmatched points needed to continue analysis.
        - max_failures: Maximum number consecutive failures before stopping analysis of a template.

        Returns:
        - Hs: List of found homographies.
        - matched_template_paths: List of template paths corresponding to the found homographies. 
        """

        scene_matcher = SceneMatcher(scene_path)
        ransac_estimator = SequentialRANSACHomography(min_inliers,
                                                      min_remaining_points,
                                                      max_failures)

        Hs = []
        matched_template_paths = []

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(TemplateMatching.find_single_template,
                                       template_path,
                                       scene_matcher,
                                       ransac_estimator): template_path
                       for template_path in template_paths}

            for future, template_path_cur in futures.items():
                try:
                    Hs_cur = future.result()
                    Hs += Hs_cur
                    matched_template_paths += [template_path_cur] * len(Hs_cur)
                except Exception as e:
                    print(f"Error processing template {futures[future]}: {e}")

        print(f"Found {len(Hs)} homographies")
        return Hs, matched_template_paths

    @staticmethod
    @log_time
    def find_single_template(template_path: str,
                             scene_matcher: SceneMatcher,
                             homographies_estimator: SequentialRANSACHomography
                             ) -> Tuple[List[np.ndarray], str]:
        """
        Find a single template in the scene image by fitting homographies.

        Parameters:
        - template_path: Path to the template image.
        - scene_matches: Instance of SceneMatcher related to a scene image
        - homographies_estimator: Instance of SequentialRANSACHomography to estimate homographies.

        Returns:
        - Hs: List of found homographies.
        """

        tname = os.path.splitext(os.path.basename(template_path))[0]
        print(f"Template: {tname}")

        # Get the matches for the current template
        template_keypoints, matches = scene_matcher.get_matches(template_path)

        # Convert matches to points for RANSAC
        template_points = get_pts(template_keypoints)[matches[:, 1]]
        scene_points = get_pts(scene_matcher.scene_keypoints)[matches[:, 0]]

        # Perform RANSAC for homography estimation
        template_height, template_width = read_rgb_img(template_path).shape[:2]
        template_homographies = homographies_estimator.run(template_points,
                                                           scene_points,
                                                           template_height,
                                                           template_width)

        return template_homographies


def get_pts(keypoints: List[cv.KeyPoint]) -> np.ndarray:
    """
    Extracts keypoints coordinates from list of OpeCV keypoints.

    Parameters:
    - keypoints: List of OpenCV `KeyPoint` objects.

    Returns:
    - points: Array of shape (N, 2), where N is the number of keypoints.
    """
    points = np.array([kp.pt for kp in keypoints])
    return points
