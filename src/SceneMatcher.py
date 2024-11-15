import numpy as np
import cv2 as cv
from src.utils import read_rgb_img
from typing import Tuple, List


class SceneMatcher:
    """
    Extract and match SIFT features between a scene image and multiple template images.
    
    Attributes:
        scene_img (np.ndarray): The loaded scene image in RGB format.
        scene_keypoints (List[cv.KeyPoint]): Keypoints extracted from the scene image.
        scene_descriptors (np.ndarray): Descriptors for keypoints in the scene image.
    """

    def __init__(self,
                 scene_path: str):
        """
        Initializes the SceneMatcher.

        Parameters:
        - scene_path: Path to the scene image.
        """

        self.scene_img = read_rgb_img(scene_path)
        self.scene_keypoints, self.scene_descriptors = sift_features(scene_path)

    def get_matches(self, template_path: str) -> Tuple[np.ndarray, List[cv.KeyPoint]]:
        """
        Finds matches between the template image and the scene image using SIFT descriptors.

        Parameters:
        - template_path (str): Path to the template image.

        Returns:
        - template_keypoints: Keypoints in the template image
        - matches: Array of shape (N, 2), where N is the number of matches. Axis 0 represents the indices of the scene points, and axis 1 represents the indices of the template points.
        """

        template_keypoints, template_descriptors = sift_features(template_path)
        matches = match_sift_descriptors(template_descriptors, 
                                         self.scene_descriptors)
        return template_keypoints, matches


def match_sift_descriptors(descriptors_template: np.ndarray,
                           descriptors_scene: np.ndarray,
                           ratio_thr: float = 0.8) -> np.ndarray:
    """
    Match SIFT descriptors of the scene with template descriptors.

    Parameters:
    - descriptors_template: Descriptors extracted from the template image.
    - descriptors_scene: Descriptors extracted from the scene image.
    - ratio_thr: threshold for Lowe's ratio test.

    Returns:
    - matches: Array of shape (N, 2), where N is the number of matches. Axis 0 represents the indices of the scene points, and axis 1 represents the indices of the template points.
    """

    # Using KDTree (algorithm=1) with 5 trees
    index_params = dict(algorithm=1, trees=5)
    # Number of checks for search optimization
    search_params = dict(checks=50)
    
    flann_matcher = cv.FlannBasedMatcher(index_params, search_params)

    # For each descriptor in the scene, look for the two nearest descriptors in the template
    # Note: the order of descriptors matters
    matches = flann_matcher.knnMatch(descriptors_scene.astype(np.float32),
                                     descriptors_template.astype(np.float32),
                                     k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thr * n.distance:
            good_matches.append((m.queryIdx, m.trainIdx))

    matches = np.array(good_matches, dtype=int)
    return matches


def sift_features(img_path: str) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    """
    Extracts SIFT keypoints and descriptors from the image at the specified input path.
    """
    img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors
