import time
import numpy as np
import cv2 as cv

def log_time(func):
    """
    Decorator function to compute the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper


def read_rgb_img(img_path: str) -> np.ndarray:
    """
    Reads an image from a file path in RGB format.

    Parameters:
    - img_path: File path of the image to read.

    Returns:
    - img_rgb: Image in RGB format.
    """
    img_bgr = cv.imread(img_path, cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    return img_rgb
