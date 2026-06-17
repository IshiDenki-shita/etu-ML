import cv2
import numpy as np


def preprocess_image(image: np.ndarray, binary_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.threshold(
        blurred,
        binary_threshold,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return opened
