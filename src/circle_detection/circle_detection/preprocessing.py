# circle_detection/preprocessing.py

import cv2
import numpy as np

def gamma_correction(image, gamma=1.5):
    """Gamma校正"""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def detect_circles(image_bgr, gamma, hough_params):
    """检测图像中的圆"""
    enhanced = gamma_correction(image_bgr, gamma=gamma)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=hough_params['dp'],
        minDist=hough_params['minDist'],
        param1=hough_params['param1'],
        param2=hough_params['param2'],
        minRadius=hough_params['minRadius'],
        maxRadius=hough_params['maxRadius'],
    )
    result = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for (x, y, r) in circles:
            result.append((x, y, r))
    return result
