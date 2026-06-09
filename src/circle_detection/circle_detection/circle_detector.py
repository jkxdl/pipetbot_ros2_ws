import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
from collections import deque

class KalmanFilter3D:
    def __init__(self, dt):
        # 初始化 Kalman 滤波器参数
        self.dt = dt
        self.state = np.zeros((6, 1), dtype=np.float32)  # [X, Y, Z, Vx, Vy, Vz]
        self.A = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0, 0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1.0  # 状态协方差
        self.Q = np.eye(6, dtype=np.float32) * 0.01  # 过程噪声
        self.R = np.eye(3, dtype=np.float32) * 0.1   # 测量噪声

    def predict(self):
        # 预测步骤
        self.state = np.dot(self.A, self.state)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.state

    def correct(self, measurement):
        # 校正步骤
        z = measurement
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        I = np.eye(self.A.shape[0], dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def update_transition_matrix(self, dt):
        # 动态更新状态转移矩阵 A
        self.A = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0, 0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=np.float32)
        # 重新计算预测后的状态和协方差
        self.predict()

def detect_circles(image_bgr, gamma, hough_params):
    """检测图像中的圆"""
    try:
        # Gamma校正
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(image_bgr, table)

        # 高斯模糊以减少噪声
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)

        # 转换为灰度图
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 霍夫圆检测
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
    except Exception as e:
        print(f"Error in detect_circles: {e}")
        return []

def compute_3d_coordinates(left_circles, right_circles, fx, fy, cx, cy, baseline):
    """计算圆环的三维坐标"""
    detected_circles_3d = []
    for left_circle in left_circles:
        xL, yL, rL = left_circle
        # 找到对应右图中的圆，基于视差计算
        possible_matches = []
        for right_circle in right_circles:
            xR, yR, rR = right_circle
            disparity = xL - xR
            if disparity > 0:
                Z = (fx * baseline) / disparity
                X = (xL - cx) * Z / fx
                Y = (yL - cy) * Z / fy
                possible_matches.append((right_circle, (X, Y, Z), disparity))
        if possible_matches:
            # 选择视差最小的匹配
            best_match = min(possible_matches, key=lambda m: m[2])
            _, (X, Y, Z), _ = best_match
            detected_circles_3d.append((X, Y, Z, xL, yL, rL))
    return detected_circles_3d

def match_circles(tracked_positions, detected_positions, max_distance=1.0):
    """使用匈牙利算法匹配跟踪圆环和检测到的圆环"""
    if len(tracked_positions) == 0 or len(detected_positions) == 0:
        return [], set(), set()

    cost_matrix = np.zeros((len(tracked_positions), len(detected_positions)), dtype=np.float32)
    for i, tracked_pos in enumerate(tracked_positions):
        for j, detected_pos in enumerate(detected_positions):
            distance = math.sqrt(
                (tracked_pos[0] - detected_pos[0])**2 +
                (tracked_pos[1] - detected_pos[1])**2 +
                (tracked_pos[2] - detected_pos[2])**2
            )
            cost_matrix[i, j] = distance

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_tracked_ids = set()
    matched_detected_indices = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_distance:
            matches.append((r, c))
            matched_tracked_ids.add(r)
            matched_detected_indices.add(c)

    return matches, matched_tracked_ids, matched_detected_indices
