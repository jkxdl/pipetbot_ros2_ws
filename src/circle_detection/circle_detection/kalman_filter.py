# circle_detection/kalman_filter.py

import cv2
import numpy as np

class KalmanFilter3D:
    def __init__(self, dt):
        """初始化3D卡尔曼滤波器"""
        self.kf = cv2.KalmanFilter(6, 3)  # 状态维度=6, 测量维度=3

        # 状态转移矩阵 A，根据 dt 动态设置
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0, 0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=np.float32)

        # 测量矩阵 H
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # 过程噪声协方差 Q
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4

        # 测量噪声协方差 R
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2

        # 初始后验状态估计
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def predict(self):
        """预测下一状态"""
        return self.kf.predict()

    def correct(self, measurement):
        """校正当前状态"""
        return self.kf.correct(measurement)

    def update_transition_matrix(self, dt):
        """动态更新状态转移矩阵 A"""
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0, 0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=np.float32)
