import os
import sys
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from pipettingrobot_interfaces.msg import PipettingSceneState, PipettingTaskStatus
from pipettingrobot_interfaces.srv import (
    SetActiveTube,
    SetCircleCount,
    SetTubeVisualState,
    StartPipetting,
)
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QImage
from PySide2.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from pipettingrobot_gui.gpu_image_view import GpuImageView
from pipettingrobot_gui.qt3d_scene import PipettingScene3D


def configure_qt_environment():
    cv2_plugin_markers = ('cv2/qt/plugins', 'opencv_python.libs')
    for env_name in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH'):
        value = os.environ.get(env_name)
        if value and any(marker in value for marker in cv2_plugin_markers):
            os.environ.pop(env_name, None)


class OperatorPanelNode(Node):
    def __init__(self):
        super().__init__('pipetting_operator_panel')
        self.declare_parameter('planning_method', 'rl')
        self.declare_parameter('rack_mesh', 'package://pipettingrobot_gui/meshes/example_test_tube_rack.dae')
        self.declare_parameter('beaker_mesh', 'package://pipettingrobot_gui/meshes/example_beaker.dae')
        self.declare_parameter('tube_mesh', 'package://pipettingrobot_gui/meshes/example_test_tube.dae')
        self.declare_parameter('aubo_type', 'aubo_C5')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('top_camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('stereo_left_topic', '/stereo/left/image_raw')
        self.declare_parameter('stereo_right_topic', '/stereo/right/image_raw')
        self.declare_parameter('camera_preview_fps', 5.0)
        self.declare_parameter('camera_preview_max_width', 480)

        self.scene_state = PipettingSceneState()
        self.scene_state_sub = self.create_subscription(
            PipettingSceneState,
            '/pipetting_ui/scene_state',
            self.scene_state_callback,
            qos_profile_sensor_data,
        )
        self.task_status = PipettingTaskStatus()
        self.task_status_sub = self.create_subscription(
            PipettingTaskStatus,
            '/pipetting/task_status',
            self.task_status_callback,
            qos_profile_sensor_data,
        )
        self.bridge = CvBridge()
        self.camera_preview_interval = 1.0 / max(
            1.0, float(self.get_parameter('camera_preview_fps').value)
        )
        self.camera_preview_max_width = int(
            self.get_parameter('camera_preview_max_width').value
        )
        self.last_camera_update = {
            'top': 0.0,
            'left': 0.0,
            'right': 0.0,
        }
        self.top_camera_frame = None
        self.stereo_left_frame = None
        self.stereo_right_frame = None
        self.image_subscriptions = [
            self.create_subscription(
                Image,
                self.get_parameter('top_camera_topic').value,
                self.top_camera_callback,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Image,
                self.get_parameter('stereo_left_topic').value,
                self.stereo_left_callback,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Image,
                self.get_parameter('stereo_right_topic').value,
                self.stereo_right_callback,
                qos_profile_sensor_data,
            ),
        ]
        self.reset_client = self.create_client(Trigger, '/pipetting_ui/reset_scene')
        self.active_client = self.create_client(SetActiveTube, '/pipetting_ui/set_active_tube')
        self.visual_client = self.create_client(
            SetTubeVisualState, '/pipetting_ui/set_tube_visual_state'
        )
        self.start_client = self.create_client(StartPipetting, '/pipetting/start')
        self.circle_count_client = self.create_client(SetCircleCount, '/pipetting/set_circle_count')

    def scene_state_callback(self, msg: PipettingSceneState):
        self.scene_state = msg

    def task_status_callback(self, msg: PipettingTaskStatus):
        self.task_status = msg

    def _image_msg_to_qimage(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to decode image on {msg.header.frame_id}: {exc}')
            return None

        height, width = cv_image.shape[:2]
        if width > self.camera_preview_max_width:
            scale = self.camera_preview_max_width / float(width)
            cv_image = cv2.resize(
                cv_image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        height, width, channels = cv_image.shape
        bytes_per_line = channels * width
        image = QImage(
            cv_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_BGR888,
        )
        return image.copy()

    def _throttled_image_update(self, stream_name: str, msg: Image):
        now = time.monotonic()
        if now - self.last_camera_update[stream_name] < self.camera_preview_interval:
            return None
        self.last_camera_update[stream_name] = now
        return self._image_msg_to_qimage(msg)

    def top_camera_callback(self, msg: Image):
        image = self._throttled_image_update('top', msg)
        if image is not None:
            self.top_camera_frame = image

    def stereo_left_callback(self, msg: Image):
        image = self._throttled_image_update('left', msg)
        if image is not None:
            self.stereo_left_frame = image

    def stereo_right_callback(self, msg: Image):
        image = self._throttled_image_update('right', msg)
        if image is not None:
            self.stereo_right_frame = image


class OperatorPanelWindow(QMainWindow):
    def __init__(self, node: OperatorPanelNode):
        super().__init__()
        self.node = node
        self.setWindowTitle('Pipetting Robot Operator Panel')
        self.resize(1440, 900)
        self.scene_view = PipettingScene3D(self.node)
        self._build_ui()

        self.spin_timer = QTimer(self)
        self.spin_timer.timeout.connect(self._spin_ros_once)
        self.spin_timer.start(100)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_view)
        self.refresh_timer.start(200)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        main_layout.addLayout(left_col, 0)
        main_layout.addLayout(right_col, 1)

        summary_group = QGroupBox('场景概览')
        summary_layout = QFormLayout(summary_group)
        self.phase_label = QLabel('idle')
        self.rack_label = QLabel('未识别')
        self.beaker_label = QLabel('未识别')
        self.tube_count_label = QLabel('0')
        self.active_label = QLabel('无')
        self.status_message_label = QLabel('ready')
        self.status_message_label.setWordWrap(True)
        summary_layout.addRow('当前阶段', self.phase_label)
        summary_layout.addRow('试管架', self.rack_label)
        summary_layout.addRow('烧杯', self.beaker_label)
        summary_layout.addRow('试管数量', self.tube_count_label)
        summary_layout.addRow('高亮试管', self.active_label)
        summary_layout.addRow('任务信息', self.status_message_label)
        left_col.addWidget(summary_group)

        progress_group = QGroupBox('任务进度')
        progress_layout = QFormLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_detail_label = QLabel('0 / 0')
        self.step_label = QLabel('0 / 10')
        progress_layout.addRow('总进度', self.progress_bar)
        progress_layout.addRow('试管进度', self.progress_detail_label)
        progress_layout.addRow('步骤进度', self.step_label)
        left_col.addWidget(progress_group)

        control_group = QGroupBox('任务控制')
        control_layout = QGridLayout(control_group)
        self.circle_count_spin = QSpinBox()
        self.circle_count_spin.setMinimum(1)
        self.circle_count_spin.setMaximum(999)
        self.circle_count_spin.setValue(50)
        self.planning_method_combo = QComboBox()
        self.planning_method_combo.addItem('RL', 'rl')
        self.planning_method_combo.addItem('Traditional', 'script')
        self.planning_method_combo.addItem('Quintic IK', 'quintic')
        initial_planning_method = self.node.get_parameter('planning_method').value
        combo_index = self.planning_method_combo.findData(initial_planning_method)
        if combo_index >= 0:
            self.planning_method_combo.setCurrentIndex(combo_index)
        self.active_spin = QSpinBox()
        self.active_spin.setMinimum(-1)
        self.active_spin.setMaximum(999)
        self.fill_tube_spin = QSpinBox()
        self.fill_tube_spin.setMinimum(0)
        self.fill_tube_spin.setMaximum(999)
        self.fill_ratio_spin = QDoubleSpinBox()
        self.fill_ratio_spin.setRange(0.0, 1.0)
        self.fill_ratio_spin.setSingleStep(0.1)
        self.fill_ratio_spin.setValue(1.0)

        start_button = QPushButton('开始移液')
        start_button.clicked.connect(self.start_pipetting)
        set_count_button = QPushButton('设置试管数量')
        set_count_button.clicked.connect(self.set_circle_count)
        highlight_button = QPushButton('高亮试管')
        highlight_button.clicked.connect(self.set_active_tube)
        clear_highlight_button = QPushButton('取消高亮')
        clear_highlight_button.clicked.connect(self.clear_active_tube)
        fill_button = QPushButton('标记已排液')
        fill_button.clicked.connect(self.mark_tube_filled)
        empty_button = QPushButton('清空液位')
        empty_button.clicked.connect(self.clear_tube_filled)
        reset_button = QPushButton('重置场景')
        reset_button.clicked.connect(self.reset_scene)

        control_layout.addWidget(QLabel('目标试管数'), 0, 0)
        control_layout.addWidget(self.circle_count_spin, 0, 1)
        control_layout.addWidget(QLabel('规划方式'), 0, 2)
        control_layout.addWidget(self.planning_method_combo, 0, 3)
        control_layout.addWidget(QLabel('试管索引'), 1, 0)
        control_layout.addWidget(self.active_spin, 1, 1)
        control_layout.addWidget(set_count_button, 1, 2)
        control_layout.addWidget(start_button, 1, 3)
        control_layout.addWidget(QLabel('液位试管'), 2, 0)
        control_layout.addWidget(self.fill_tube_spin, 2, 1)
        control_layout.addWidget(highlight_button, 2, 2)
        control_layout.addWidget(clear_highlight_button, 2, 3)
        control_layout.addWidget(QLabel('液位比例'), 3, 0)
        control_layout.addWidget(self.fill_ratio_spin, 3, 1)
        control_layout.addWidget(fill_button, 3, 2)
        control_layout.addWidget(empty_button, 3, 3)
        control_layout.addWidget(reset_button, 4, 0, 1, 4)
        left_col.addWidget(control_group)

        help_group = QGroupBox('3D 说明')
        help_layout = QVBoxLayout(help_group)
        help_text = QLabel(
            '右侧为 Qt3D 原生场景：机器人、试管架、烧杯、试管与液位将随 ROS 状态更新。\n'
            '鼠标左键旋转，滚轮缩放，中键平移。'
        )
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        left_col.addWidget(help_group)
        left_col.addStretch(1)

        scene_container = self.scene_view.create_container(self)
        scene_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_col.addWidget(scene_container, 4)

        camera_group = QGroupBox('相机画面')
        camera_layout = QGridLayout(camera_group)
        self.top_camera_label = self._create_camera_label('顶视 RGB')
        self.left_camera_label = self._create_camera_label('左目 RGB')
        self.right_camera_label = self._create_camera_label('右目 RGB')
        camera_layout.addWidget(self.top_camera_label, 0, 0, 1, 2)
        camera_layout.addWidget(self.left_camera_label, 1, 0)
        camera_layout.addWidget(self.right_camera_label, 1, 1)
        right_col.addWidget(camera_group, 2)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['试管', 'x', 'y', 'z', '液位'])
        right_col.addWidget(self.table, 1)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        right_col.addWidget(self.log_output, 1)

    def _create_camera_label(self, title: str):
        label = GpuImageView(title)
        label.setStyleSheet(
            'background-color: #16181c; color: #c9d1d9; border: 1px solid #30363d;'
        )
        return label

    def _update_camera_label(self, label: GpuImageView, image: QImage | None, placeholder: str):
        if image is None:
            label.set_placeholder(placeholder)
            label.clear_image()
            return
        label.set_placeholder(placeholder)
        label.set_image(image)

    def _spin_ros_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def refresh_view(self):
        state = self.node.scene_state
        task_status = self.node.task_status
        self.phase_label.setText(task_status.phase or state.phase or 'idle')
        self.rack_label.setText('已识别' if state.rack_detected else '未识别')
        self.beaker_label.setText('已识别' if state.beaker_detected else '未识别')
        self.tube_count_label.setText(str(len(state.tubes)))
        self.active_label.setText(str(state.active_tube_index) if state.active_tube_index >= 0 else '无')
        self.status_message_label.setText(task_status.message or task_status.last_error or 'ready')
        self.progress_bar.setValue(int(max(0.0, min(1.0, task_status.progress)) * 100.0))
        current_tube_human = task_status.current_tube_index + 1 if task_status.total_tubes > 0 else 0
        completed_display = min(task_status.completed_tubes, task_status.total_tubes)
        self.progress_detail_label.setText(
            f'{completed_display} / {task_status.total_tubes} (当前 {current_tube_human})'
        )
        self.step_label.setText(f'{task_status.current_step} / {task_status.total_steps}')
        self._update_camera_label(self.top_camera_label, self.node.top_camera_frame, '顶视 RGB\n等待图像...')
        self._update_camera_label(self.left_camera_label, self.node.stereo_left_frame, '左目 RGB\n等待图像...')
        self._update_camera_label(self.right_camera_label, self.node.stereo_right_frame, '右目 RGB\n等待图像...')

        self.table.setRowCount(len(state.tubes))
        for row, tube in enumerate(state.tubes):
            values = [
                tube.label,
                f'{tube.opening.x:.3f}',
                f'{tube.opening.y:.3f}',
                f'{tube.opening.z:.3f}',
                f'{tube.fill_ratio:.2f}' if tube.filled else '0.00',
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if tube.index == state.active_tube_index:
                    item.setBackground(Qt.yellow)
                self.table.setItem(row, col, item)

    def append_log(self, text: str):
        self.log_output.append(text)

    def call_service(self, client, request, description: str):
        if not client.wait_for_service(timeout_sec=0.2):
            self.append_log(f'{description}: service unavailable')
            return
        future = client.call_async(request)
        future.add_done_callback(lambda fut: self._handle_service_response(fut, description))

    def _handle_service_response(self, future, description: str):
        try:
            response = future.result()
        except Exception as exc:
            self.append_log(f'{description}: {exc}')
            return
        message = getattr(response, 'message', '')
        success = getattr(response, 'success', True)
        self.append_log(f'{description}: {"OK" if success else "FAIL"} {message}')

    def reset_scene(self):
        request = Trigger.Request()
        self.call_service(self.node.reset_client, request, 'reset_scene')

    def start_pipetting(self):
        request = StartPipetting.Request()
        request.expected_circle_count = self.circle_count_spin.value()
        request.reset_visual_state = True
        request.planning_method = self.planning_method_combo.currentData()
        self.call_service(self.node.start_client, request, 'start_pipetting')

    def set_circle_count(self):
        request = SetCircleCount.Request()
        request.expected_circle_count = self.circle_count_spin.value()
        self.call_service(self.node.circle_count_client, request, 'set_circle_count')

    def set_active_tube(self):
        request = SetActiveTube.Request()
        request.tube_index = self.active_spin.value()
        self.call_service(self.node.active_client, request, 'set_active_tube')

    def clear_active_tube(self):
        request = SetActiveTube.Request()
        request.tube_index = -1
        self.call_service(self.node.active_client, request, 'clear_active_tube')

    def mark_tube_filled(self):
        request = SetTubeVisualState.Request()
        request.tube_index = self.fill_tube_spin.value()
        request.filled = True
        request.fill_ratio = float(self.fill_ratio_spin.value())
        self.call_service(self.node.visual_client, request, 'set_tube_visual_state')

    def clear_tube_filled(self):
        request = SetTubeVisualState.Request()
        request.tube_index = self.fill_tube_spin.value()
        request.filled = False
        request.fill_ratio = 0.0
        self.call_service(self.node.visual_client, request, 'clear_tube_visual_state')

    def closeEvent(self, event):
        self.scene_view.shutdown()
        super().closeEvent(event)


def main(args=None):
    configure_qt_environment()
    rclpy.init(args=args)
    node = OperatorPanelNode()
    app = QApplication(sys.argv)
    window = OperatorPanelWindow(node)
    window.show()
    try:
        sys.exit(app.exec_())
    finally:
        node.destroy_node()
        rclpy.shutdown()
