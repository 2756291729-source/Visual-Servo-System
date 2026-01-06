# -*- coding: utf-8 -*-
import sys
import os
import time
import threading
import cv2
import numpy as np
from ctypes import *
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTextEdit, QMessageBox, QSlider, QGridLayout,
    QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QSplitter
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
import traceback

# 添加海康SDK路径
sys.path.append(r'c:\Users\QIT_SZTU\PycharmProjects\PythonProject2')

# 尝试导入海康SDK
try:
    from MvCameraControl_class import *
    from PixelType_header import *
    from MvErrorDefine_const import *
    from CamOperation_class import CameraOperation

    HAS_HIKVISION_SDK = True
    print("✅ 成功导入海康SDK及相关模块")
except ImportError as e:
    HAS_HIKVISION_SDK = False
    print(f"❌ 导入海康SDK或CamOperation_class失败: {e}。将仅可使用OpenCV模式。")

# 导入舵机控制模块
try:
    from shuangduoji import STM32SafeController
except ImportError:
    print("❌ 无法导入舵机控制模块 'shuangduoji'")
    STM32SafeController = None

# 导入像素误差到毫米/角度转换模块
from wuchajaiodu import PixelToMMConverter


class FuzzyController:
    """
    模糊控制器 - 高精度模式 (目标误差 < 4px)
    """

    def __init__(self):
        # 1. 优化核心死区（ZE）和小误差范围，减少输出跳跃
        self.input_ranges = {
            'NB': (-300, -80),  # 扩大大误差范围，加速响应
            'NM': (-100, -30),  # 增加"负中"档，细化中等误差
            'NS': (-40, -4),  # 负小误差范围扩展到-4px
            'ZE': (-4, 4),  # 核心死区±4px（目标精度内）
            'PS': (4, 40),  # 正小误差范围扩展到4px
            'PM': (30, 100),  # 增加"正中"档，细化中等误差
            'PB': (80, 300)  # 扩大大误差范围，加速响应
        }

        # 2. 优化输出值，在小误差范围内使用更小的调整量，减少震荡
        self.output_values = {
            'NB': -20,  # 最大负向调整（加速大误差修正）
            'NM': -8,  # 中等负向调整
            'NS': -1,  # 负小误差使用更小的调整量
            'ZE': 0,  # 死区不调整
            'PS': 1,  # 正小误差使用更小的调整量
            'PM': 8,  # 中等正向调整
            'PB': 20  # 最大正向调整（加速大误差修正）
        }

    def _trimf(self, x, params):
        """三角形隶属度函数"""
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        return 0.0

    def _trapmf(self, x, params):
        """梯形隶属度函数"""
        a, b, c, d = params
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return 1.0
        elif c <= x < d:
            return (d - x) / (d - c)
        return 0.0

    def get_membership(self, error):
        """优化隶属度函数计算，提高误差区分度"""

        # NB: 梯形 (-300, -300, -100, -80)
        mu_NB = self._trapmf(error, (-300, -300, -100, -80))

        # NM: 三角形 (-100, -60, -30) -> 峰值在-60
        mu_NM = self._trimf(error, (-100, -60, -30))

        # NS: 三角形 (-40, -20, -4) -> 峰值在-20，调整结束点为-4px
        mu_NS = self._trimf(error, (-40, -20, -4))

        # ZE: 三角形 (-4, 0, 4) -> 核心死区±4px
        mu_ZE = self._trimf(error, (-4, 0, 4))

        # PS: 三角形 (4, 20, 40) -> 峰值在20，调整起始点为4px
        mu_PS = self._trimf(error, (4, 20, 40))

        # PM: 三角形 (30, 60, 100) -> 峰值在60
        mu_PM = self._trimf(error, (30, 60, 100))

        # PB: 梯形 (80, 100, 300, 300)
        mu_PB = self._trapmf(error, (80, 100, 300, 300))

        return {
            'NB': mu_NB, 'NM': mu_NM, 'NS': mu_NS,
            'ZE': mu_ZE, 'PS': mu_PS, 'PM': mu_PM, 'PB': mu_PB
        }

    def compute(self, error):
        """执行模糊推理和解模糊"""
        # 1. 模糊化
        memberships = self.get_membership(error)

        # 2. 模糊推理 & 3. 解模糊 (加权平均法)
        numerator = 0.0
        denominator = 0.0

        for label, mu in memberships.items():
            numerator += mu * self.output_values[label]
            denominator += mu

        if denominator == 0:
            return 0

        output = numerator / denominator
        return output


class ChessboardDetector:
    def __init__(self, chessboard_size=(11, 8), square_size=30):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.last_detected_center = None
        self.detection_stable_count = 0
        self.stable_threshold = 3
        self.camera_matrix = None
        self.dist_coeffs = None
        self.load_camera_calibration()
        self.detect_time = 0.0
        # 初始化像素误差到角度转换器
        self.pixel_to_mm_converter = PixelToMMConverter()
        # 设置默认工作距离（可根据实际情况调整）
        self.pixel_to_mm_converter.set_distance(1500.0)
        
        # 3D世界坐标点准备
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        # 计算标定板中心在3D世界坐标中的位置
        self.calib_board_center_3d = np.array([
            (self.chessboard_size[0] - 1) * self.square_size / 2,
            (self.chessboard_size[1] - 1) * self.square_size / 2,
            0
        ], dtype=np.float32)
        
        # 当前测量的工作距离
        self.current_distance = 1500.0

    def load_camera_calibration(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            calib_file = os.path.join(current_dir, "camera_calibration_results.npz")
            if os.path.exists(calib_file):
                with np.load(calib_file) as X:
                    self.camera_matrix, self.dist_coeffs = X['camera_matrix'], X['dist_coeffs']
                    print("✅ 成功加载相机标定参数")
            else:
                self.camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
                print("⚠️ 未找到标定文件，使用默认相机参数")
        except Exception as e:
            print(f"❌ 加载相机标定参数失败: {e}")
            self.camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def calculate_pose(self, corners, frame):
        """Calculate pose and distance using solvePnP"""
        try:
            if self.camera_matrix is None or self.dist_coeffs is None:
                # 如果相机参数无效，使用默认距离
                self.current_distance = 1500.0
                self.pixel_to_mm_converter.set_distance(self.current_distance)
                return self.current_distance
            
            success, rvec, tvec = cv2.solvePnP(
                self.objp, corners, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Calculate object distance (mm)
                distance = np.sqrt(tvec[0][0] ** 2 + tvec[1][0] ** 2 + tvec[2][0] ** 2)
                
                # 确保距离为正数
                if distance > 0:
                    # Update current distance
                    self.current_distance = distance
                    # Update the converter's distance
                    self.pixel_to_mm_converter.set_distance(distance)
                return self.current_distance
            else:
                # 如果solvePnP失败，保持当前距离不变
                return self.current_distance
        except Exception as e:
            print(f"Pose calculation error: {e}")
            # 异常情况下，使用默认距离
            self.current_distance = 1500.0
            self.pixel_to_mm_converter.set_distance(self.current_distance)
            return self.current_distance

    def detect_chessboard(self, frame):
        start_time = time.time()
        try:
            if frame is None: return None, None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, find_flags)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                center = (int(np.mean(corners2[:, 0, 0])), int(np.mean(corners2[:, 0, 1])))

                if self.last_detected_center is None or np.linalg.norm(
                        np.array(center) - np.array(self.last_detected_center)) > 20:
                    self.detection_stable_count = 1
                else:
                    self.detection_stable_count += 1
                self.last_detected_center = center

                # Calculate pose and distance
                self.calculate_pose(corners2, frame)

                if self.detection_stable_count >= self.stable_threshold:
                    return center, corners2

            return None, None
        finally:
            self.detect_time = (time.time() - start_time) * 1000

    def get_image_center(self, frame):
        if frame is None: return (640, 480)
        height, width = frame.shape[:2]
        return (width // 2, height // 2)


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, float)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    target_info = pyqtSignal(object, tuple)
    time_stats_updated = pyqtSignal(float, float, float, float, float, float, float, float)

    def __init__(self, camera_index=0, desired_fps=100.0, use_hikvision=True, enable_undistortion=True,
                 processing_scale=0.5):
        super().__init__()
        self.camera_index = camera_index
        self.desired_fps = desired_fps
        self.use_hikvision = use_hikvision
        self.enable_undistortion = enable_undistortion
        self.processing_scale = processing_scale
        self.running = False
        self.camera = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.chessboard_detector = ChessboardDetector()
        self.capture_time = 0.0
        self.control_time = 0.0
        self.servo_time = 0.0
        self.frame_timestamp = 0.0

    def run(self):
        self.running = True
        if not self._init_camera():
            self.error_occurred.emit("相机初始化失败")
            return

        last_frame_time = time.time()

        try:
            while self.running:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                wait_time = (1.0 / self.desired_fps) - elapsed
                if wait_time > 0.001:
                    time.sleep(wait_time)
                last_frame_time = time.time()

                self.capture_time = 0.0
                self.chessboard_detector.detect_time = 0.0
                self.control_time = 0.0
                self.servo_time = 0.0

                frame = self._get_frame()

                if frame is not None:
                    if self.enable_undistortion and self.chessboard_detector.camera_matrix is not None:
                        frame = cv2.undistort(frame, self.chessboard_detector.camera_matrix,
                                              self.chessboard_detector.dist_coeffs, None, None)

                    image_center = self.chessboard_detector.get_image_center(frame)

                    if self.processing_scale < 1.0:
                        h, w = frame.shape[:2]
                        new_w = max(1, int(w * self.processing_scale))
                        new_h = max(1, int(h * self.processing_scale))
                        small_frame = cv2.resize(frame, (new_w, new_h))
                    else:
                        small_frame = frame

                    center, corners = self.chessboard_detector.detect_chessboard(small_frame)

                    cv2.circle(frame, image_center, 10, (0, 255, 0), 2)
                    cv2.line(frame, (image_center[0] - 15, image_center[1]), (image_center[0] + 15, image_center[1]),
                             (0, 255, 0), 2)
                    cv2.line(frame, (image_center[0], image_center[1] - 15), (image_center[0], image_center[1] + 15),
                             (0, 255, 0), 2)

                    if center:
                        scale = self.processing_scale if self.processing_scale > 0 else 1.0
                        original_center = (int(center[0] / scale), int(center[1] / scale))
                        self.target_info.emit(original_center, image_center)

                        if corners is not None:
                            original_corners = corners / scale
                            try:
                                cv2.drawChessboardCorners(frame, self.chessboard_detector.chessboard_size,
                                                          original_corners.astype(np.float32), True)
                            except Exception as draw_e:
                                print(f"绘制角点时出错: {draw_e}")
                        cv2.circle(frame, original_center, 12, (0, 0, 255), -1)
                        cv2.line(frame, image_center, original_center, (0, 165, 255), 2)
                        # 计算并显示实时误差
                        error_x = original_center[0] - image_center[0]
                        error_y = original_center[1] - image_center[1]
                        error_magnitude = np.sqrt(error_x ** 2 + error_y ** 2)
                        
                        # 计算角度误差（使用实时测量的工作距离）
                        deg_error_x, deg_error_y = self.chessboard_detector.pixel_to_mm_converter.pixel_error_to_deg(error_x, error_y)
                        deg_error_magnitude = np.sqrt(deg_error_x ** 2 + deg_error_y ** 2)
                        
                        # 获取实时测量的工作距离
                        real_time_distance = self.chessboard_detector.current_distance
                        
                        # 显示像素误差、角度误差和实时工作距离
                        error_text = f"Error: X={error_x:.1f}px, Y={error_y:.1f}px, Mag={error_magnitude:.1f}px"
                        cv2.putText(frame, error_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 2)
                        
                        # 调整角度误差的显示格式，去掉特殊字符，使用更简单的文本
                        angle_text = f"Angle: X={deg_error_x:.2f}deg, Y={deg_error_y:.2f}deg, Mag={deg_error_magnitude:.2f}deg"
                        cv2.putText(frame, angle_text, (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 165, 0), 2)
                        
                        distance_text = f"Distance: {real_time_distance:.1f}mm"
                        cv2.putText(frame, distance_text, (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
                    else:
                        self.target_info.emit(None, image_center)
                        text = "Target Not Detected..."
                        (wt, ht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        cv2.putText(frame, text, (image_center[0] - wt // 2, image_center[1] - ht // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    self.frame_timestamp = time.time()
                    self.frame_ready.emit(frame, self.frame_timestamp)

                    self.frame_count += 1
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count / (current_time - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = current_time

        finally:
            self._close_camera()

    def _init_camera(self):
        if self.use_hikvision and HAS_HIKVISION_SDK:
            return self._init_hikvision_camera()
        return self._init_opencv_camera()

    def _init_hikvision_camera(self):
        for retry in range(3):
            try:
                obj_cam = MvCamera()
                st_device_list = MV_CC_DEVICE_INFO_LIST()
                n_ret = obj_cam.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, st_device_list)
                if n_ret != 0: raise Exception(f"枚举设备失败, ret=0x{n_ret:x}")
                if st_device_list.nDeviceNum == 0: raise Exception("未找到相机设备")

                self.camera = CameraOperation(obj_cam, st_device_list)
                self.camera.n_connect_num = self.camera_index
                ret = self.camera.Open_device()
                if ret != 0: raise Exception(f"打开设备失败, ret=0x{ret:x}")

                self.setup_auto_exposure_gain()
                ret = self.camera.Start_grabbing(0)
                if ret != 0: raise Exception(f"开始采集失败, ret=0x{ret:x}")

                self.status_changed.emit("海康相机启动成功")
                return True
            except Exception as e:
                self.error_occurred.emit(f"海康相机初始化失败({retry + 1}/3): {e}")
                if retry < 2: time.sleep(1)
        return False

    def _init_opencv_camera(self):
        try:
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened(): raise Exception(f"无法打开相机ID: {self.camera_index}")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.status_changed.emit(f"OpenCV相机 {self.camera_index} 启动成功")
            return True
        except Exception as e:
            self.error_occurred.emit(f"OpenCV初始化失败: {e}")
            return False

    def _get_frame(self):
        start_time = time.time()
        try:
            if self.use_hikvision and HAS_HIKVISION_SDK and self.camera:
                if not hasattr(self, 'camera') or self.camera is None: return None
                if not hasattr(self.camera, 'buf_save_image') or not hasattr(self.camera, 'st_frame_info'): return None
                if self.camera.buf_save_image is None: return None
                try:
                    info = self.camera.st_frame_info
                    if not hasattr(info, 'nWidth') or info.nWidth <= 0 or not hasattr(info,
                                                                                      'nHeight') or info.nHeight <= 0: return None
                    img_buff = np.frombuffer(self.camera.buf_save_image, dtype=np.uint8, count=info.nFrameLen)
                    expected_size = info.nHeight * info.nWidth
                    if info.enPixelType == PixelType_Gvsp_RGB8_Packed: expected_size *= 3
                    if img_buff.size < expected_size: return None
                    pixel_type = info.enPixelType
                    img_shaped = img_buff.reshape(
                        (info.nHeight, info.nWidth) if pixel_type != PixelType_Gvsp_RGB8_Packed else (info.nHeight,
                                                                                                      info.nWidth, 3))
                    if pixel_type == PixelType_Gvsp_Mono8:
                        return cv2.cvtColor(img_shaped, cv2.COLOR_GRAY2BGR)
                    elif pixel_type == PixelType_Gvsp_BayerRG8:
                        return cv2.cvtColor(img_shaped, cv2.COLOR_BAYER_RG2BGR)
                    elif pixel_type == PixelType_Gvsp_RGB8_Packed:
                        return cv2.cvtColor(img_shaped, cv2.COLOR_RGB2BGR)
                    else:
                        self.error_occurred.emit(f"未支持的像素格式: {pixel_type}");
                        return None
                except Exception as e:
                    return None
            elif hasattr(self, 'camera') and self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                return frame if ret else None
            return None
        finally:
            self.capture_time = (time.time() - start_time) * 1000

    def _close_camera(self):
        if self.camera:
            try:
                if self.use_hikvision and HAS_HIKVISION_SDK:
                    if hasattr(self.camera, 'Stop_grabbing'): self.camera.Stop_grabbing()
                    if hasattr(self.camera, 'Close_device'): self.camera.Close_device()
                else:
                    self.camera.release()
                self.status_changed.emit("相机已关闭")
            except Exception as e:
                print(f"关闭相机时出错: {e}")
            finally:
                self.camera = None

    def setup_camera_parameters(self):
        try:
            if self.camera and hasattr(self.camera, 'obj_cam'):
                self.camera.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
                self.camera.obj_cam.MV_CC_SetFloatValue("ExposureTime", 1000.0)
                self.camera.obj_cam.MV_CC_SetEnumValue("GainAuto", 0)
                self.camera.obj_cam.MV_CC_SetFloatValue("Gain", 0.0)
                self.camera.obj_cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 0)
                self.camera.obj_cam.MV_CC_SetEnumValue("AcquisitionFrameRateAuto", 0)
                self.camera.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 100.0)
                self.status_changed.emit("相机参数设置成功：1ms曝光，100fps帧率")
        except Exception as e:
            self.error_occurred.emit(f"设置相机参数失败: {e}")
            print(f"警告：部分相机参数设置失败，但继续初始化: {e}")

    def setup_auto_exposure_gain(self):
        self.setup_camera_parameters()

    def set_exposure_time(self, exposure_time):
        try:
            if self.camera and hasattr(self.camera, 'obj_cam'):
                self.camera.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
                self.camera.obj_cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
                self.status_changed.emit(f"曝光时间设置为: {exposure_time}μs")
                return True
            else:
                self.error_occurred.emit("相机未初始化，无法设置曝光时间")
                return False
        except Exception as e:
            self.error_occurred.emit(f"设置曝光时间失败: {e}")
            return False

    def stop(self):
        self.running = False
        self.wait(1000)


class VisualServoController:
    """视觉伺服控制器 - 集成卡尔曼滤波器与模糊控制"""

    def __init__(self):
        self.servo_controller = None
        self.servo_ids = ["000", "001"]
        self.target_position = (640, 360)
        self.current_angles = [1500, 1500]
        self.is_enabled = False
        self.mutex = QMutex()

        # 控制参数
        self.last_command_time = 0
        self.wait_time_after_move = 1.5

        # 【修改点】硬性死区设为 3 像素 (为了实现 < 4 的控制)
        self.dead_zone_pixels = 3

        self.settled_count = 0
        self.settle_threshold = 5

        # 搜索状态
        self.search_state = {'x_angle': 1500, 'y_angle': 1500, 'x_direction': 1, 'step': 50}
        self.search_limits = {'x_min': 500, 'x_max': 2500, 'y_min': 1000, 'y_max': 2000}

        # 时间统计变量
        self.control_time = 0.0
        self.servo_time = 0.0

        # 模糊控制器
        self.fuzzy_controller = FuzzyController()
        self.use_fuzzy_control = True

    def connect_servo(self, port_name=None):
        print("【调试】进入 connect_servo 方法...")
        if STM32SafeController is None:
            print("【调试】错误：STM32SafeController 未导入。")
            return False
        try:
            print("【调试】步骤 1: 正在创建 STM32SafeController 实例...")
            self.servo_controller = STM32SafeController()
            print("【调试】步骤 1: STM32SafeController 实例创建成功。")

            print(f"【调试】步骤 2: 即将调用 connect_without_reset (端口: {port_name})...")
            connection_success = self.servo_controller.connect_without_reset(port_name)
            print(f"【调试】步骤 2a: connect_without_reset 调用返回: {connection_success}")

            if connection_success:
                print("【调试】步骤 2b: connect_without_reset 调用成功。")
                print("【调试】步骤 3: 等待 0.5 秒...")
                time.sleep(0.5)
                print("【调试】步骤 3: 等待结束。connect_servo 即将返回 True。")
                return True
            else:
                print("【调试】步骤 2b: connect_without_reset 返回 False (连接失败)。请检查端口。")
                self.servo_controller = None
                return False
        except Exception as e:
            print(f"【调试】在 connect_servo 过程中捕获到异常: {e}")
            traceback.print_exc()
            self.servo_controller = None
            return False
        finally:
            print("【调试】退出 connect_servo 方法。")

    def set_control_params(self, wait_time, dead_zone, settle_frames):
        with QMutexLocker(self.mutex):
            self.wait_time_after_move = wait_time
            self.dead_zone_pixels = dead_zone
            self.settle_threshold = settle_frames

    def enable(self):
        with QMutexLocker(self.mutex):
            self.is_enabled = True
            self.settled_count = 0

    def disable(self):
        with QMutexLocker(self.mutex):
            self.is_enabled = False

    def calculate_and_send_commands(self, current_position):
        self.control_time = 0.0
        self.servo_time = 0.0

        try:
            with QMutexLocker(self.mutex):
                if not self.is_enabled:
                    return False, "模糊跟踪未启用", self.control_time, self.servo_time
                if self.servo_controller is None:
                    return False, "舵机未连接", self.control_time, self.servo_time

                calc_start_time = time.time()

                filtered_position = current_position

                error_x = filtered_position[0] - self.target_position[0]
                error_y = filtered_position[1] - self.target_position[1]
                error_magnitude = np.sqrt(error_x ** 2 + error_y ** 2)

                is_stable = False
                algo_name = "模糊"
                status_message = f"{algo_name}跟踪中... 误差: {error_magnitude:.1f}px"

                if error_magnitude < self.dead_zone_pixels:
                    self.settled_count += 1
                    if self.settled_count >= self.settle_threshold:
                        is_stable = True
                        status_message = f"目标已在静区内 (连续 {self.settled_count} 帧 < {self.dead_zone_pixels}px)。持续跟踪..."
                    else:
                        self.control_time = (time.time() - calc_start_time) * 1000
                        return False, f"进入静区，等待稳定... ({self.settled_count}/{self.settle_threshold})", self.control_time, self.servo_time
                else:
                    self.settled_count = 0

                # ----------------------------------------------------
                # 控制算法分支：模糊控制
                # ----------------------------------------------------
                # >>> 模糊控制逻辑 - 高精度版 <<<

                # 1. 硬性死区拦截 (4px)，与模糊控制器死区保持一致
                if error_magnitude < 4:
                    self.control_time = (time.time() - calc_start_time) * 1000
                    return False, f"模糊控制死区 (<4px) 误差:{error_magnitude:.1f}", self.control_time, self.servo_time

                # 2. 优化动态等待时间，增加基础等待时间，避免频繁调整
                base_wait_time = 0.3  # 增加基础等待时间，减少震荡
                dynamic_wait = base_wait_time + (error_magnitude / 100.0) * 0.3
                dynamic_wait = min(dynamic_wait, 0.6)  # 延长最大等待时间

                if time.time() - self.last_command_time < dynamic_wait:
                    self.control_time = (time.time() - calc_start_time) * 1000
                    msg = status_message if is_stable else f"等待舵机稳定... (Fuzzy wait: {dynamic_wait:.2f}s)"
                    return False, msg, self.control_time, self.servo_time

                # 3. 计算模糊输出
                adjust_val_x = self.fuzzy_controller.compute(error_x)
                adjust_val_y = self.fuzzy_controller.compute(error_y)

                # 4. 符号修正
                sign_x, sign_y = -1.0, 1.0

                # 5. 使用更小的调整量，避免跳跃，减少震荡
                # 不再使用round()函数，而是直接取整，保留更小的调整量
                adjust_x = int(sign_x * adjust_val_x)
                adjust_y = int(sign_y * adjust_val_y)

                if adjust_x == 0 and adjust_y == 0:
                    self.control_time = (time.time() - calc_start_time) * 1000
                    return False, f"模糊软缓冲 (误差{error_magnitude:.1f}不足以触发移动)", self.control_time, self.servo_time

                # 应用角度限制
                new_angle_x = np.clip(self.current_angles[0] + adjust_x, 500, 2500)
                new_angle_y = np.clip(self.current_angles[1] + adjust_y, 500, 2500)

                self.control_time = (time.time() - calc_start_time) * 1000
                self.current_angles = [new_angle_x, new_angle_y]

                fuzzy_info = " [Fuzzy]"
                cmd = f"{{#{self.servo_ids[0]}P{new_angle_x}T1000!#{self.servo_ids[1]}P{new_angle_y}T1000!}}"

                servo_start_time = time.time()
                try:
                    # 只有当计算出的调整量不为0时才发送指令，避免无效通讯
                    if adjust_x == 0 and adjust_y == 0:
                        return False, f"保持位置 (无调整量) {fuzzy_info}", self.control_time, 0

                    if self.servo_controller.send_cmd(cmd):
                        self.servo_time = (time.time() - servo_start_time) * 1000
                        self.last_command_time = time.time()

                        msg = f"跟踪中{fuzzy_info}... 误差X:{error_x:.1f}, Y:{error_y:.1f}"
                        if is_stable:
                            msg = status_message + fuzzy_info

                        return True, msg, self.control_time, self.servo_time
                    else:
                        self.servo_time = (time.time() - servo_start_time) * 1000
                        return False, "发送指令失败", self.control_time, self.servo_time
                except Exception as servo_e:
                    self.servo_time = (time.time() - servo_start_time) * 1000
                    return False, f"发送指令出错: {servo_e}", self.control_time, self.servo_time
        except Exception as e:
            print(f"calculate_and_send_commands 出错: {e}")
            traceback.print_exc()
            return False, f"控制计算出错: {e}", self.control_time, self.servo_time

    def search_step(self):
        with QMutexLocker(self.mutex):
            if self.servo_controller is None: return False, "舵机未连接"
            if time.time() - self.last_command_time < 0.5:
                return False, "等待搜索移动完成..."

            state = self.search_state
            limits = self.search_limits

            next_x_angle = state['x_angle'] + state['step'] * state['x_direction']

            if (state['x_direction'] == 1 and next_x_angle > limits['x_max']) or \
                    (state['x_direction'] == -1 and next_x_angle < limits['x_min']):
                state['x_direction'] *= -1
                state['x_angle'] = np.clip(state['x_angle'], limits['x_min'], limits['x_max'])
                state['x_angle'] += state['step'] * state['x_direction']
            else:
                state['x_angle'] = next_x_angle

            state['x_angle'] = np.clip(state['x_angle'], limits['x_min'], limits['x_max'])
            y_angle = 1500

            cmd = f"{{#{self.servo_ids[0]}P{state['x_angle']}T500!#{self.servo_ids[1]}P{y_angle}T500!}}"

            if self.servo_controller.send_cmd(cmd):
                self.current_angles = [state['x_angle'], y_angle]
                self.last_command_time = time.time()
                return True, f"水平搜索中... 指令: {cmd}"
            else:
                return False, "发送搜索指令失败"

    def reset_search(self):
        with QMutexLocker(self.mutex):
            center_x = 1500
            center_y = 1500
            self.search_state = {
                'x_angle': center_x, 'y_angle': center_y, 'x_direction': 1, 'step': 50
            }
            self.current_angles = [center_x, center_y]
            cmd = f"{{#{self.servo_ids[0]}P{center_x}T1000!#{self.servo_ids[1]}P{center_y}T1000!}}"
            if self.servo_controller:
                self.servo_controller.send_cmd(cmd)
                time.sleep(1.1)
                self.last_command_time = time.time()


class VisualServoUI(QMainWindow):
    STATE_IDLE = 0
    STATE_SEARCHING = 1
    STATE_TRACKING = 2
    STATE_STABILIZED = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视觉伺服系统 - 集成卡尔曼滤波与模糊控制")
        self.setGeometry(100, 100, 1400, 900)
        self.camera_worker = None
        self.visual_servo = VisualServoController()
        self.current_state = self.STATE_IDLE
        self.target_lost_counter = 0
        self.target_lost_threshold = 15
        self.search_timer = QTimer(self)
        self.search_timer.timeout.connect(self.execute_search_step)
        self.search_interval = 600
        self.init_ui()
        self.on_control_params_changed()
        self.is_camera_running = False
        self.is_servo_connected = False
        self.last_frame_timestamp = 0.0
        self.thread_comm_delay = 0.0
        self.ui_display_delay = 0.0
        self.last_ui_update_start = 0.0
        
        # 数据收集相关变量
        self.data_collection = []  # 存储收集的数据
        self.data_count = 0  # 当前收集的组数
        self.max_data_count = 10  # 最大收集组数
        self.tracking_start_time = 0.0  # 开始追踪的时间
        self.data_file = "tracking_time_data.txt"  # 数据保存文件

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        control_panel = QGroupBox("控制与状态")
        control_panel.setMaximumWidth(500)
        control_layout = QVBoxLayout(control_panel)

        cam_group = QGroupBox("相机控制")
        cam_layout = QGridLayout(cam_group)
        cam_layout.addWidget(QLabel("相机模式:"), 0, 0)
        self.combo_camera_mode = QComboBox()
        if HAS_HIKVISION_SDK:
            self.combo_camera_mode.addItem("海康SDK", True)
        self.combo_camera_mode.addItem("OpenCV", False)
        cam_layout.addWidget(self.combo_camera_mode, 0, 1)
        self.btn_start_camera = QPushButton("启动相机")
        self.btn_start_camera.clicked.connect(self.toggle_camera)
        cam_layout.addWidget(self.btn_start_camera, 1, 0, 1, 2)

        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("曝光时间:"))
        self.slider_exposure = QSlider(Qt.Horizontal)
        self.slider_exposure.setMinimum(100)
        self.slider_exposure.setMaximum(100000)
        self.slider_exposure.setSingleStep(100)
        self.slider_exposure.setValue(15000)
        self.slider_exposure.valueChanged.connect(self.on_exposure_changed)
        exposure_layout.addWidget(self.slider_exposure)
        self.label_exposure_value = QLabel("15000μs")
        exposure_layout.addWidget(self.label_exposure_value)

        btn_dec_exposure = QPushButton("-500μs")
        btn_dec_exposure.clicked.connect(lambda: self.adjust_exposure(-500))
        btn_inc_exposure = QPushButton("+500μs")
        btn_inc_exposure.clicked.connect(lambda: self.adjust_exposure(500))
        exposure_layout.addWidget(btn_dec_exposure)
        exposure_layout.addWidget(btn_inc_exposure)

        cam_layout.addLayout(exposure_layout, 2, 0, 1, 2)
        control_layout.addWidget(cam_group)

        perf_group = QGroupBox("性能与精度")
        perf_layout = QGridLayout(perf_group)
        self.check_undistort = QCheckBox("启用畸变校正")
        self.check_undistort.setChecked(True)
        perf_layout.addWidget(self.check_undistort, 0, 0, 1, 2)
        perf_layout.addWidget(QLabel("处理分辨率:"), 1, 0)
        self.combo_scale = QComboBox()
        self.combo_scale.addItem("完整 (精度高)", 1.0)
        self.combo_scale.addItem("1/2 (推荐)", 0.5)
        self.combo_scale.addItem("1/4 (速度快)", 0.25)
        self.combo_scale.setCurrentIndex(1)
        perf_layout.addWidget(self.combo_scale, 1, 1)

        control_layout.addWidget(perf_group)

        servo_group = QGroupBox("模糊伺服控制")
        servo_layout = QGridLayout(servo_group)
        self.btn_connect_servo = QPushButton("连接舵机")
        self.btn_connect_servo.clicked.connect(self.toggle_servo)
        servo_layout.addWidget(self.btn_connect_servo, 0, 0)
        self.btn_enable_servo = QPushButton("开始搜索/跟踪")
        self.btn_enable_servo.setCheckable(True)
        self.btn_enable_servo.clicked.connect(self.toggle_search_tracking)
        self.btn_enable_servo.setEnabled(False)
        servo_layout.addWidget(self.btn_enable_servo, 0, 1)
        self.btn_reset_servos = QPushButton("舵机复位")
        self.btn_reset_servos.clicked.connect(self.reset_servos_to_center)
        self.btn_reset_servos.setEnabled(False)
        servo_layout.addWidget(self.btn_reset_servos, 0, 2)

        control_layout.addWidget(servo_group)

        fine_group = QGroupBox("精细控制参数")
        fine_layout = QGridLayout(fine_group)
        fine_layout.addWidget(QLabel("目标静区(px):"), 0, 0)
        self.spin_dead_zone = QSpinBox()
        self.spin_dead_zone.setRange(1, 20)
        self.spin_dead_zone.setValue(5)
        self.spin_dead_zone.valueChanged.connect(self.on_control_params_changed)
        fine_layout.addWidget(self.spin_dead_zone, 0, 1)

        fine_layout.addWidget(QLabel("稳定所需帧数:"), 1, 0)
        self.spin_settle_frames = QSpinBox()
        self.spin_settle_frames.setRange(1, 10)
        self.spin_settle_frames.setValue(5)
        self.spin_settle_frames.valueChanged.connect(self.on_control_params_changed)
        fine_layout.addWidget(self.spin_settle_frames, 1, 1)

        fine_layout.addWidget(QLabel("控制等待(s):"), 2, 0)
        self.spin_wait_time = QDoubleSpinBox()
        self.spin_wait_time.setRange(0.0, 1.0)
        self.spin_wait_time.setSingleStep(0.05)
        self.spin_wait_time.setValue(1.5)
        self.spin_wait_time.valueChanged.connect(self.on_control_params_changed)
        fine_layout.addWidget(self.spin_wait_time, 2, 1)
        control_layout.addWidget(fine_group)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFixedHeight(150)
        control_layout.addWidget(QLabel("状态日志:"))
        control_layout.addWidget(self.status_text)

        self.image_label = QLabel("等待图像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMinimumSize(1024, 576)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.image_label)

    def set_state(self, new_state):
        if self.current_state == new_state:
            return

        if self.current_state == self.STATE_SEARCHING:
            self.search_timer.stop()

        self.current_state = new_state
        state_str = ["空闲", "搜索中", "跟踪中", "已稳定"][new_state]
        self.update_status(f"系统状态 -> {state_str}")

        if new_state == self.STATE_SEARCHING:
            self.visual_servo.reset_search()
            self.visual_servo.disable()
            self.btn_enable_servo.setChecked(True)
            self.btn_enable_servo.setText("停止搜索/跟踪")
        elif new_state == self.STATE_TRACKING:
            self.visual_servo.enable()
            self.btn_enable_servo.setChecked(True)
            self.btn_enable_servo.setText("停止搜索/跟踪")
        elif new_state == self.STATE_STABILIZED:
            self.visual_servo.disable()
            self.btn_enable_servo.setChecked(False)
            self.btn_enable_servo.setText("开始搜索/跟踪")
        elif new_state == self.STATE_IDLE:
            self.visual_servo.disable()
            self.btn_enable_servo.setChecked(False)
            self.btn_enable_servo.setText("开始搜索/跟踪")

    def toggle_camera(self):
        if not self.is_camera_running:
            use_hik = self.combo_camera_mode.currentData()
            enable_undistort = self.check_undistort.isChecked()
            processing_scale = self.combo_scale.currentData()

            self.camera_worker = CameraWorker(use_hikvision=use_hik,
                                              enable_undistortion=enable_undistort,
                                              processing_scale=processing_scale)
            self.camera_worker.frame_ready.connect(self.update_frame)
            self.camera_worker.status_changed.connect(self.update_status)
            self.camera_worker.error_occurred.connect(self.show_error)
            self.camera_worker.target_info.connect(self.handle_target_update)
            self.camera_worker.time_stats_updated.connect(self.handle_time_stats)
            self.camera_worker.start()

            self.is_camera_running = True
            self.btn_start_camera.setText("停止相机")
            self.combo_camera_mode.setEnabled(False)
            self.combo_scale.setEnabled(False)
            self.update_status(f"以 {self.combo_camera_mode.currentText()} 模式启动...")

            QTimer.singleShot(500, lambda: self.on_exposure_changed(self.slider_exposure.value()))
        else:
            if self.camera_worker:
                self.camera_worker.stop()
            self.is_camera_running = False
            self.btn_start_camera.setText("启动相机")
            self.combo_camera_mode.setEnabled(True)
            self.combo_scale.setEnabled(True)
            self.image_label.setText("相机已停止")
            self.update_status("相机已停止")
            self.set_state(self.STATE_IDLE)

    def toggle_servo(self):
        if not self.is_servo_connected:
            try:
                if self.visual_servo.connect_servo():
                    self.is_servo_connected = True
                    self.btn_connect_servo.setText("断开舵机")
                    self.btn_enable_servo.setEnabled(True)
                    self.btn_reset_servos.setEnabled(True)
                    self.update_status("舵机连接成功")
                    self.visual_servo.reset_search()
                else:
                    self.show_error("舵机连接失败 (connect_servo 返回 False)")
            except Exception as e:
                self.show_error(f"连接舵机时崩溃: {e}")
                traceback.print_exc()
        else:
            self.set_state(self.STATE_IDLE)
            self.visual_servo.disable()
            self.btn_enable_servo.setChecked(False)
            self.btn_enable_servo.setText("开始搜索/跟踪")
            try:
                if self.visual_servo.servo_controller and hasattr(self.visual_servo.servo_controller, 'close'):
                    self.visual_servo.servo_controller.close()
            except Exception as e:
                print(f"关闭舵机连接时出错: {e}")
            self.visual_servo.servo_controller = None
            self.is_servo_connected = False
            self.btn_connect_servo.setText("连接舵机")
            self.btn_enable_servo.setEnabled(False)
            self.btn_reset_servos.setEnabled(False)
            self.update_status("舵机已断开")

    def toggle_search_tracking(self):
        if self.btn_enable_servo.isChecked():
            if not self.is_servo_connected:
                self.show_error("请先连接舵机！")
                self.btn_enable_servo.setChecked(False)
                return
            if not self.is_camera_running:
                self.show_error("请先启动相机！")
                self.btn_enable_servo.setChecked(False)
                return
            self.set_state(self.STATE_SEARCHING)
        else:
            self.set_state(self.STATE_IDLE)

    def execute_search_step(self):
        return

    def reset_servos_to_center(self):
        if not self.is_servo_connected:
            self.show_error("请先连接舵机！")
            return
        try:
            self.visual_servo.reset_search()
            self.update_status("舵机已复位到中心位置 (1500, 1500)")
        except Exception as e:
            self.show_error(f"舵机复位失败: {e}")
            traceback.print_exc()

    def handle_target_update(self, target_pos, image_center_pos):
        if self.camera_worker:
            current_time = time.time()
            self.thread_comm_delay = (current_time - self.camera_worker.frame_timestamp) * 1000

        if target_pos is not None:
            self.target_lost_counter = 0
            if self.current_state == self.STATE_SEARCHING:
                # 开始追踪，记录时间
                self.tracking_start_time = time.time()
                self.set_state(self.STATE_TRACKING)

            if self.current_state == self.STATE_TRACKING:
                self.visual_servo.target_position = image_center_pos
                success, message, control_time, servo_time = self.visual_servo.calculate_and_send_commands(target_pos)

                if self.camera_worker:
                    self.camera_worker.control_time = control_time
                    self.camera_worker.servo_time = servo_time

                    capture_time = self.camera_worker.capture_time
                    detect_time = self.camera_worker.chessboard_detector.detect_time
                    servo_physical_time = 1000.0 if servo_time > 0.0 else 0.0

                    ui_display_start = time.time()
                    total_time = capture_time + detect_time + control_time + servo_time + self.thread_comm_delay

                    self.handle_time_stats(
                        capture_time,
                        detect_time,
                        control_time,
                        servo_time,
                        servo_physical_time,
                        self.thread_comm_delay,
                        self.ui_display_delay,
                        total_time
                    )
                    self.last_ui_update_start = ui_display_start

                if message:
                    self.update_status(message)
                    if "目标已稳定" in message or "自动停止伺服" in message:
                        # 计算追踪时间
                        tracking_time = (time.time() - self.tracking_start_time) * 1000  # 转换为毫秒
                        
                        # 保存数据
                        self.data_collection.append(tracking_time)
                        self.data_count += 1
                        self.update_status(f"已收集 {self.data_count}/10 组数据，本次追踪时间: {tracking_time:.1f}ms")
                        
                        # 检查是否收集了10组数据
                        if self.data_count >= self.max_data_count:
                            self.save_data()
                            self.update_status(f"已收集 {self.max_data_count} 组数据，数据已保存到 {self.data_file}")
                            # 重置数据收集
                            self.data_count = 0
                            self.data_collection = []
                        
                        self.set_state(self.STATE_STABILIZED)
        else:
            self.on_target_lost()

    def handle_time_stats(self, capture_time, detect_time, control_time, servo_time, servo_physical_time,
                          thread_comm_delay,
                          ui_display_delay, total_time):
        if total_time > 10:
            self.update_status(
                f"系统延迟: {total_time:.1f}ms (采集:{capture_time:.1f}ms, 检测:{detect_time:.1f}ms, Ctrl:{control_time:.1f}ms)")

    def on_target_lost(self):
        if self.current_state == self.STATE_TRACKING or self.current_state == self.STATE_STABILIZED:
            self.target_lost_counter += 1
            self.update_status(f"目标丢失... ({self.target_lost_counter}/{self.target_lost_threshold})")
            if self.target_lost_counter >= self.target_lost_threshold:
                self.update_status("【警告】目标丢失时间过长，重新开始搜索！")
                self.set_state(self.STATE_SEARCHING)
        elif self.current_state == self.STATE_SEARCHING:
            pass
    
    def save_data(self):
        """保存收集的数据到文件"""
        try:
            with open(self.data_file, 'w') as f:
                f.write("# 视觉伺服系统追踪时间数据\n")
                f.write(f"# 收集时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 组数: {len(self.data_collection)}\n")
                f.write("\n")
                f.write("# 原始数据 (单位: ms)\n")
                for i, t in enumerate(self.data_collection, 1):
                    f.write(f"{i}: {t:.1f}\n")
                f.write("\n")
                f.write("# 统计信息\n")
                if self.data_collection:
                    f.write(f"平均值: {np.mean(self.data_collection):.1f}ms\n")
                    f.write(f"最大值: {np.max(self.data_collection):.1f}ms\n")
                    f.write(f"最小值: {np.min(self.data_collection):.1f}ms\n")
                    f.write(f"标准差: {np.std(self.data_collection):.1f}ms\n")
            self.update_status(f"数据已成功保存到 {self.data_file}")
        except Exception as e:
            self.update_status(f"保存数据失败: {e}")

    def on_control_params_changed(self):
        self.visual_servo.set_control_params(
            self.spin_wait_time.value(),
            self.spin_dead_zone.value(),
            self.spin_settle_frames.value()
        )
        self.update_status("精细控制参数已更新")

    def update_frame(self, frame, timestamp):
        try:
            ui_update_start = time.time()

            h, w, ch = frame.shape
            q_image = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            max_w, max_h = 1920, 1080
            scaled_pixmap = pixmap.scaled(min(w, max_w), min(h, max_h), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.image_label.setPixmap(scaled_pixmap)

            self.ui_display_delay = (time.time() - ui_update_start) * 1000

        except Exception as e:
            print(f"更新图像失败: {e}")
            traceback.print_exc()

    def update_status(self, message):
        try:
            state_prefix = ["[空闲]", "[搜索中]", "[跟踪中]", "[已稳定]"][self.current_state]
            self.status_text.append(f"[{time.strftime('%H:%M:%S')}] {state_prefix} {message}")
            scrollbar = self.status_text.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            print(f"更新状态时出错: {e}")

    def on_exposure_changed(self, value):
        self.label_exposure_value.setText(f"{value}μs")
        if self.is_camera_running and self.camera_worker:
            if self.combo_camera_mode.currentData() and HAS_HIKVISION_SDK:
                self.camera_worker.set_exposure_time(value)

    def adjust_exposure(self, delta):
        current_value = self.slider_exposure.value()
        new_value = current_value + delta
        new_value = max(self.slider_exposure.minimum(), min(self.slider_exposure.maximum(), new_value))
        self.slider_exposure.setValue(new_value)

    def show_error(self, message):
        self.update_status(f"【错误】: {message}")

    def closeEvent(self, event):
        self.set_state(self.STATE_IDLE)
        if self.is_camera_running:
            if self.camera_worker and self.camera_worker.isRunning():
                self.camera_worker.stop()
                self.camera_worker.wait(1000)
        if self.is_servo_connected:
            self.toggle_servo()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualServoUI()
    window.show()
    sys.exit(app.exec_())