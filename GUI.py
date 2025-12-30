import sys
import cv2
import numpy as np
import joblib
import pygame
from PyQt5.QtWidgets import QMessageBox
from collections import deque
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QIcon
from PyQt5.QtCore import QTimer, Qt

from face_utils import FaceMeshDetector
from datetime import datetime

# ================= CONFIG =================
ROI_RATIO = (0.2, 0.1, 0.6, 0.8)
MODEL_PATH = r"C:\Users\LENOVO\Downloads\Project_TGM_HM\drowsiness_ensemble.pkl"

SLEEP_THRESH = 15
YAWN_THRESH = 15
WINDOW_SIZE = 50

# ================= IMAGE ENHANCE =================
def enhance_image_cv(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    

# ================= PROGRESS CIRCLE =================
class ProgressCircle(QWidget):
    def __init__(self, label, color, parent=None):
        super().__init__(parent)
        self.value = 0
        self.label = label
        self.color = QColor(color)

        self.drive_start_time = None
        self.drive_timer = QTimer()
        self.drive_timer.timeout.connect(self.update_drive_time)

        # ===== LABEL THỜI GIAN LÁI =====
        self.drive_time_label = QLabel("Thời gian lái: 00:00:00", self)
        self.drive_time_label.setGeometry(1100, 450, 420, 40)
        self.drive_time_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.drive_time_label.setStyleSheet("color:white; background:rgba(0,0,0,150); padding:5px;")
        self.drive_time_label.setAlignment(Qt.AlignCenter)

        self.drive_start_time = None
        self.drive_timer = QTimer()
        self.drive_timer.timeout.connect(self.update_drive_time)

    def update_drive_time(self):
        if self.drive_start_time is None:
            return
        elapsed = datetime.now() - self.drive_start_time
        total_sec = int(elapsed.total_seconds())
        h = total_sec // 3600
        m = (total_sec % 3600) // 60
        s = total_sec % 60
        self.drive_time_label.setText(f"Thời gian lái: {h:02d}:{m:02d}:{s:02d}")

        # Cập nhật label hiển thị thời gian lái
        if hasattr(self, "drive_time_label"):
            self.drive_time_label.setText(f"Thời gian lái: {h:02d}:{m:02d}:{s:02d}")

        # Tính fatigue dựa trên thời gian lái + sleep + yawn
        fatigue = self.calculate_fatigue(total_sec)
        if hasattr(self, "fatigue_bar"):
            self.fatigue_bar.setValue(fatigue)

    def calculate_fatigue(self, total_seconds):
        """
        Tính mức độ mệt mỏi dựa trên:
        - Số lượng frame sleep và yawn
        - Thời gian lái (giây)
        Trả về giá trị 0-100 (%)
        """
        if self.day_total == 0:
            return 0

        # Fatigue cơ bản dựa trên số lượng sleep & yawn
        base_fatigue = (self.day_sleep + 0.5 * self.day_yawn) / self.day_total * 100

        # Hệ số thời gian lái: mỗi 10 phút lái tăng 10% fatigue
        time_factor = total_seconds / 600 * 10  # 600s = 10 phút

        # Tổng fatigue = cơ bản + hệ số thời gian
        fatigue = base_fatigue + time_factor

        return min(100, fatigue)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

        # khởi động thời gian lái
        self.drive_start_time = datetime.now()
        self.drive_timer.start(1000)  # cập nhật mỗi giây

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.stop_alarm()

        # reset thời gian lái
        self.drive_start_time = None
        self.drive_timer.stop()
        self.drive_time_label.setText("Thời gian lái: 00:00:00")

    def setValue(self, value):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(15, 15, -15, -15)

        painter.setPen(QPen(QColor(60, 60, 60), 14))
        painter.drawArc(rect, 0, 360 * 16)

        painter.setPen(QPen(self.color, 14))
        span = int(360 * self.value / 100)
        painter.drawArc(rect, 90 * 16, -span * 16)

        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        painter.drawText(
            self.rect(),
            Qt.AlignCenter,
            f"{self.label}\n{int(self.value)}%"
        )
# ================= SINGLE FATIGUE BAR =================
class FatigueBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0  # 0–100

    def setValue(self, value):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bar_height = 32
        margin = 20
        bar_width = self.width() - 2 * margin
        y = self.height() // 2 - bar_height // 2

        # ===== BACKGROUND =====
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(70, 70, 70))
        painter.drawRoundedRect(margin, y, bar_width, bar_height, 10, 10)

        # ===== COLOR THEO MỨC ĐỘ MỆT =====
        if self.value < 40:
            color = QColor("#2ecc71")   # xanh
            status = "TỈNH TÁO"
        elif self.value < 70:
            color = QColor("#f1c40f")   # vàng
            status = "BẮT ĐẦU MỆT"
        else:
            color = QColor("#e74c3c")   # đỏ
            status = "NGUY HIỂM"

        fill_width = int(bar_width * self.value / 100)
        painter.setBrush(color)
        painter.drawRoundedRect(margin, y, fill_width, bar_height, 10, 10)

        # ===== TEXT =====
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        painter.drawText(
            self.rect(),
            Qt.AlignCenter,
            f"MỨC ĐỘ MỆT MỎI: {int(self.value)}%  |  {status}"
        )

# ================= APP =================
class DrowsinessApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Drowsiness Detection System by HDPE")
        self.resize(1280, 720)

        # trong __init__ của DrowsinessApp
        self.drive_time_label = QLabel("Thời gian lái: 00:00:00", self)
        self.drive_time_label.setGeometry(1250, 550, 420, 40)
        self.drive_time_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.drive_time_label.setStyleSheet("color:white; background:rgba(0,0,0,150); padding:5px;")
        self.drive_time_label.setAlignment(Qt.AlignCenter)

        # Timer cập nhật thời gian lái
        self.drive_start_time = None
        self.drive_timer = QTimer()
        self.drive_timer.timeout.connect(self.update_drive_time)


        # ===== FATIGUE BAR (1 THANH) =====
        self.fatigue_bar = FatigueBar(self)
        self.fatigue_bar.setGeometry(1250, 600, 420, 80)
        self.fatigue_bar.setStyleSheet("background:rgba(0,0,0,120);")


        # ===== AUDIO =====
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("chuongqd.wav")
        self.alarm_sound.set_volume(1.0)

        # Âm thanh báo fatigue > 50%
        self.fatigue_sound = pygame.mixer.Sound("bip.wav")
        self.fatigue_sound.set_volume(1.0)


        self.alarm_playing = False
        self.fatigue_playing = False
        self.alarm_muted = False
        self.awake_start_time = None
        self.AWAKE_STOP_ALARM_SEC = 5

        # ===== BACKGROUND =====
        self.bg_label = QLabel(self)
        self.bg_label.setPixmap(QPixmap("HDPE.png"))
        self.bg_label.setScaledContents(True)
        self.bg_label.lower()

        # ===== VIDEO =====
        self.video_label = QLabel(self)
        self.video_label.setGeometry(100, 250, 900, 660)
        self.video_label.setStyleSheet("border:3px solid white; background:black;")

        # ===== TIME =====
        self.time_label = QLabel(self)
        self.time_label.setGeometry(50, 80, 350, 45)
        self.time_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.time_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.time_label.setStyleSheet(
            "color:white; background:rgba(0,0,0,160); padding-left:10px;"
        )

        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)

        # ===== STATUS =====
        self.status_label = QLabel("STATUS: WAITING", self)
        self.status_label.setGeometry(1270, 400, 380, 80)
        self.status_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "color:yellow; background:rgba(0,0,0,170);"
        )

        # ===== BUTTONS =====
        self.start_btn = QPushButton("START", self)
        self.start_btn.setGeometry(1100, 750, 300, 55)
        self.start_btn.clicked.connect(self.start_camera)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;   /* xanh lá */
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)

        self.stop_btn = QPushButton("STOP", self)
        self.stop_btn.setGeometry(1500, 750, 300, 55)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;   /* đỏ */
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #922b21;
            }
        """)

        # ===== MUTE =====
        self.mute_btn = QPushButton(self)
        self.mute_btn.setGeometry(900, 290, 64, 64)
        self.mute_btn.setIcon(QIcon("mute.png"))
        self.mute_btn.setIconSize(self.mute_btn.size())
        self.mute_btn.setStyleSheet("background:transparent; border:none;")
        self.mute_btn.clicked.connect(self.toggle_mute)

        # ===== PROGRESS CIRCLES =====
        self.circle_awake = ProgressCircle("AWAKE", "#00ff00", self)
        self.circle_awake.setGeometry(450, 30, 180, 180)

        self.circle_yawn = ProgressCircle("YAWN", "#ffa500", self)
        self.circle_yawn.setGeometry(750, 30, 180, 180)

        self.circle_sleep = ProgressCircle("SLEEP", "#ff0000", self)
        self.circle_sleep.setGeometry(1050, 30, 180, 180)

        # 🔥 FATIGUE AVG DAY (THÊM MỚI)
        self.circle_fatigue = ProgressCircle("FATIGUE\nAVG DAY", "#ff00ff", self)
        self.circle_fatigue.setGeometry(1350, 30, 180, 180)

        # ===== AI =====
        self.model = joblib.load(MODEL_PATH)
        self.detector = FaceMeshDetector()

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.state_window = deque(maxlen=WINDOW_SIZE)
        # ===== DAILY STATISTICS =====
        self.day_total = 0
        self.day_awake = 0
        self.day_yawn = 0
        self.day_sleep = 0

    def update_drive_time(self):
        if self.drive_start_time is None:
            return
        elapsed = datetime.now() - self.drive_start_time
        total_sec = int(elapsed.total_seconds())
        h = total_sec // 3600
        m = (total_sec % 3600) // 60
        s = total_sec % 60
        self.drive_time_label.setText(f"Thời gian lái: {h:02d}:{m:02d}:{s:02d}")

    

    def resizeEvent(self, event):
        self.bg_label.setGeometry(0, 0, self.width(), self.height())

    def update_time(self):
        self.time_label.setText(
            "🕒 " + datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        )

    # ===== CAMERA =====
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.drive_start_time = datetime.now()
        self.drive_timer.start(1000)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.stop_alarm()
        self.drive_timer.stop()
        self.drive_start_time = None
        self.drive_time_label.setText("Thời gian lái: 00:00:00")

    # ================= FRAME =================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame = enhance_image_cv(frame)

        if self.day_total > 0:
            fatigue_day = (
                self.day_sleep * 1.0 +
                self.day_yawn * 0.5
            ) / self.day_total * 100

            self.circle_fatigue.setValue(fatigue_day)
            self.fatigue_bar.setValue(fatigue_day)

            # ======= KIỂM TRA MỨC ĐỘ MỆT VỚI ÂM THANH =======
            if fatigue_day > 50:
                if not hasattr(self, "_fatigue_warned") or not self._fatigue_warned:
                    self._fatigue_warned = True

                    # ======= PHÁT ÂM THANH FATIGUE =======
                    if not self.fatigue_playing and not self.alarm_muted:
                        self.fatigue_sound.play(loops=-1)
                        self.fatigue_playing = True

                    # ======= HIỆN POPUP =======
                    QMessageBox.warning(
                        self,
                        "Cảnh báo mệt mỏi",
                        f"Mức độ mệt mỏi đã vượt 50% ({fatigue_day:.0f}%)! Hãy nghỉ ngơi ngay!"
                    )
            else:
                # fatigue <= 50%, reset cảnh báo và tắt âm thanh
                self._fatigue_warned = False
                if self.fatigue_playing:
                    self.fatigue_sound.stop()
                    self.fatigue_playing = False



        # ========= DRIVER ZONE (ROI) =========
        h, w, _ = frame.shape
        rx, ry, rw, rh = ROI_RATIO

        roi_x = int(w * rx)
        roi_y = int(h * ry)
        roi_w = int(w * rw)
        roi_h = int(h * rh)

        roi_color = (200, 200, 200)  # xám mặc định
        # ====================================

        # ===== AI DETECTION =====
        features, bbox = self.detector.extract_features(frame)

        now = datetime.now()
        status = "NO DRIVER"
        color = "yellow"

        if bbox is not None:
            fx_min, fy_min, fx_max, fy_max = bbox

            # Tâm khuôn mặt
            face_center_x = (fx_min + fx_max) // 2
            face_center_y = (fy_min + fy_max) // 2

            # Kiểm tra trong vùng lái
            is_in_zone = (
                roi_x < face_center_x < roi_x + roi_w and
                roi_y < face_center_y < roi_y + roi_h
            )

            if is_in_zone:
                roi_color = (0, 255, 0)
                cv2.rectangle(frame, (fx_min, fy_min), (fx_max, fy_max), (0, 255, 0), 2)

                pred = self.model.predict([features])[0]
                self.state_window.append(pred)

                # ===== DAILY COUNT (CHỈ 1 LẦN) =====
                self.day_total += 1
                if pred == 0:
                    self.day_awake += 1
                elif pred == 1:
                    self.day_sleep += 1
                elif pred == 2:
                    self.day_yawn += 1

                sleep = self.state_window.count(1)
                yawn = self.state_window.count(2)

                if sleep >= SLEEP_THRESH:
                    status = "CANH BAO: NGU GAT !!!"
                    color = "red"
                    roi_color = (0, 0, 255)
                    self.play_alarm()
                    self.awake_start_time = None

                elif yawn >= YAWN_THRESH:
                    status = "CANH BAO: NGAP"
                    color = "orange"
                    roi_color = (0, 165, 255)
                    self.awake_start_time = None

                else:
                    status = "BINH THUONG"
                    color = "lime"
                    self.alarm_muted = False

                    if self.alarm_playing:
                        if self.awake_start_time is None:
                            self.awake_start_time = now
                        elif (now - self.awake_start_time).total_seconds() >= self.AWAKE_STOP_ALARM_SEC:
                            self.stop_alarm()
            else:
                status = "FACE OUT OF ZONE"
                color = "orange"
                cv2.rectangle(frame, (fx_min, fy_min), (fx_max, fy_max), (0, 165, 255), 2)

        # ===== VẼ DRIVER ZONE =====
        cv2.rectangle(
            frame,
            (roi_x, roi_y),
            (roi_x + roi_w, roi_y + roi_h),
            roi_color,
            2
        )
        cv2.putText(
            frame,
            "DRIVER ZONE",
            (roi_x, roi_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            roi_color,
            2
        )

        # ===== STATUS =====
        self.status_label.setText(status)
        self.status_label.setStyleSheet(
            f"color:{color}; background:rgba(0,0,0,180);"
        )

        # ===== UPDATE CIRCLES =====
        total = len(self.state_window)
        if total:
            self.circle_awake.setValue(self.state_window.count(0) / total * 100)
            self.circle_yawn.setValue(self.state_window.count(2) / total * 100)
            self.circle_sleep.setValue(self.state_window.count(1) / total * 100)

        # ===== FATIGUE AVG DAY =====
        if self.day_total > 0:
            fatigue_day = (
                self.day_sleep * 1.0 +
                self.day_yawn * 0.5
            ) / self.day_total * 100
            self.circle_fatigue.setValue(fatigue_day)

        # ===== SHOW FRAME (CHỈ 1 LẦN) =====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0],
            3 * rgb.shape[1], QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

         
        # ===== UPDATE CIRCLES =====
        total = len(self.state_window)
        if total:
            awake_cnt = self.state_window.count(0)
            yawn_cnt  = self.state_window.count(2)
            sleep_cnt = self.state_window.count(1)

            self.circle_awake.setValue(awake_cnt / total * 100)
            self.circle_yawn.setValue(yawn_cnt / total * 100)
            self.circle_sleep.setValue(sleep_cnt / total * 100)

        # 🔥 FATIGUE AVG DAY
        if self.day_total > 0:
            fatigue_day = (
                self.day_sleep * 1.0 +
                self.day_yawn * 0.5
            ) / self.day_total * 100

            self.circle_fatigue.setValue(fatigue_day)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0],
            3 * rgb.shape[1], QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(pixmap)

    # ================= SOUND =================
    def play_alarm(self):
        if not self.alarm_playing and not self.alarm_muted:
            self.alarm_sound.play(loops=-1)
            self.alarm_playing = True

    def stop_alarm(self):
        self.alarm_sound.stop()
        self.alarm_playing = False

    def toggle_mute(self):
        self.alarm_muted = True
        self.stop_alarm()


# ================= MAIN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessApp()
    window.show()
    sys.exit(app.exec_())
