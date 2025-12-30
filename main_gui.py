import sys
import cv2
import numpy as np
import joblib
import time
import os
import pygame
from face_utils import FaceMeshDetector
from datetime import datetime
from collections import deque

# PyQt5 Imports
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QIcon
from PyQt5.QtCore import QTimer, Qt


# ================= CẤU HÌNH ĐƯỜNG DẪN & NGƯỠNG =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Thay vì dùng đường dẫn C://... dài dòng và dễ sai
MODEL_PATH = os.path.join(CURRENT_DIR, "drowsiness_ensemble.pkl")

# File tài nguyên
BG_IMAGE_PATH = os.path.join(CURRENT_DIR, "HDPE.jpg")
MUTE_ICON_PATH = os.path.join(CURRENT_DIR, "mute.jpg")
SOUND_ALARM_PATH = os.path.join(CURRENT_DIR, "chuongqd.wav")
SOUND_WARN_PATH = os.path.join(CURRENT_DIR, "bip.wav")

# Ngưỡng Logic
EYE_CLOSE_TIME_THRESH = 2.0
YAWN_TIME_THRESH = 1.0
NOD_COUNT_THRESH = 8
NOD_RESET_TIME = 4.0

# ================= CLASS LOGIC GẬT ĐẦU =================
class NodDetector:
    def __init__(self):
        self.reset()
        self.threshold = 60
        self.awake_start_time = None
        self.AWAKE_STOP_ALARM_TIME = 5.0  # giây
        self.total_drive_seconds = 0        # tổng thời gian lái trong ngày
        self.daily_drive_seconds_cache = 0  # cache khi stop/start
        self.session_drive_seconds = 0      # thời gian phiên hiện tại




    def reset(self):
        """Hàm reset trạng thái về ban đầu"""
        self.min_y = None; self.max_y = None
        self.state = 0; self.nod_count = 0
        self.last_nod_time = time.time()

    def update(self, nose_y):
        current_time = time.time()
        # Reset nếu lâu quá không gật tiếp
        if current_time - self.last_nod_time > NOD_RESET_TIME and self.nod_count > 0:
            if self.nod_count < NOD_COUNT_THRESH: 
                self.nod_count = 0; self.state = 0
        
        if self.min_y is None: self.min_y = nose_y; self.max_y = nose_y; return 0
        self.min_y = min(self.min_y, nose_y)
        self.max_y = max(self.max_y, nose_y)

        # State Machine: 0->1->2->Count
        if self.state == 0:
            if nose_y > self.min_y + self.threshold: self.state = 1
        elif self.state == 1:
            if nose_y < self.max_y - self.threshold: self.state = 2
        elif self.state == 2:
             if nose_y < self.max_y - self.threshold:
                self.nod_count += 1
                self.last_nod_time = current_time
                self.min_y = nose_y; self.max_y = nose_y; self.state = 0
        return self.nod_count

# ================= WIDGET: PROGRESS CIRCLE =================
class ProgressCircle(QWidget):
    def __init__(self, label, color, parent=None):
        super().__init__(parent)
        self.value = 0
        self.label = label
        self.color = QColor(color)
        self.display_text = ""   # 👈 THÊM


    def setDisplayText(self, text):
        self.display_text = text
        self.update()


    def setValue(self, value):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(15, 15, -15, -15)

        # Vẽ nền
        painter.setPen(QPen(QColor(60, 60, 60), 14))
        painter.drawArc(rect, 0, 360 * 16)

        # Vẽ giá trị
        painter.setPen(QPen(self.color, 14))
        span = int(-360 * self.value / 100)
        painter.drawArc(rect, 90 * 16, span * 16)

        # Vẽ chữ
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        painter.drawText(
        self.rect(),
        Qt.AlignCenter,
        f"{self.label}\n{self.display_text}"
)


        

# ================= SINGLE FATIGUE BAR =================
class FatigueBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0

    def setValue(self, value):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bar_h = 32
        margin = 20
        bar_w = self.width() - 2 * margin
        y = self.height() // 2 - bar_h // 2

        painter.setBrush(QColor(70, 70, 70))
        painter.drawRoundedRect(margin, y, bar_w, bar_h, 10, 10)

        if self.value < 40:
            color = QColor("#2ecc71")
            status = "TỈNH TÁO"
        elif self.value < 70:
            color = QColor("#f1c40f")
            status = "BẮT ĐẦU MỆT"
        else:
            color = QColor("#e74c3c")
            status = "NGUY HIỂM"

        painter.setBrush(color)
        painter.drawRoundedRect(
            margin, y, int(bar_w * self.value / 100), bar_h, 10, 10
        )

        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        painter.drawText(
            self.rect(),
            Qt.AlignCenter,
            f"MỨC ĐỘ MỆT: {int(self.value)}% | {status}"
        )


# ================= MAIN APPLICATION =================
class DrowsinessApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Drowsiness Detection System by HDPE")
        self.resize(1280, 720)
        
        # Load Model
        try:
            self.clf = joblib.load(MODEL_PATH)
            self.model_loaded = True
            print(f"Load model thành công tại: {MODEL_PATH}")
        except Exception as e:
            print(f"Lỗi thực tế khi load model là: {e}") # Nó sẽ hiện lỗi thật ở đây
            self.model_loaded = False

        self.detector = FaceMeshDetector()
        self.nod_logic = NodDetector()

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

        # --- BIẾN ĐIỂM SỐ ---
        self.score_sleep = 0.0
        self.score_yawn = 0.0
        self.score_alert = 100.0

        # ===== FATIGUE THEO NGÀY =====
        self.day_total = 0
        self.day_sleep = 0
        self.day_yawn = 0
        self._fatigue_warned = False

        
        # Biến đếm thời gian
        self.eye_start = None
        self.yawn_start = None
        
        self.setup_ui()
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.total_drive_seconds = 0
        # ===== DRIVE TIME (DAILY / SESSION) =====
        self.daily_drive_seconds_cache = 0   # lưu tổng thời gian khi STOP
        self.session_drive_seconds = 0       # thời gian phiên hiện tại



    def setup_ui(self):
        # Background
        self.bg_label = QLabel(self)
        if os.path.exists(BG_IMAGE_PATH):
            self.bg_label.setPixmap(QPixmap(BG_IMAGE_PATH))
        else:
            self.bg_label.setStyleSheet("background-color: #222;") 
        self.bg_label.setScaledContents(True)
        self.bg_label.lower()

        # Video
        self.video_label = QLabel(self)
        self.video_label.setGeometry(100, 250, 900, 660)
        self.video_label.setStyleSheet("border:3px solid white; background:black;")

        # Time
        self.time_label = QLabel(self)
        self.time_label.setGeometry(50, 80, 350, 45)
        self.time_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.time_label.setStyleSheet("color:white; background:rgba(0,0,0,160); padding-left:10px;")
        
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)

        # Drive Time
        self.drive_time_label = QLabel("Thời gian lái: 00:00:00", self)
        self.drive_time_label.setGeometry(1250, 550, 420, 40)
        self.drive_time_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.drive_time_label.setStyleSheet("color:white; background:rgba(0,0,0,150); padding:5px;")
        self.drive_time_label.setAlignment(Qt.AlignCenter)
        self.drive_start_time = None
        self.drive_timer = QTimer()
        self.drive_timer.timeout.connect(self.update_drive_time)

        # ===== FATIGUE BAR =====
        self.fatigue_bar = FatigueBar(self)
        self.fatigue_bar.setGeometry(1250, 600, 420, 80)
        self.fatigue_bar.setStyleSheet("background:rgba(0,0,0,120);")

        # Status
        self.status_label = QLabel("STATUS: WAITING", self)
        self.status_label.setGeometry(1270, 400, 380, 80)
        self.status_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color:yellow; background:rgba(0,0,0,170);")

        # Buttons
        self.start_btn = QPushButton("START", self)
        self.start_btn.setGeometry(1100, 750, 300, 55)
        self.start_btn.clicked.connect(self.start_camera)
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; border-radius: 12px; font-size: 18px;")

        self.stop_btn = QPushButton("STOP", self)
        self.stop_btn.setGeometry(1500, 750, 300, 55)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; border-radius: 12px; font-size: 18px;")

        self.mute_btn = QPushButton(self)
        self.mute_btn.setGeometry(900, 290, 64, 64)
        if os.path.exists(MUTE_ICON_PATH):
            self.mute_btn.setIcon(QIcon(MUTE_ICON_PATH))
            self.mute_btn.setIconSize(self.mute_btn.size())
        else:
            self.mute_btn.setText("MUTE")
        self.mute_btn.setStyleSheet("background:transparent; border:none; color: white; font-weight: bold;")
        self.mute_btn.clicked.connect(self.toggle_mute)

        # Circles
        self.circle_awake = ProgressCircle("AWAKE", "#00ff00", self)
        self.circle_awake.setGeometry(500, 30, 180, 180)

        self.circle_sleep = ProgressCircle("SLEEP", "#ff0000", self)
        self.circle_sleep.setGeometry(1000, 30, 180, 180)

        self.circle_yawn = ProgressCircle("YAWN", "#ffa500", self)
        self.circle_yawn.setGeometry(750, 30, 180, 180)

        # ===== DAILY DRIVING TIME =====
        self.circle_daily_fatigue = ProgressCircle("DAILY\nDRIVING\nTIME", "#9b59b6", self)
        self.circle_daily_fatigue.setGeometry(1250, 30, 180, 180)


    def resizeEvent(self, event):
        self.bg_label.setGeometry(0, 0, self.width(), self.height())

    def update_time(self):
        self.time_label.setText("🕒 " + datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))

    def update_drive_time(self):
        if not self.drive_start_time:
            return

        elapsed = datetime.now() - self.drive_start_time
        self.session_drive_seconds = int(elapsed.total_seconds())

        # ===== TOTAL = CACHE + SESSION =====
        self.total_drive_seconds = (
            self.daily_drive_seconds_cache + self.session_drive_seconds
        )

        # ===== GIỚI HẠN 10 GIỜ =====
        self.total_drive_seconds = min(self.total_drive_seconds, 10 * 3600)

        # ===== LABEL: CHỈ PHIÊN =====
        sh = self.session_drive_seconds // 3600
        sm = (self.session_drive_seconds % 3600) // 60
        ss = self.session_drive_seconds % 60

        self.drive_time_label.setText(
            f"Thời gian lái: {sh:02d}:{sm:02d}:{ss:02d}"
        )

        # ===== CIRCLE DAILY =====
        circle_value = (self.total_drive_seconds / (10 * 3600)) * 100
        self.circle_daily_fatigue.setValue(circle_value)

        dh = self.total_drive_seconds // 3600
        dm = (self.total_drive_seconds % 3600) // 60
        self.circle_daily_fatigue.setDisplayText(f"{dh:02d} : {dm:02d}")


    # ================= LOGIC CHÍNH =================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        frame_ai = frame 
        h, w = frame_ai.shape[:2]

        if self.model_loaded:
            features, bbox, nose = self.detector.extract_features(frame_ai)
        else: features = None

        status_text = "BINH THUONG"
        box_color = (0, 255, 0)
        is_warning = False
        has_error = False
        nods = 0

        if features is not None:
            (fx, fy, fw, fh) = bbox
            pred = self.clf.predict([features])[0]

            self.day_total += 1
            if pred == 1:
                self.day_sleep += 1
            elif pred == 2:
                self.day_yawn += 1

            # Logic Ghi đè AI
            mar = features[2]
            if mar > 0.4: pred = 2 
            elif pred == 2 and mar < 0.3: pred = 0 

            # Logic Gật đầu
            nods = self.nod_logic.update(nose[1])

            # 1. XỬ LÝ NGỦ (SLEEP)
            if pred == 1:
                if self.eye_start is None: self.eye_start = time.time()
                elapsed = time.time() - self.eye_start
                status_text = f"NGU GAT: {elapsed:.1f}s"
                box_color = (0, 165, 255)
                has_error = True
                
                # Tăng điểm Sleep (max 100)
                self.score_sleep = min(self.score_sleep + 0.5, 100)
                
                if elapsed > EYE_CLOSE_TIME_THRESH: is_warning = True
            else:
                self.eye_start = None
                # Giảm điểm Sleep từ từ
                self.score_sleep = max(self.score_sleep - 0.2, 0)

            # 2. XỬ LÝ NGÁP (YAWN)
            if pred == 2:
                if self.yawn_start is None: self.yawn_start = time.time()
                dur = time.time() - self.yawn_start
                if dur > YAWN_TIME_THRESH:
                    status_text = f"NGAP !!! ({dur:.1f}s)"
                    box_color = (0, 255, 255)
                    has_error = True
                    # Tăng điểm Yawn
                    self.score_yawn = min(self.score_yawn + 0.5, 100)
                else:
                    status_text = " "
                    box_color = (200, 255, 200)
            else:
                self.yawn_start = None
                # Giảm điểm Yawn từ từ
                self.score_yawn = max(self.score_yawn - 0.2, 0)

            # 3. XỬ LÝ GẬT ĐẦU
            if nods >= NOD_COUNT_THRESH:
                is_warning = True
                has_error = True
                status_text = "NGU GAT !!!"
                # Phạt nặng điểm Sleep
                self.score_sleep = min(self.score_sleep + 2.0, 100)

            # --- TÍNH ĐIỂM BÙ TRỪ (TỔNG 100%) ---
            # Công thức: Awake = 100 - (Sleep + Yawn)
            # Nếu Sleep + Yawn > 100 thì co lại cho phù hợp
            
            total_fatigue = self.score_sleep + self.score_yawn
            
            if total_fatigue > 100:
                # Nếu tổng mệt > 100, chuẩn hóa lại theo tỉ lệ
                ratio = 100 / total_fatigue
                self.score_sleep *= ratio
                self.score_yawn *= ratio
                self.score_alert = 0
            else:
                self.score_alert = 100 - total_fatigue

            # --- VẼ GIAO DIỆN ---
            # ================== ALARM & WARNING LOGIC ==================

            if is_warning:
                # ---- CÓ CẢNH BÁO (NGỦ / GẬT) ----
                self.awake_start_time = None      # reset đếm awake
                self.alarm_muted = False          # ✅ TỰ BỎ MUTE KHI NGỦ LẠI

                cv2.rectangle(frame_ai, (0, 0), (w, h), (0, 0, 255), 10)
                msg = "NGU GAT !!!"
                cv2.putText(
                    frame_ai,
                    msg,
                    (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    5
                )

                self.play_alarm()

            else:
                # ---- KHÔNG CẢNH BÁO (AWAKE) ----

                # Nếu đã bấm mute → tắt chuông ngay
                if self.alarm_muted:
                    self.stop_alarm()
                    self.awake_start_time = None

                else:
                    # Chưa mute → bắt đầu / tiếp tục đếm awake
                    if self.awake_start_time is None:
                        self.awake_start_time = time.time()
                    else:
                        awake_duration = time.time() - self.awake_start_time

                        # Awake đủ 5 giây → tắt chuông
                        if awake_duration >= self.AWAKE_STOP_ALARM_SEC:
                            self.stop_alarm()

            # ================== DRAW NORMAL UI ==================
            cv2.rectangle(frame_ai, (fx, fy), (fw, fh), box_color, 2)
            cv2.putText(
                frame_ai,
                status_text,
                (fx, fy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

            cv2.putText(
                frame_ai,
                f"NODS: {nods}/{NOD_COUNT_THRESH}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            # ================== STATUS LABEL ==================
            self.status_label.setText(status_text if is_warning else "MONITORING...")
            self.status_label.setStyleSheet(
                f"color:{'red' if is_warning else 'lime'}; background:rgba(0,0,0,180);"
            )

        # Update Circles
        self.circle_awake.setValue(self.score_alert)
        self.circle_sleep.setValue(self.score_sleep)
        self.circle_yawn.setValue(self.score_yawn)

        fatigue = self.calculate_fatigue(self.total_drive_seconds)

        self.fatigue_bar.setValue(fatigue)

        if fatigue > 70:
            if not self._fatigue_warned:
                self._fatigue_warned = True

                if not self.alarm_muted:
                    self.fatigue_sound.play(loops=-1)
                    self.fatigue_playing = True

                QMessageBox.warning(
                    self,
                    "CẢNH BÁO MỆT MỎI",
                    f"Mức độ mệt mỏi đã vượt {fatigue:.0f}%!\nHãy nghỉ ngơi ngay!"
                )
        else:
            self._fatigue_warned = False
            if self.fatigue_playing:
                self.fatigue_sound.stop()
                self.fatigue_playing = False

        # Display Image
        rgb = cv2.cvtColor(frame_ai, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        ))
    
    def calculate_fatigue(self, total_seconds):
        """
        Tính mức độ mệt mỏi dựa trên:
        - Số lượng frame sleep và yawn
        - Thời gian lái (giây)
        Trả về giá trị 0-100 (%)
        """
        if self.day_total == 0:
            return 0

        # Fatigue cơ bản (hành vi)
        base_fatigue = (self.day_sleep + 0.5 * self.day_yawn) / self.day_total * 110

        # Fatigue theo thời gian lái: 4 tiếng = 70%
        time_factor = total_seconds / 14400 * 70

        fatigue = base_fatigue + time_factor
        return min(100, fatigue)

    # ================= CONTROLS =================
    def reset_system_state(self):
        """Hàm reset toàn bộ trạng thái về mặc định"""
        # 1. Reset logic gật đầu
        self.nod_logic.reset()
        
        # 2. Reset các biến đếm thời gian
        self.eye_start = None
        self.yawn_start = None
        
        # 3. Reset điểm số
        self.score_alert = 100.0
        self.score_sleep = 0.0
        self.score_yawn = 0.0
        
        # 4. Cập nhật giao diện ngay lập tức
        self.circle_awake.setValue(100)
        self.circle_sleep.setValue(0)
        self.circle_yawn.setValue(0)
        self.status_label.setText("STATUS: READY")
        self.stop_alarm()

    def start_camera(self):
        self.reset_system_state()
        self.awake_start_time = None

        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

        self.drive_start_time = datetime.now()
        self.drive_timer.start(1000)

        self.status_label.setText("STATUS: STARTED")

        # Reset hành vi trong phiên
        self.day_total = 0
        self.day_sleep = 0
        self.day_yawn = 0
        self._fatigue_warned = False
        self.fatigue_bar.setValue(0)

        # ✅ CHỈ reset PHIÊN
        self.session_drive_seconds = 0
        self.drive_start_time = datetime.now()   # OK


    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()

        self.stop_alarm()

        self.drive_timer.stop()
        self.drive_start_time = None

        # ✅ CHỐT DAILY TẠI ĐÂY (DÒNG QUYẾT ĐỊNH)
        self.daily_drive_seconds_cache = self.total_drive_seconds
        self.session_drive_seconds = 0

        self.drive_time_label.setText("Thời gian lái: 00:00:00")

        self.eye_start = None
        self.yawn_start = None
        self.awake_start_time = None

        self.status_label.setText("STATUS: STOPPED")

    def play_alarm(self):
        if not self.alarm_playing and not self.alarm_muted:
            try: self.alarm_sound.play(loops=-1)
            except: pass
            self.alarm_playing = True

    def stop_alarm(self):
        try: self.alarm_sound.stop()
        except: pass
        self.alarm_playing = False

    def toggle_mute(self):
        self.alarm_muted = True
        self.stop_alarm()
        self.awake_start_time = None


    def closeEvent(self, event):
        self.stop_camera()
        pygame.mixer.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessApp()
    window.show()
    sys.exit(app.exec_())