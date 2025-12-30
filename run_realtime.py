import cv2
import numpy as np
import joblib
import time
import os
from face_utils import FaceMeshDetector

# --- CẤU HÌNH ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "drowsiness_ensemble.pkl")

# --- CÁC NGƯỠNG THỜI GIAN (Thresholds) ---
EYE_CLOSE_TIME_THRESH = 2.0  # Nhắm mắt > 2s -> Cảnh báo
YAWN_TIME_THRESH = 3.0       # [MỚI] Ngáp > 3s -> Mới hiện cảnh báo
NOD_COUNT_THRESH = 4         # Gật đầu > 4 cái -> Cảnh báo
NOD_RESET_TIME = 4.0         

# --- CLASS PHÁT HIỆN GẬT ĐẦU ---
class NodDetector:
    def __init__(self):
        self.min_y = None; self.max_y = None
        self.state = 0; self.nod_count = 0
        self.last_nod_time = time.time()
        self.threshold = 20 # Độ nhạy

    def update(self, nose_y):
        current_time = time.time()
        if current_time - self.last_nod_time > NOD_RESET_TIME and self.nod_count > 0:
            if self.nod_count < NOD_COUNT_THRESH: 
                self.nod_count = 0; self.state = 0
        
        if self.min_y is None: self.min_y = nose_y; self.max_y = nose_y; return 0

        self.min_y = min(self.min_y, nose_y)
        self.max_y = max(self.max_y, nose_y)

        if self.state == 0:
            if nose_y > self.min_y + self.threshold: self.state = 1
        elif self.state == 1:
            if nose_y < self.max_y - self.threshold: self.state = 2
        elif self.state == 2:
             if nose_y < self.max_y - self.threshold:
                self.nod_count += 1
                self.last_nod_time = current_time
                self.min_y = nose_y; self.max_y = nose_y; self.state = 0
                print(f"-> GẬT LẦN THỨ: {self.nod_count}")
        return self.nod_count

# --- HÀM VẼ CẢNH BÁO ---
def draw_critical_warning(frame, message="NGU GAT !!!"):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 0, 255), 10)
    
    font_scale = 1.5; thickness = 3
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2; text_y = (h + text_size[1]) // 2
    cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

# --- CHƯƠNG TRÌNH CHÍNH ---
def main():
    print("[INIT] Đang tải model...")
    try:
        clf = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("LỖI: Chưa có file model (.pkl). Hãy chạy train_model.py trước.")
        return

    detector = FaceMeshDetector()
    nod_logic = NodDetector()
    cap = cv2.VideoCapture(0)

    # --- BIẾN QUẢN LÝ THỜI GIAN ---
    eye_closed_start = None
    yawn_start_time = None  # [MỚI] Biến đếm thời gian ngáp
    
    # --- BIẾN QUẢN LÝ ĐIỂM SỐ (PHẦN TRĂM) ---
    score_alert = 100.0 # Tỉnh táo (Max 100)
    score_yawn = 0.0    # Điểm ngáp (Max 100)
    score_sleep = 0.0   # Điểm buồn ngủ (Max 100)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        features, bbox, nose_point = detector.extract_features(frame)
        
        status_text = "BINH THUONG"
        color = (0, 255, 0) # Xanh lá

        # Mặc định là không cảnh báo
        IS_WARNING = False
        current_nods = 0
        current_eye_ratio = 0.0
        
        # Biến cờ (Flag) xem frame này có bị lỗi gì không để tính điểm
        frame_has_error = False 

        if features is not None:
            (fx_min, fy_min, fx_max, fy_max) = bbox
            nose_y = nose_point[1]
            
            # 1. AI DỰ ĐOÁN
            prediction = clf.predict([features])[0] # 0: Normal, 1: Sleep, 2: Yawn
            
            # 2. XỬ LÝ MAR (OVERRIDES + CHẶN NGÁP ẢO)
            current_mar = features[2]
            FORCE_YAWN_THRESH = 0.7  # Lớn hơn mức này -> Ép thành NGÁP
            BLOCK_YAWN_THRESH = 0.5  # Nhỏ hơn mức này -> Chặn ngáp ảo

            if current_mar > FORCE_YAWN_THRESH:
                prediction = 2
            elif prediction == 2 and current_mar < BLOCK_YAWN_THRESH:
                prediction = 0 

            # 3. LOGIC GẬT ĐẦU
            current_nods = nod_logic.update(nose_y)

            # -------------------------------------------------------------
            # [LOGIC MỚI] XỬ LÝ TRẠNG THÁI & TÍNH THỜI GIAN
            # -------------------------------------------------------------
            
            # A. Xử lý NGỦ (Nhắm mắt)
            if prediction == 1: 
                if eye_closed_start is None: eye_closed_start = time.time()
                elapsed = time.time() - eye_closed_start
                current_eye_ratio = min(elapsed / EYE_CLOSE_TIME_THRESH, 1.0)
                
                status_text = f"NHAM MAT: {elapsed:.1f}s"
                color = (0, 165, 255) # Cam
                frame_has_error = True
                
                # Cập nhật điểm SLEEP tăng nhanh
                score_sleep = min(score_sleep + 0.5, 100)
            else:
                eye_closed_start = None
                current_eye_ratio = 0.0
                score_sleep = max(score_sleep - 0.5, 0) # Giảm dần nếu mở mắt

            # B. Xử lý NGÁP (Có độ trễ 3s)
            if prediction == 2:
                if yawn_start_time is None: yawn_start_time = time.time()
                yawn_duration = time.time() - yawn_start_time
                
                # Logic quan trọng: Chỉ hiện chữ NGÁP nếu đã ngáp > 3 giây
                if yawn_duration > YAWN_TIME_THRESH:
                    status_text = f"NGAP !!! ({yawn_duration:.1f}s)"
                    color = (0, 255, 255) # Vàng
                    frame_has_error = True
                    
                    # Cập nhật điểm YAWN tăng
                    score_yawn = min(score_yawn + 0.5, 100)
                else:
                    # Đang ngáp nhưng chưa đủ 3s -> Vẫn coi là bình thường hoặc cảnh báo nhẹ
                    status_text = f"Dang ha mieng... {yawn_duration:.1f}s"
                    color = (200, 255, 200) # Xanh nhạt
            else:
                yawn_start_time = None
                score_yawn = max(score_yawn - 0.2, 0) # Giảm từ từ

            # C. Xử lý GẬT ĐẦU
            if current_nods >= NOD_COUNT_THRESH:
                frame_has_error = True
                score_sleep = min(score_sleep + 2.0, 100) # Gật đầu cộng điểm ngủ cực mạnh

            # -------------------------------------------------------------
            # [LOGIC MỚI] TÍNH TOÁN PHẦN TRĂM TỈNH TÁO (SCORE SYSTEM)
            # -------------------------------------------------------------
            if not frame_has_error:
                # Nếu không lỗi gì -> Hồi phục sự tỉnh táo
                score_alert = min(score_alert + 0.5, 100)
            else:
                # Nếu có lỗi -> Giảm tỉnh táo
                score_alert = max(score_alert - 1.0, 0)
            
            # Ràng buộc: Alert = 100 - (Sleep + Yawn) để logic hợp lý hơn (tương đối)
            # Hoặc giữ độc lập để dễ theo dõi. Ở đây tôi để độc lập nhưng ảnh hưởng nhau.

            # KIỂM TRA ĐIỀU KIỆN CẢNH BÁO TOÀN MÀN HÌNH
            if current_eye_ratio >= 1.0 or current_nods >= NOD_COUNT_THRESH:
                IS_WARNING = True

            # -------------------------------------------------------------
            # VẼ GIAO DIỆN
            # -------------------------------------------------------------
            if IS_WARNING:
                msg = "NGU GAT (GAT DAU)" if current_nods >= NOD_COUNT_THRESH else "NGU GAT (MAT)"
                draw_critical_warning(frame, msg)
            else:
                # Vẽ khung mặt
                cv2.rectangle(frame, (fx_min, fy_min), (fx_max, fy_max), color, 2)
                cv2.putText(frame, status_text, (fx_min, fy_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # --- VẼ BẢNG THÔNG SỐ PHẦN TRĂM (GÓC TRÁI) ---
                # Nền bảng đen mờ
                cv2.rectangle(frame, (10, 10), (250, 130), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (250, 130), (255, 255, 255), 1)
                
                # 1. Tỉnh táo (Xanh lá)
                cv2.putText(frame, f"TINH TAO: {int(score_alert)}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Thanh bar cho tỉnh táo
                cv2.rectangle(frame, (170, 25), (170 + int(score_alert*0.8), 40), (0, 255, 0), -1)

                # 2. Buồn ngủ (Đỏ)
                cv2.putText(frame, f"BUON NGU: {int(score_sleep)}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (170, 65), (170 + int(score_sleep*0.8), 80), (0, 0, 255), -1)

                # 3. Ngáp (Vàng)
                cv2.putText(frame, f"NGAP    : {int(score_yawn)}%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (170, 105), (170 + int(score_yawn*0.8), 120), (0, 255, 255), -1)

                # --- THANH LOADING NHẮM MẮT (GIỮA MÀN HÌNH) ---
                if current_eye_ratio > 0:
                    bar_w = 200; bar_h = 20
                    bar_x = (w - bar_w) // 2; bar_y = 30
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                    fill_w = int(bar_w * current_eye_ratio)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 0, 255), -1)
                    cv2.putText(frame, "TIME SLEEP", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Driver Drowsiness System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()