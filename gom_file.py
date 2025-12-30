import cv2
import os
import csv
import numpy as np
from face_utils import FaceMeshDetector

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Bạn nhớ sửa lại đường dẫn cho đúng máy mình nhé
OUTPUT_FILE = r"C://Users//admin//Downloads//Computer vision//KTHP//Project_TGM_HM//geometry_features.csv"
DATASET_ROOT = r"C://Users//admin//Downloads//Computer vision//KTHP//Project_TGM_HM//dataset"

# Cấu hình thư mục và nhãn
FOLDERS = {
    "no_yawn": 0,
    "open":    0,
    "closed":  1,
    "yawn":    2
}

# --- HÀM XỬ LÝ ẢNH NÂNG CAO (PRE-PROCESSING) ---
def preprocess_image(image):
    """
    Hàm này làm sạch và tối ưu ảnh trước khi đưa vào nhận diện
    """
    # 1. CLAHE: Cân bằng ánh sáng cục bộ (Giúp nhìn rõ mắt khi trời tối)
    # Chuyển sang hệ màu LAB (L: Lightness, A, B: Color)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE lên kênh sáng (L)
    # clipLimit=2.0: Ngưỡng tương phản (cao quá sẽ bị nhiễu hạt)
    # tileGridSize=(8,8): Chia ảnh thành lưới 8x8 để xử lý từng ô
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Gộp lại và chuyển về BGR
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. SHARPENING: Làm sắc nét ảnh (Giúp viền mắt/môi rõ hơn)
    # Ma trận làm nét cơ bản
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_img = cv2.filter2D(enhanced_img, -1, kernel)

    return sharpened_img

# --- CHƯƠNG TRÌNH CHÍNH ---
detector = FaceMeshDetector()

print(f"[INFO] Đang tạo file CSV tại: {OUTPUT_FILE}")
# Mở file để ghi (mode 'w' sẽ xóa dữ liệu cũ ghi mới)
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    # Ghi tiêu đề cột
    writer.writerow(["LeftEAR", "RightEAR", "MAR", "Label"])
    
    total = 0
    for folder, label in FOLDERS.items():
        path = os.path.join(DATASET_ROOT, folder)
        
        # Kiểm tra folder có tồn tại không
        if not os.path.exists(path):
            print(f"[BỎ QUA] Không tìm thấy folder: {path}")
            continue
            
        print(f"-> Đang xử lý folder: {folder} (Label: {label})")
        
        # Duyệt qua từng ảnh trong folder
        for fname in os.listdir(path):
            img_path = os.path.join(path, fname)
            
            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None: 
                continue
            
            # === [BƯỚC NÂNG CẤP] ===
            # Thay vì đưa ảnh gốc, ta đưa ảnh đã qua xử lý
            processed_img = preprocess_image(img)
            # =======================
            
            # Trích xuất đặc trưng
            result = detector.extract_features(processed_img)
            
            # --- XỬ LÝ DỮ LIỆU TRẢ VỀ (Tương thích cả bản cũ và mới) ---
            if result is None: 
                continue 
            
            # Nếu result là tuple (features, bbox, nose_point...) -> Lấy phần tử đầu tiên
            if isinstance(result, tuple):
                features = result[0]
            else:
                features = result
            
            # Kiểm tra lại lần cuối xem features có hợp lệ không
            if features is not None:
                # Features là numpy array, cần chuyển thành list để ghi vào CSV
                # Cấu trúc ghi: [LeftEAR, RightEAR, MAR, Label]
                row_data = list(features) + [label]
                writer.writerow(row_data)
                total += 1

print(f"✅ HOÀN TẤT! Tổng cộng đã trích xuất được {total} dòng dữ liệu.")
print("-> Bây giờ bạn hãy chạy file train_model.py để huấn luyện lại nhé!")