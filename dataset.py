import cv2
import os

# --- CẤU HÌNH ---
TARGET_COUNT = 145

TASKS = [
    # Bạn cứ thêm video mới vào đây thoải mái, ảnh cũ vẫn còn nguyên
    ("C://Users//TanLoc//OneDrive//Desktop//Project_TGM_HM//video//deokinh.mp4",   "C://Users//TanLoc//OneDrive//Desktop//Project_TGM_HM//dataset//open"),
    ("C://Users//TanLoc//OneDrive//Desktop//Project_TGM_HM//video//deokinh_nham.mp4", "C://Users//TanLoc//OneDrive//Desktop//Project_TGM_HM//dataset//closed"),
    # Ví dụ thêm video của Khoa:
    # ("C://Users//...//Khoa_open.mp4", "C://Users//...//dataset//open"),
]

for video_path, output_folder in TASKS:
    # 1. Tạo folder nếu chưa có
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    
    if not os.path.exists(video_path):
        print(f"[BỎ QUA] Không tìm thấy file: {video_path}")
        continue

    # --- [SỬA QUAN TRỌNG] Lấy tên file video để làm tên ảnh ---
    # Ví dụ: video_path là ".../Chopper_open.mp4" -> base_name sẽ là "Chopper_open"
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"[LỖI] Video {video_path} bị lỗi.")
        continue

    print(f"--- Đang xử lý: {base_name} ---")
    print(f"    Tổng độ dài: {total_frames} frames. Mục tiêu: Lấy {TARGET_COUNT} ảnh.")

    saved_count = 0
    
    for i in range(TARGET_COUNT):
        frame_idx = int(i * (total_frames / TARGET_COUNT))
        
        if frame_idx >= total_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if ret:
            # --- [SỬA QUAN TRỌNG] Đặt tên file theo Tên Video + Số thứ tự ---
            # Kết quả: Chopper_open_0.jpg, Chopper_open_1.jpg ...
            filename = f"{output_folder}/{base_name}_{saved_count}.jpg"
            
            cv2.imwrite(filename, frame)
            saved_count += 1
        else:
            print(f"    [Cảnh báo] Lỗi frame thứ {frame_idx}")

    cap.release()

    print(f"✅ HOÀN TẤT: Đã thêm {saved_count} ảnh từ video '{base_name}' vào dataset.\n")
