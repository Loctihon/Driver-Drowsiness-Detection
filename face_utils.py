import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self):
        # Khởi tạo MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,                
            refine_landmarks=True,          
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_distance(self, p1, p2, img_w, img_h):
        """Tính khoảng cách giữa 2 điểm"""
        x1, y1 = int(p1.x * img_w), int(p1.y * img_h)
        x2, y2 = int(p2.x * img_w), int(p2.y * img_h)
        return np.linalg.norm([x1 - x2, y1 - y2])

    def get_ear(self, landmarks, indices, w, h):
        """Tính chỉ số EAR (Mắt)"""
        p1, p4 = landmarks[indices[0]], landmarks[indices[3]]
        p2, p6 = landmarks[indices[1]], landmarks[indices[5]]
        p3, p5 = landmarks[indices[2]], landmarks[indices[4]]

        d_h = self.calculate_distance(p1, p4, w, h)
        d_v1 = self.calculate_distance(p2, p6, w, h)
        d_v2 = self.calculate_distance(p3, p5, w, h)
        return (d_v1 + d_v2) / (2.0 * d_h) if d_h != 0 else 0

    def get_mar(self, landmarks, w, h):
        """Tính chỉ số MAR (Miệng)"""
        d_v = self.calculate_distance(landmarks[13], landmarks[14], w, h)
        d_h = self.calculate_distance(landmarks[61], landmarks[291], w, h)
        return d_v / d_h if d_h != 0 else 0

    def extract_features(self, image):
        """
        Trả về 3 giá trị: (features, bbox, nose_point)
        """
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        # Nếu không thấy mặt, trả về None cho cả 3
        if not results.multi_face_landmarks:
            return None, None, None
        
        lm = results.multi_face_landmarks[0].landmark

        # 1. Tính toán Features (EAR, MAR)
        left_ear = self.get_ear(lm, [362, 385, 387, 263, 373, 380], w, h)
        right_ear = self.get_ear(lm, [33, 160, 158, 133, 153, 144], w, h)
        mar = self.get_mar(lm, w, h)
        features = np.array([left_ear, right_ear, mar])

        # 2. Tính Bounding Box (Khung mặt)
        x_list = [pt.x for pt in lm]
        y_list = [pt.y for pt in lm]
        x_min, x_max = int(min(x_list) * w), int(max(x_list) * w)
        y_min, y_max = int(min(y_list) * h), int(max(y_list) * h)
        
        # 3. Lấy tọa độ đầu mũi (Landmark số 1) - QUAN TRỌNG ĐỂ PHÁT HIỆN GẬT ĐẦU
        nose_x, nose_y = int(lm[1].x * w), int(lm[1].y * h)

        return features, (x_min, y_min, x_max, y_max), (nose_x, nose_y)
