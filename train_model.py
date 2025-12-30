import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Dùng đường dẫn tương đối để tránh lỗi máy khác nhau
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(current_dir, "geometry_features.csv")
MODEL_PATH = os.path.join(current_dir, "drowsiness_ensemble.pkl")

print("[1] 📥 Đang tải dữ liệu...")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"-> Đã tải {len(df)} dòng dữ liệu.")
except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file {CSV_FILE}")
    print("-> Hãy chạy gom_file.py để tạo dữ liệu trước!")
    exit()

# Lấy dữ liệu đầu vào (Features) và nhãn (Label)
X = df[["LeftEAR", "RightEAR", "MAR"]]
y = df["Label"]

# Chia tập train/test (80% học, 20% thi)ẽ
# stratify=y: Đảm bảo tỷ lệ các nhãn (Ngáp, Ngủ, Bình thường) ở tập train và test giống nhau
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[2] ⚙️ Đang thiết lập kiến trúc 'Siêu Model' (Ensemble)...")

# --- KỸ THUẬT 1: PIPELINE & SCALING (MỚI) ---
# SVM rất nhạy cảm với dữ liệu chưa chuẩn hóa.
# Ta tạo một 'đường ống' (Pipeline): Dữ liệu đi qua Scaler (làm sạch) -> rồi mới vào SVM.
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()), # Chuẩn hóa dữ liệu về dạng chuẩn (Mean=0, Std=1)
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced'))
])

# --- KỸ THUẬT 2: RANDOM FOREST (GIỮ NGUYÊN) ---
# Random Forest không cần Scale, nó giỏi xử lý nhiễu.
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# --- KỸ THUẬT 3: GRADIENT BOOSTING (MỚI - CỰC MẠNH) ---
# Model này học theo kiểu "Sửa sai". Nó nhìn xem các model trước sai ở đâu để tập trung học chỗ đó.
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# --- TỔNG HỢP: VOTING CLASSIFIER (HỘI ĐỒNG GIÁM KHẢO) ---
# Kết hợp cả 3 ông lớn: SVM (Toán học) + Random Forest (Thống kê) + Gradient Boosting (Học sâu chuỗi)
voting_clf = VotingClassifier(
    estimators=[
        ('svm_pipe', svm_pipeline), 
        ('rf', rf_clf),
        ('gb', gb_clf)
    ],
    voting='soft', # 'soft': Tính trung bình độ tin cậy (xác suất) thay vì chỉ đếm phiếu bầu
    weights=[2, 1, 1] # (Tuỳ chọn) Cho SVM quyền lực gấp đôi nếu nó chính xác nhất
)

print("[3] 🧠 Đang huấn luyện (Training)...")
voting_clf.fit(X_train, y_train)

# Đánh giá kết quả
print("\n--- 📊 KẾT QUẢ ĐÁNH GIÁ MODEL ---")
predictions = voting_clf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Độ chính xác tổng thể: {acc*100:.2f}%")
print(classification_report(y_test, predictions, target_names=["Normal", "Sleep", "Yawn"]))

# Lưu model
joblib.dump(voting_clf, MODEL_PATH)
print(f"✅ Đã lưu model thành công tại: {MODEL_PATH}")
print("-> Model mới đã tích hợp bộ chuẩn hóa (Scaler) bên trong.")
print("-> Bạn không cần sửa code run_realtime.py, cứ chạy là nó tự hiểu!")