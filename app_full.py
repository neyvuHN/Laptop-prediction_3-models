import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional imports
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# =========================
# 1. CẤU HÌNH & HÀM HỖ TRỢ
# =========================

# Tham số tối ưu (Best params) - Đặt làm mặc định
BEST_PARAMS = {
    "RandomForestRegressor": {"n_estimators": 1000, "max_depth": 20},
    "XGBoostRegressor": {"n_estimators": 2000, "learning_rate": 0.03, "max_depth": 8},
    "LightGBMRegressor": {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 31}
}

def load_data(default_dir: Path, train_up, val_up, test_up) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _read_csv(uploaded, fallback_path: Path) -> pd.DataFrame:
        if uploaded is not None: return pd.read_csv(uploaded)
        if fallback_path.exists(): return pd.read_csv(fallback_path)
        return pd.DataFrame() 

    df_train = _read_csv(train_up, default_dir / "data_train.csv")
    df_val = _read_csv(val_up, default_dir / "data_validation.csv")
    df_test = _read_csv(test_up, default_dir / "data_test.csv")
    return df_train, df_val, df_test

def train_and_eval(X_train, y_train, X_val, y_val, model_name, params):
    # Tự động phân loại cột số và cột chữ
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Pipeline xử lý số: Điền dữ liệu thiếu bằng trung vị -> Chuẩn hóa
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()) 
    ])

    # Pipeline xử lý chữ: Điền thiếu -> One-Hot Encode (tự động biến 'Intel' thành vector)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ], remainder="drop", verbose_feature_names_out=False)

    # Khởi tạo Model
    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(n_jobs=-1, random_state=42, **params)
    elif model_name == "XGBoostRegressor":
        model = XGBRegressor(n_jobs=-1, random_state=42, **params)
    elif model_name == "LightGBMRegressor":
        model = LGBMRegressor(random_state=42, **params)
    
    # Ghép toàn bộ thành 1 Pipeline
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    
    # Train
    pipe.fit(X_train, y_train)
    
    # Eval
    y_pred = pipe.predict(X_val)
    metrics = {
        "MAE": mean_absolute_error(y_val, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)),
        "R2": r2_score(y_val, y_pred)
    }
    return pipe, metrics, y_pred

# =========================
# 2. GIAO DIỆN STREAMLIT
# =========================
st.set_page_config(page_title="Laptop Price Predictor", layout="wide", page_icon="💻")
st.title("💻 Hệ thống Dự báo Giá Laptop AI")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Dữ liệu")
    train_up = st.file_uploader("Upload Train", type=["csv"])
    val_up = st.file_uploader("Upload Validation", type=["csv"])
    test_up = st.file_uploader("Upload Test", type=["csv"])
    
    st.divider()
    st.header("2. Model & Hyperparameters")
    
    # Chọn Model
    avail_models = ["RandomForestRegressor"]
    if XGBRegressor: avail_models.append("XGBoostRegressor")
    if LGBMRegressor: avail_models.append("LightGBMRegressor")
    
    model_name = st.selectbox("Chọn thuật toán", avail_models, index=len(avail_models)-1) # Mặc định chọn cái cuối (thường là LightGBM/XGB)
    
    # Lấy tham số mặc định tối ưu
    defaults = BEST_PARAMS.get(model_name, {})
    
    params = {}
    if model_name == "RandomForestRegressor":
        params["n_estimators"] = st.number_input("n_estimators", 100, 5000, defaults.get("n_estimators", 1000))
        params["max_depth"] = st.number_input("max_depth (0=None)", 0, 100, defaults.get("max_depth", 0))
        if params["max_depth"] == 0: params["max_depth"] = None
        
    elif model_name == "XGBoostRegressor":
        params["n_estimators"] = st.number_input("n_estimators", 100, 5000, defaults.get("n_estimators", 2000))
        params["learning_rate"] = st.number_input("learning_rate", 0.001, 1.0, defaults.get("learning_rate", 0.03), format="%.3f")
        params["max_depth"] = st.number_input("max_depth", 1, 20, defaults.get("max_depth", 8))
        
    elif model_name == "LightGBMRegressor":
        params["n_estimators"] = st.number_input("n_estimators", 100, 5000, defaults.get("n_estimators", 3000))
        params["learning_rate"] = st.number_input("learning_rate", 0.001, 1.0, defaults.get("learning_rate", 0.03), format="%.3f")
        params["num_leaves"] = st.number_input("num_leaves", 10, 200, defaults.get("num_leaves", 31))

    st.divider()
    train_btn = st.button("🚀 Huấn luyện Model", type="primary")

# --- XỬ LÝ DỮ LIỆU ---
default_dir = Path(__file__).parent
df_train, df_val, df_test = load_data(default_dir, train_up, val_up, test_up)

TARGET = "price_base"
DROP_COLS = ["title", "link", "id", "product_url"] # Các cột không dùng để train

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None

# --- TAB 1: HUẤN LUYỆN ---
tab_train, tab_pred = st.tabs(["🧠 Huấn luyện & Đánh giá", "🔮 Dự báo giá (Giao diện chuẩn)"])

with tab_train:
    if train_btn:
        if df_train.empty or df_val.empty:
            st.error("Chưa có dữ liệu Train/Validation!")
        elif TARGET not in df_train.columns:
            st.error(f"Không tìm thấy cột mục tiêu '{TARGET}'.")
        else:
            with st.spinner("Đang huấn luyện Pipeline (Encode -> Train)..."):
                # Chuẩn bị dữ liệu
                cols_to_drop = [c for c in DROP_COLS if c in df_train.columns] + [TARGET]
                X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
                y_train = df_train[TARGET]
                
                X_val = df_val.drop(columns=cols_to_drop, errors='ignore')
                y_val = df_val[TARGET]
                
                # Lưu danh sách cột feature để khớp lúc dự đoán
                st.session_state.feature_names = X_train.columns.tolist()
                
                # Train
                pipe, metrics, y_pred_val = train_and_eval(X_train, y_train, X_val, y_val, model_name, params)
                
                st.session_state.pipeline = pipe
                st.session_state.metrics = metrics
                st.success("Huấn luyện xong!")
    
    # Hiển thị kết quả
    if st.session_state.metrics:
        m = st.session_state.metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (Sai số)", f"{m['MAE']:,.0f} VNĐ")
        c2.metric("RMSE", f"{m['RMSE']:,.0f}")
        c3.metric("R2 Score", f"{m['R2']:.3f}")

# --- TAB 2: DỰ BÁO (GIAO DIỆN CŨ CỦA BẠN) ---
with tab_pred:
    if st.session_state.pipeline is None:
        st.warning("⚠️ Vui lòng huấn luyện model ở Tab 1 trước.")
    else:
        st.write("Nhập thông số cấu hình (Input thô - Pipeline sẽ tự động Encode):")
        
        with st.form("prediction_form"):
            t1, t2, t3 = st.tabs(["🚀 Cấu hình lõi", "🖥️ Màn hình", "🔋 Pin & Khác"])
            
            with t1:
                c1, c2, c3 = st.columns(3)
                cpu_cores = c1.number_input("Số nhân CPU", 2, 64, 8)
                cpu_threads = c2.number_input("Số luồng CPU", 2, 128, 12)
                cpu_gen = c3.number_input("Thế hệ CPU (Gen)", 0, 14, 12)
                
                ram_size = c1.selectbox("RAM (GB)", [4, 8, 16, 32, 64, 128], index=2)
                # Lưu ý: Value của selectbox phải khớp với dữ liệu trong file csv (vd: DDR4, DDR5)
                ram_type = c2.selectbox("Loại RAM", ["DDR4", "DDR5", "DDR3", "LPDDR4", "LPDDR5", "Other"], index=0)
                storage_size = c3.number_input("SSD (GB)", 128, 8192, 512)
                
                cpu_brand = c1.radio("Hãng CPU", ["Intel", "AMD", "Apple", "Other"], horizontal=True)
                gpu_vram = c2.selectbox("VRAM (GB)", [0, 2, 4, 6, 8, 12, 16, 24], index=2)
                gpu_class = c3.radio("Loại GPU", ["Discrete", "Integrated", "Unknown"], horizontal=True)

            with t2:
                c1, c2, c3 = st.columns(3)
                screen_size = c1.number_input("Kích thước (inch)", 10.0, 18.0, 15.6)
                brightness = c2.number_input("Độ sáng (nits)", 200, 1000, 300)
                # True/False sẽ được convert sang 1/0 hoặc giữ nguyên tùy pipeline
                anti_glare = c3.toggle("Chống chói (Anti-glare)", value=True)
                
                res_w = c1.number_input("Độ phân giải ngang (px)", 1280, 4000, 1920)
                res_h = c2.number_input("Độ phân giải dọc (px)", 720, 3000, 1080)
                srgb = c3.slider("Độ phủ màu sRGB (%)", 45, 100, 65)

            with t3:
                c1, c2 = st.columns(2)
                brand_score = c1.slider("Điểm thương hiệu (Brand Score)", 0, 30, 20)
                gpu_brand = c2.selectbox("Hãng GPU", ["NVIDIA", "AMD", "Intel", "Apple", "Other"])
                
                battery_wh = c1.number_input("Pin (Wh)", 30.0, 99.9, 50.0)
                battery_cells = c2.number_input("Số Cell Pin", 2, 6, 3)

            submit = st.form_submit_button("💰 ĐỊNH GIÁ NGAY")

        if submit:
            # 1. Tạo DataFrame từ Input thô (Raw Data)
            # Tên cột (keys) phải TRÙNG KHỚP với tên cột trong file CSV huấn luyện
            input_data = {
                'cpu_cores': cpu_cores,
                'cpu_threads': cpu_threads,
                'ram_size': ram_size,
                'storage_size': storage_size,
                'screen_size': screen_size,
                'cpu_gen': cpu_gen,
                'screen_brightness_nits': brightness,
                'screen_srgb_percent': srgb,
                'battery_wh': battery_wh,
                'battery_cells': battery_cells,
                'res_width': res_w,
                'res_height': res_h,
                'res_total_pixels': res_w * res_h,
                'gpu_vram': gpu_vram,
                'brand_score': brand_score,
                'screen_anti_glare': 1 if anti_glare else 0,
                'raw_specs_count': 25, # Giá trị placeholder nếu cần
                
                # Cột dạng chữ (Category) - Pipeline sẽ tự lo One-Hot Encoding
                'ram_type': ram_type,
                'cpu_brand': cpu_brand,
                'gpu_brand': gpu_brand,   # Không cần gpu_brand_te nữa
                'gpu_class': gpu_class    # Không cần gpu_class_te nữa
            }
            
            # Chuyển thành DataFrame 1 dòng
            input_df = pd.DataFrame([input_data])
            
            # Lọc chỉ giữ lại các cột mà Model đã học (để tránh lỗi thừa cột)
            valid_cols = [c for c in st.session_state.feature_names if c in input_df.columns]
            
            # Nếu thiếu cột nào đó (ví dụ file train có cột 'weight' mà form không có) -> Tạo cột đó với giá trị NaN
            missing_cols = set(st.session_state.feature_names) - set(input_df.columns)
            for c in missing_cols:
                input_df[c] = np.nan # Imputer trong pipeline sẽ điền giá trị này
            
            # Sắp xếp lại cột cho đúng thứ tự
            input_df = input_df[st.session_state.feature_names]

            # 2. Dự báo
            try:
                pred = st.session_state.pipeline.predict(input_df)[0]
                
                # Tính khoảng giá
                mae = st.session_state.metrics['MAE']
                lower = pred - mae
                upper = pred + mae
                
                st.divider()
                c_kq1, c_kq2 = st.columns([1, 2])
                with c_kq1:
                    st.success("KẾT QUẢ DỰ BÁO")
                    st.metric("Giá trung bình", f"{pred:,.0f} đ")
                with c_kq2:
                    st.info(f"Khoảng giá tin cậy (±MAE):\n### {lower:,.0f} - {upper:,.0f} VNĐ")
                    
            except Exception as e:
                st.error(f"Lỗi khi dự báo: {str(e)}")
                st.write("Vui lòng kiểm tra lại tên cột trong file CSV có khớp với form nhập liệu không.")
