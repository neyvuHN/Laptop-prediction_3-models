import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# --- BẮT BUỘC: Định nghĩa Class OptimizedModel để Pickle có thể đọc được ---
class OptimizedModel:
    def __init__(self, n_est=1000, lr=0.03, depth=6):
        self.model = GradientBoostingRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=depth,
            subsample=0.8,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
    def predict(self, X):
        return self.model.predict(X)

# --- DANH SÁCH FEATURE CHUẨN (39 Features) ---
TRAIN_COLS = [
    'cpu_cores', 'cpu_threads', 'ram_size', 'storage_size', 'screen_size', 
    'raw_specs_count', 'ram_type_DDR3', 'ram_type_DDR3L', 'ram_type_DDR4', 
    'ram_type_DDR4X', 'ram_type_DDR5', 'ram_type_DDR5X', 'ram_type_DDR6', 
    'ram_type_RAM_TYPE', 'brand_score', 'cpu_gen', 'cpu_performance_score', 
    'cpu_brand_Intel', 'cpu_brand_AMD', 'gpu_vram', 'gpu_brand_te', 
    'gpu_class_te', 'screen_brightness_nits', 'screen_srgb_percent', 
    'screen_ntsc_percent', 'screen_anti_glare', 'battery_wh', 'battery_cells', 
    'res_width', 'res_height', 'res_total_pixels', 'gpu_brand_AMD', 
    'gpu_brand_Intel', 'gpu_brand_NVIDIA', 'gpu_brand_Other', 
    'gpu_brand_Unknown', 'gpu_class_Discrete', 'gpu_class_Integrated', 
    'gpu_class_Unknown'
]

# --- THÔNG TIN MÔ HÌNH (Đã sửa Key cho khớp với file pkl của bạn) ---
MODEL_INFO = {
    'lgbm': {'name': 'LightGBM', 'mae': 4110382, 'is_log': True},
    'xgb': {'name': 'XGBoost', 'mae': 3965587, 'is_log': False},
    'rf': {'name': 'Random Forest', 'mae': 3795342, 'is_log': False}
}

@st.cache_resource
def load_data():
    try:
        with open('laptop_models.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi đọc file pkl: {e}")
        return None

# --- GIAO DIỆN ---
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Hệ thống Dự báo Giá Laptop")
st.write("Nhập thông số để xem kết quả so sánh từ 3 thuật toán.")

all_data = load_data()
if not all_data:
    st.stop()

models = all_data['models']

# --- NHẬP LIỆU ---
with st.form("input_form"):
    tab1, tab2, tab3 = st.tabs(["🚀 Cấu hình", "🖥️ Màn hình", "🔋 Pin & Khác"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        cpu_cores = c1.number_input("Số nhân CPU", 2, 64, 8)
        cpu_threads = c2.number_input("Số luồng CPU", 2, 128, 16)
        cpu_gen = c3.number_input("Thế hệ CPU (Gen)", 0, 14, 12)
        ram_size = c1.selectbox("RAM (GB)", [8, 16, 32, 64])
        ram_type = c2.selectbox("Loại RAM", ["DDR4", "DDR5", "LPDDR5", "DDR3"])
        storage_size = c3.selectbox("Ổ cứng (GB)", [256, 512, 1024, 2048])
        cpu_brand = c1.radio("Hãng CPU", ["Intel", "AMD"])
        gpu_vram = c2.selectbox("VRAM (GB)", [0, 4, 6, 8, 12])
        gpu_class = c3.radio("Loại GPU", ["Discrete", "Integrated"])

    with tab2:
        c1, c2, c3 = st.columns(3)
        screen_size = c1.number_input("Kích thước (inch)", 12.0, 18.0, 15.6)
        brightness = c2.number_input("Độ sáng (nits)", 200, 1000, 300)
        srgb = c3.slider("sRGB (%)", 45, 100, 100)
        res_w = c1.number_input("Ngang (px)", 1280, 3840, 1920)
        res_h = c2.number_input("Dọc (px)", 720, 2400, 1080)
        anti_glare = c3.checkbox("Chống chói")

    with tab3:
        c1, c2 = st.columns(2)
        brand_score = c1.slider("Điểm thương hiệu", 10.0, 30.0, 20.0)
        gpu_brand = c2.selectbox("Hãng GPU", ["NVIDIA", "AMD", "Intel"])
        battery_wh = c1.number_input("Dung lượng Pin (Wh)", 30, 100, 56)
        battery_cells = c2.number_input("Số Cell Pin", 2, 6, 3)

    submit = st.form_submit_button("📊 PHÂN TÍCH GIÁ")

# --- XỬ LÝ DỰ ĐOÁN ---
if submit:
    # Chuẩn bị dữ liệu đầu vào
    input_df = pd.DataFrame(0.0, index=[0], columns=TRAIN_COLS)
    
    # Gán các giá trị trực tiếp
    input_df['cpu_cores'] = cpu_cores
    input_df['cpu_threads'] = cpu_threads
    input_df['ram_size'] = ram_size
    input_df['storage_size'] = storage_size
    input_df['screen_size'] = screen_size
    input_df['cpu_gen'] = cpu_gen
    input_df['screen_brightness_nits'] = brightness
    input_df['screen_srgb_percent'] = srgb
    input_df['battery_wh'] = battery_wh
    input_df['battery_cells'] = battery_cells
    input_df['res_width'] = res_w
    input_df['res_height'] = res_h
    input_df['res_total_pixels'] = res_w * res_h
    input_df['gpu_vram'] = gpu_vram
    input_df['brand_score'] = brand_score
    input_df['screen_anti_glare'] = 1 if anti_glare else 0
    input_df['raw_specs_count'] = 25 
    
    # Encode One-hot
    if f"ram_type_{ram_type}" in TRAIN_COLS: input_df[f"ram_type_{ram_type}"] = 1
    if f"cpu_brand_{cpu_brand}" in TRAIN_COLS: input_df[f"cpu_brand_{cpu_brand}"] = 1
    if f"gpu_brand_{gpu_brand}" in TRAIN_COLS: input_df[f"gpu_brand_{gpu_brand}"] = 1
    if f"gpu_class_{gpu_class}" in TRAIN_COLS: input_df[f"gpu_class_{gpu_class}"] = 1
    
    # Placeholder cho Target Encoding
    input_df['gpu_brand_te'] = 15000000.0
    input_df['gpu_class_te'] = 15000000.0

    st.divider()
    cols = st.columns(3)
    
    # Hiển thị kết quả từ 3 mô hình
    for i, (m_key, info) in enumerate(MODEL_INFO.items()):
        if m_key in models:
            # Dự đoán
            pred = models[m_key].predict(input_df)[0]
            
            # Chuyển đổi ngược nếu là Log (cho LightGBM)
            final_price = np.expm1(pred) if info['is_log'] else pred
            
            # Tính khoảng giá dựa trên MAE
            lower = max(0, final_price - info['mae'])
            upper = final_price + info['mae']
            
            with cols[i]:
                st.subheader(f"🤖 {info['name']}")
                st.metric("Giá dự báo", f"{final_price:,.0f} đ")
                st.info(f"**Khoảng giá ước tính:**\n\n{lower:,.0f} - {upper:,.0f} VNĐ")
        else:
            with cols[i]:
                st.warning(f"Không tìm thấy mô hình '{m_key}' trong file pkl.")
