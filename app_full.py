import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# 1. ĐỊNH NGHĨA CLASS (Bắt buộc phải nằm trước khi load pickle)
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

    def fit(self, X, y_log):
        self.model.fit(X, y_log)

    def predict(self, X):
        return self.model.predict(X)

# 2. DANH SÁCH FEATURE CHUẨN (39 columns)
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

# 3. CẤU HÌNH SAI SỐ MAE (Dựa trên kết quả train của bạn)
# Lưu ý: Key ở đây phải khớp với Key bạn đặt trong file laptop_models.pkl
MODEL_METRICS = {
    'LightGBM': {'mae': 4110382, 'is_log': True},
    'XGBoost': {'mae': 3965587, 'is_log': False},
    'Random Forest': {'mae': 3795342, 'is_log': False}
}

@st.cache_resource
def load_models():
    try:
        with open('laptop_models.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# --- THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# Sidebar để chọn mô hình
st.sidebar.header("🤖 Cấu hình dự đoán")
all_data = load_models()

if all_data is None:
    st.error("❌ Không tìm thấy file 'laptop_models.pkl'. Vui lòng kiểm tra lại!")
    st.stop()

models_in_pkl = all_data['models']
model_options = ["So sánh cả 3"] + list(MODEL_METRICS.keys())
selected_model = st.sidebar.selectbox("Chọn mô hình dự đoán:", model_options)

st.title("💻 Hệ thống Dự báo Giá Laptop")
st.write(f"Đang sử dụng: **{selected_model}**")

# --- FORM NHẬP LIỆU ---
with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["🚀 Cấu hình lõi", "🖥️ Màn hình", "🔋 Pin & Thương hiệu"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        cpu_cores = c1.number_input("Số nhân CPU", 2, 64, 8)
        cpu_threads = c2.number_input("Số luồng CPU", 2, 128, 12)
        cpu_gen = c3.number_input("Thế hệ CPU (Gen)", 0, 14, 0)
        
        ram_size = c1.selectbox("Dung lượng RAM (GB)", [4, 8, 16, 32, 64, 128])
        ram_type = c2.selectbox("Loại RAM", ["DDR4", "DDR5", "DDR3", "DDR3L", "DDR4X", "DDR5X", "DDR6", "RAM_TYPE"])
        storage_size = c3.number_input("Ổ cứng (GB)", 128, 4096, 512)
        
        cpu_brand = c1.radio("Hãng CPU", ["Intel", "AMD"])
        gpu_vram = c2.selectbox("VRAM (GB)", [0, 2, 4, 6, 8, 12, 16])
        gpu_class = c3.radio("Loại GPU", ["Discrete", "Integrated", "Unknown"])

    with tab2:
        c1, c2, c3 = st.columns(3)
        screen_size = c1.number_input("Kích thước (inch)", 10.0, 18.0, 15.6)
        brightness = c2.number_input("Độ sáng (nits)", 200, 1000, 300)
        anti_glare = c3.toggle("Chống chói (Anti-glare)")
        
        res_w = c1.number_input("Độ phân giải ngang (px)", 1280, 3840, 1920)
        res_h = c2.number_input("Độ phân giải dọc (px)", 720, 2400, 1080)
        srgb = c3.slider("Độ phủ màu sRGB (%)", 45, 100, 100)

    with tab3:
        c1, c2 = st.columns(2)
        brand_score = c1.slider("Điểm thương hiệu (Brand Score)", 10.0, 30.0, 19.0)
        gpu_brand = c2.selectbox("Hãng GPU", ["NVIDIA", "AMD", "Intel", "Other", "Unknown"])
        
        battery_wh = c1.number_input("Dung lượng Pin (Wh)", 30, 100, 50)
        battery_cells = c2.number_input("Số Cell Pin", 2, 6, 3)

    submit = st.form_submit_button("📊 PHÂN TÍCH GIÁ")

# --- XỬ LÝ DỰ ĐOÁN ---
if submit:
    # 1. Tạo DataFrame input
    input_df = pd.DataFrame(0.0, index=[0], columns=TRAIN_COLS)
    
    # 2. Map dữ liệu số
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
    input_df['gpu_brand_te'] = 17000000.0 # Placeholder
    input_df['gpu_class_te'] = 17000000.0 # Placeholder

    # 3. Map One-hot Encoding
    if f"ram_type_{ram_type}" in TRAIN_COLS: input_df[f"ram_type_{ram_type}"] = 1
    if f"cpu_brand_{cpu_brand}" in TRAIN_COLS: input_df[f"cpu_brand_{cpu_brand}"] = 1
    if f"gpu_brand_{gpu_brand}" in TRAIN_COLS: input_df[f"gpu_brand_{gpu_brand}"] = 1
    if f"gpu_class_{gpu_class}" in TRAIN_COLS: input_df[f"gpu_class_{gpu_class}"] = 1

    # 4. Thực hiện dự đoán
    st.divider()
    
    # Xác định danh sách mô hình cần chạy
    if selected_model == "So sánh cả 3":
        models_to_run = list(MODEL_METRICS.keys())
    else:
        models_to_run = [selected_model]

    cols = st.columns(len(models_to_run))

    for i, m_name in enumerate(models_to_run):
        # Lấy mô hình từ file pkl (Lưu ý: key trong pkl có thể là 'lgbm', 'xgb'... hãy kiểm tra lại)
        # Ở đây mình giả định key trong pkl khớp với MODEL_METRICS.keys()
        model_obj = models_in_pkl.get(m_name)
        
        if model_obj:
            raw_pred = model_obj.predict(input_df)[0]
            
            # Xử lý Log transform
            final_pred = np.expm1(raw_pred) if MODEL_METRICS[m_name]['is_log'] else raw_pred
            
            # Tính khoảng giá
            mae = MODEL_METRICS[m_name]['mae']
            lower = max(0, final_pred - mae)
            upper = final_pred + mae
            
            with cols[i]:
                st.subheader(f"🤖 {m_name}")
                st.metric("Giá dự báo", f"{final_pred:,.0f} đ")
                st.info(f"**Khoảng giá ước tính:**\n\n{lower:,.0f} - {upper:,.0f} VNĐ")
        else:
            st.warning(f"Mô hình '{m_name}' không tìm thấy trong file pkl.")
