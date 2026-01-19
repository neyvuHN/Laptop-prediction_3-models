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
from sklearn.preprocessing import OneHotEncoder

# Optional imports (XGBoost/LightGBM) - Handle cases where they might not be installed
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# =========================
# 1. CORE UTILITY FUNCTIONS
# =========================

def load_data(
    default_dir: Path,
    train_uploader,
    val_uploader,
    test_uploader,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test CSV from Uploaders or Default path."""
    def _read_csv(uploaded, fallback_path: Path) -> pd.DataFrame:
        if uploaded is not None:
            return pd.read_csv(uploaded)
        if fallback_path.exists():
            return pd.read_csv(fallback_path)
        # Return empty DF if nothing found to prevent crash before upload
        return pd.DataFrame() 

    train_path = default_dir / "data_train.csv"
    val_path = default_dir / "data_validation.csv"
    test_path = default_dir / "data_test.csv"

    df_train = _read_csv(train_uploader, train_path)
    df_val = _read_csv(val_uploader, val_path)
    df_test = _read_csv(test_uploader, test_path)

    return df_train, df_val, df_test

def align_columns(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "price_base",
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Ensure Schema Consistency across datasets."""
    drop_cols = drop_cols or []
    must_drop = set(drop_cols + [target_col])

    # Feature candidates
    f_train = [c for c in df_train.columns if c not in must_drop]
    f_val = [c for c in df_val.columns if c not in must_drop]
    f_test = [c for c in df_test.columns if c not in must_drop]

    # Intersection to find common features
    inter = [c for c in f_train if (c in f_val) and (c in f_test)]

    if not inter:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    X_train = df_train[inter].copy()
    X_val = df_val[inter].copy()
    X_test = df_test[inter].copy()

    return X_train, X_val, X_test, inter

def build_model(model_name: str, params: Dict) -> object:
    """Factory function to create model instances."""
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=None if params.get("max_depth", 0) == 0 else int(params["max_depth"]),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=int(params.get("random_state", 42)),
        )

    if model_name == "XGBoostRegressor":
        if XGBRegressor is None: raise ImportError("XGBoost not installed.")
        return XGBRegressor(
            n_estimators=int(params.get("n_estimators", 1500)),
            max_depth=int(params.get("max_depth", 8)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            n_jobs=-1,
            random_state=int(params.get("random_state", 42)),
        )

    if model_name == "LightGBMRegressor":
        if LGBMRegressor is None: raise ImportError("LightGBM not installed.")
        return LGBMRegressor(
            n_estimators=int(params.get("n_estimators", 3000)),
            num_leaves=int(params.get("num_leaves", 63)),
            max_depth=-1 if params.get("max_depth", 0) == 0 else int(params["max_depth"]),
            learning_rate=float(params.get("learning_rate", 0.03)),
            random_state=int(params.get("random_state", 42)),
        )

    raise ValueError(f"Unknown model: {model_name}")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # Safe MAPE
    y_true_safe = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100.0)
    
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE(%)": mape,
    }

def train_and_eval(X_train, y_train, X_val, y_val, model_name, params):
    # Identify column types
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # Preprocessing Pipelines
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ], remainder="drop", verbose_feature_names_out=False)

    # Full Pipeline
    model = build_model(model_name, params)
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

    # Fit
    pipe.fit(X_train, y_train)
    
    # Predict Val
    y_pred_val = pipe.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred_val)
    
    return pipe, metrics, y_pred_val

def predict_single(pipe, row_df, expected_cols):
    """Predict for a single row, handling missing/extra columns."""
    x = pd.DataFrame(columns=expected_cols)
    # Combine schema with input data
    for c in expected_cols:
        if c in row_df.columns:
            x.loc[0, c] = row_df.iloc[0][c]
        else:
            x.loc[0, c] = np.nan # Pipeline will impute this
    
    # Ensure dtypes match loosely to prevent object/float mismatches
    x = x.infer_objects()
    return pipe.predict(x)[0]

# =========================
# 2. STREAMLIT UI SETUP
# =========================
st.set_page_config(page_title="Laptop Price AI", layout="wide", page_icon="💻")
st.title("💻 Hệ thống Huấn luyện & Dự báo Giá Laptop AI")
st.write("Tích hợp Pipeline tự động hóa: Upload dữ liệu -> Train Model -> Dự báo giá.")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("📂 1. Dữ liệu đầu vào")
    st.caption("Upload các file CSV tương ứng")
    train_up = st.file_uploader("Train Set (data_train.csv)", type=["csv"], key="train_up")
    val_up = st.file_uploader("Val Set (data_validation.csv)", type=["csv"], key="val_up")
    test_up = st.file_uploader("Test Set (data_test.csv)", type=["csv"], key="test_up")

    st.divider()
    st.header("⚙️ 2. Cấu hình Model")
    
    avail_models = ["RandomForestRegressor"]
    if XGBRegressor: avail_models.append("XGBoostRegressor")
    if LGBMRegressor: avail_models.append("LightGBMRegressor")
    
    model_name = st.selectbox("Chọn thuật toán", avail_models)
    
    # Hyperparameters simple UI
    params = {"random_state": 42}
    if model_name == "RandomForestRegressor":
        params["n_estimators"] = st.slider("Số lượng cây (n_estimators)", 100, 2000, 500, 100)
        params["max_depth"] = st.slider("Độ sâu tối đa (0=None)", 0, 50, 0)
    elif model_name == "XGBoostRegressor":
        params["n_estimators"] = st.slider("n_estimators", 500, 5000, 1000)
        params["learning_rate"] = st.number_input("Learning Rate", 0.001, 0.5, 0.05)
    elif model_name == "LightGBMRegressor":
        params["n_estimators"] = st.slider("n_estimators", 500, 5000, 2000)
        params["learning_rate"] = st.number_input("Learning Rate", 0.001, 0.5, 0.05)

    st.divider()
    train_btn = st.button("🚀 Bắt đầu Huấn luyện", type="primary")

# --- DATA LOADING ---
default_dir = Path(__file__).parent
df_train, df_val, df_test = load_data(default_dir, train_up, val_up, test_up)

# --- TABS LAYOUT ---
tab_data, tab_train, tab_pred = st.tabs(["📊 Dữ liệu", "🧠 Huấn luyện & Đánh giá", "🔮 Dự đoán giá"])

# --- TAB 1: DATA OVERVIEW ---
with tab_data:
    if df_train.empty:
        st.warning("Vui lòng upload file `data_train.csv` ở menu bên trái để bắt đầu.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.subheader("Train Set")
            st.dataframe(df_train.head(), use_container_width=True)
            st.caption(f"Shape: {df_train.shape}")
        with c2: 
            st.subheader("Validation Set")
            st.dataframe(df_val.head(), use_container_width=True)
            st.caption(f"Shape: {df_val.shape}")
        with c3: 
            st.subheader("Test Set")
            st.dataframe(df_test.head(), use_container_width=True)
            st.caption(f"Shape: {df_test.shape}")

# --- GLOBAL VARIABLES & STATE ---
TARGET = "price_base"
DROP_COLS = ["title", "link", "id"] # Add any ID columns to drop

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []

# --- TAB 2: TRAINING ---
with tab_train:
    if train_btn:
        if df_train.empty or df_val.empty:
            st.error("Thiếu dữ liệu Train hoặc Validation!")
        elif TARGET not in df_train.columns:
            st.error(f"Không tìm thấy cột mục tiêu '{TARGET}' trong dữ liệu.")
        else:
            with st.spinner("Đang xử lý dữ liệu và huấn luyện mô hình..."):
                # 1. Align Data
                X_train, X_val, X_test, feature_cols = align_columns(
                    df_train, df_val, df_test, target_col=TARGET, drop_cols=DROP_COLS
                )
                
                if not feature_cols:
                    st.error("Không tìm thấy cột dữ liệu chung (feature intersection).")
                    st.stop()

                y_train = df_train[TARGET]
                y_val = df_val[TARGET]
                
                # 2. Train
                pipe, metrics, y_pred_val = train_and_eval(
                    X_train, y_train, X_val, y_val, model_name, params
                )
                
                # 3. Save State
                st.session_state.pipeline = pipe
                st.session_state.feature_cols = feature_cols
                st.session_state.metrics = metrics
                st.session_state.y_val = y_val
                st.session_state.y_pred_val = y_pred_val
                
                st.success("Huấn luyện hoàn tất!")

    # Display Results if available
    if st.session_state.pipeline is not None:
        metrics = st.session_state.metrics
        st.subheader("Kết quả đánh giá (Validation Set)")
        
        # Metrics Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE (Sai số tuyệt đối)", f"{metrics['MAE']:,.0f} đ")
        m2.metric("RMSE", f"{metrics['RMSE']:,.0f}")
        m3.metric("MAPE (Sai số %)", f"{metrics['MAPE(%)']:.2f} %")
        m4.metric("R2 Score", f"{metrics['R2']:.3f}")
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Biểu đồ Dự đoán vs Thực tế**")
            fig, ax = plt.subplots()
            ax.scatter(st.session_state.y_val, st.session_state.y_pred_val, alpha=0.5, color='blue')
            min_val = min(st.session_state.y_val.min(), st.session_state.y_pred_val.min())
            max_val = max(st.session_state.y_val.max(), st.session_state.y_pred_val.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            ax.set_xlabel("Giá Thực tế")
            ax.set_ylabel("Giá Dự báo")
            st.pyplot(fig)
            
        with c2:
            st.markdown("**Feature Importance (Top 20)**")
            try:
                # Try getting importance from inner model
                model = st.session_state.pipeline.named_steps['model']
                preprocessor = st.session_state.pipeline.named_steps['preprocess']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    # Try getting feature names from OneHot
                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except:
                        feature_names = [f"Feature {i}" for i in range(len(importances))]
                    
                    # Sort and plot
                    indices = np.argsort(importances)[::-1][:20]
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.barh(range(len(indices)), importances[indices], align='center')
                    ax2.set_yticks(range(len(indices)))
                    ax2.set_yticklabels(np.array(feature_names)[indices])
                    ax2.invert_yaxis()
                    st.pyplot(fig2)
                else:
                    st.info("Model này không hỗ trợ hiển thị Feature Importance.")
            except Exception as e:
                st.warning(f"Không thể vẽ Feature Importance: {e}")

        # Download Model
        st.divider()
        buf = io.BytesIO()
        joblib.dump(st.session_state.pipeline, buf)
        st.download_button(
            "⬇️ Tải xuống Model đã train (.joblib)", 
            data=buf.getvalue(), 
            file_name="laptop_price_model.joblib", 
            mime="application/octet-stream"
        )

# --- TAB 3: PREDICTION ---
with tab_pred:
    if st.session_state.pipeline is None:
        st.info("⚠️ Bạn cần huấn luyện mô hình ở Tab 2 trước.")
    else:
        st.subheader("Dự báo giá Laptop theo thông số")
        st.markdown("Form nhập liệu được tạo tự động dựa trên các cột trong dữ liệu Train.")
        
        feature_cols = st.session_state.feature_cols
        
        with st.form("pred_form"):
            # Create dynamic inputs based on columns
            input_data = {}
            cols_per_row = 3
            cols = st.columns(cols_per_row)
            
            # Determine data types from X_train (cached implicitly via features)
            # We use a trick: If we don't have X_train stored, we assume numbers for specific keywords, else text
            
            for i, col_name in enumerate(feature_cols):
                with cols[i % cols_per_row]:
                    # Simple heuristic for input types if we don't have types stored
                    # In a full app, we should store dtypes in session_state
                    label = col_name.replace("_", " ").title()
                    
                    if any(x in col_name.lower() for x in ['size', 'gen', 'core', 'thread', 'score', 'px', 'wh', 'cell', 'nits', 'percent']):
                        val = st.number_input(label, value=0.0)
                        input_data[col_name] = val
                    elif any(x in col_name.lower() for x in ['anti_glare', 'touch']):
                        val = st.selectbox(label, [0, 1])
                        input_data[col_name] = val
                    else:
                        # Assuming text/categorical for brands, types
                        val = st.text_input(label, "")
                        input_data[col_name] = val if val != "" else np.nan

            submit_pred = st.form_submit_button("💰 Định giá ngay")
        
        if submit_pred:
            row_df = pd.DataFrame([input_data])
            try:
                pred_price = predict_single(st.session_state.pipeline, row_df, feature_cols)
                
                # Calculate simple range based on MAE
                mae = st.session_state.metrics['MAE']
                lower = max(0, pred_price - mae)
                upper = pred_price + mae
                
                st.divider()
                c_res1, c_res2 = st.columns([1, 2])
                with c_res1:
                    st.metric("Giá dự báo", f"{pred_price:,.0f} VNĐ")
                with c_res2:
                    st.info(f"**Khoảng giá khuyến nghị (±MAE):**\n\n{lower:,.0f} - {upper:,.0f} VNĐ")
                    
            except Exception as e:
                st.error(f"Lỗi dự báo: {str(e)}")
