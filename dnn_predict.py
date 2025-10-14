import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import os 
from tensorflow.keras.models import load_model
from datetime import datetime
from io import BytesIO 

# 모델 파일 저장 베이스 경로 (dnn_train.py와 동일)
BASE_MODEL_PATH = "model"

# 모델 구성 요소 파일 이름 (폴더 내에서 고정됨)
MODEL_FILE_NAME = "eepf_dnn_model.h5"
PARAMS_FILE_NAME = "eepf_norm_params.npz"

# EEPF 스펙트럼 범위 정의
EV_MIN = 0.0
EV_MAX = 17.01
EV_STEP = 0.045

def get_available_models():
    """
    BASE_MODEL_PATH 내에서 사용 가능한 모든 모델 버전(폴더 이름)을 가져옵니다.
    가장 최근 학습된 모델이 위로 오도록 역순 정렬합니다.
    """
    if not os.path.exists(BASE_MODEL_PATH):
        return []
    
    # 폴더 이름만 필터링하여 반환
    model_versions = sorted([
        d for d in os.listdir(BASE_MODEL_PATH) 
        if os.path.isdir(os.path.join(BASE_MODEL_PATH, d))
    ], reverse=True)
    
    return model_versions

@st.cache_resource(show_spinner="저장된 모델 및 매개변수 로드 중...")
def load_model_for_prediction(model_version_name: str):
    """
    선택된 모델 버전(폴더 이름)에 해당하는 모델 파일과 정규화 매개변수를 로드합니다.
    """
    # 로드 실패 시 반환할 None 튜플 (Unpacking 에러 방지)
    ERROR_RETURN = (None, None, None, None) 

    # 전체 파일 경로 생성: model/버전_폴더_이름/파일_이름
    MODEL_PATH = os.path.join(BASE_MODEL_PATH, model_version_name, MODEL_FILE_NAME)
    PARAMS_PATH = os.path.join(BASE_MODEL_PATH, model_version_name, PARAMS_FILE_NAME)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PARAMS_PATH):
        st.error(f"오류: 선택된 모델 버전 '{model_version_name}'에서 모델 파일 또는 매개변수 파일이 발견되지 않았습니다.")
        st.info("DNN 모델 학습 및 저장 페이지에서 모델을 먼저 학습하고 저장해주세요.")
        return ERROR_RETURN

    try:
        # Keras 모델 로드 (Normalization Layer 포함)
        model = load_model(MODEL_PATH, compile=False)
        
        # 정규화 매개변수 로드
        params = np.load(PARAMS_PATH)
        Y_train_mean = params['Y_train_mean'].item()
        Y_train_std = params['Y_train_std'].item()
        
        train_size = params['train_size'].item()
        test_size = params['test_size'].item()
        val_size = params.get('val_size', 0).item()
        
        st.success(f"모델 버전 '{model_version_name}' 로드 완료. (총 학습 데이터 포인트: {train_size + val_size + test_size})")
        
        return model, Y_train_mean, Y_train_std, (train_size, test_size, val_size)
        
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return ERROR_RETURN

def to_excel_download_link(df: pd.DataFrame, file_name: str) -> BytesIO:
    """DataFrame을 Excel(xlsx) 파일로 변환하고 BytesIO 객체를 반환합니다."""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Predicted_EEPF')
    # writer 객체를 저장합니다.
    writer.close()
    output.seek(0)
    return output

def run_dnn_prediction_page():
    """
    Streamlit 페이지: 저장된 DNN 모델을 로드하여 EEPF를 추론합니다.
    """
    st.title("DNN 모델 로드 및 EEPF 추론")
    st.write("학습된 모델을 사용하여 입력 조건에 해당하는 EEPF 스펙트럼을 예측합니다.")

    # 1. 사용 가능한 모델 버전 목록 가져오기
    available_models = get_available_models()

    if not available_models:
        st.error("오류: 'model' 폴더에 학습된 모델 버전이 없습니다.")
        st.info("DNN 모델 학습 및 저장 페이지에서 모델을 먼저 학습하고 저장해주세요.")
        return
        
    # 2. 모델 선택 UI
    st.subheader("모델 버전 선택")
    
    # Select box를 통해 사용자가 모델 버전을 선택하도록 합니다.
    selected_model_version = st.selectbox(
        "사용할 모델 버전(하이퍼파라미터 폴더)을 선택하세요:",
        available_models,
        help="폴더 이름은 학습 당시의 타임스탬프와 하이퍼파라미터 정보를 포함하며, 가장 최근 모델이 상단에 표시됩니다."
    )
    st.markdown("---")

    # 3. 모델 로드 (캐싱 적용)
    # 선택된 모델 버전을 기준으로 로드 함수 호출
    model, Y_train_mean, Y_train_std, sizes = load_model_for_prediction(selected_model_version)

    if model is None:
        st.stop()
        
    train_size, test_size, val_size = sizes
    st.markdown(f"**로드된 모델 요약:** 버전 **`{selected_model_version}`**")
    st.markdown(f"총 데이터 포인트: **{train_size + val_size + test_size}** (Train: {train_size}, Validation: {val_size}, Test: {test_size})")
    st.markdown("---")
    
    st.subheader("EEPF 예측 입력값")
    st.write("아래 입력값을 변경하고 버튼을 누르면, $e V$ 0부터 $17.01$까지의 EEPF 스펙트럼이 추론됩니다.")

    # User input for prediction
    col1, col2 = st.columns(2)
    with col1:
        pressure_input = st.number_input("압력 (pressure)", min_value=0.1, value=5.0, step=0.1)
        power_input = st.number_input("파워 (power)", min_value=1.0, value=110.0, step=1.0)
    with col2:
        ne_input = st.number_input("플라즈마 밀도 ($N_e$)", min_value=1e9, value=1.5e10, step=1e9, format="%.1e")
        st.markdown("<br>", unsafe_allow_html=True) # 공간 확보

    if st.button("EEPF 추론 실행", type="primary"):
        with st.spinner('EEPF 추론 중...'):
            # Generate eV values
            ev_values = np.arange(EV_MIN, EV_MAX + EV_STEP, EV_STEP)
            
            # Create a 2D array of inputs for batch prediction
            custom_inputs = np.column_stack((
                np.full(ev_values.shape, pressure_input),
                np.full(ev_values.shape, power_input),
                np.full(ev_values.shape, ne_input),
                ev_values
            ))

            # Perform batch prediction (모델에 Normalization Layer가 포함되어 있어 정규화 불필요)
            predicted_norms = model.predict(custom_inputs, verbose=0).flatten()
            
            # Denormalize the predicted EEPF values
            predicted_eepfs = predicted_norms * Y_train_std + Y_train_mean
            
            # 예측 결과를 DataFrame으로 변환
            predicted_df = pd.DataFrame({
                'Energy (eV)': ev_values,
                'Predicted EEPF': predicted_eepfs,
                'Pressure': pressure_input,
                'Power': power_input,
                'Ne': ne_input
            })

        st.success("EEPF 추론 완료")
        
        # Plot the predicted EEPF spectrum
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ev_values, predicted_eepfs, label='Predicted EEPF', color='blue')
        ax.set_yscale("log")
        ax.set_xlabel("Energy [eV]", fontsize=12)
        ax.set_ylabel(r"EEPF [eV$^{-3/2}$ cm$^{-3}$]", fontsize=12)
        
        ne_formatted = f"{ne_input:.2e}"
        ax.set_title(f"Predicted EEPF Spectrum (Pressure={pressure_input:.1f}, Power={power_input:.1f}, $N_e$={ne_formatted})", fontsize=14)
        ax.grid(True, which="both", ls="--")
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Excel 다운로드 버튼
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file_name = f"EEPF_Prediction_P{pressure_input:.1f}_W{power_input:.1f}_{timestamp}.xlsx"
        
        excel_data = to_excel_download_link(predicted_df, excel_file_name)
        
        st.download_button(
            label="📊 Export (Excel 다운로드)",
            data=excel_data,
            file_name=excel_file_name,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            help="추론된 EEPF 데이터를 엑셀 파일로 다운로드합니다."
        )
