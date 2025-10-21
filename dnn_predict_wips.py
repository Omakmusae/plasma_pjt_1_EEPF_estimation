import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import os 
from tensorflow.keras.models import load_model
from datetime import datetime
from io import BytesIO 
from pathlib import Path
from typing import Dict, Tuple

# --- 고정 변수 설정 ---
BASE_MODEL_PATH = Path("model") 
TARGETS = ['EEPF', 'Ne', 'Te']
# [8개 Feature] pressure, power, ion_flux, i_1w, i_2w, Np, wips_Te, eV
FEATURES = ['pressure', 'power', 'ion_flux', 'i_1w', 'i_2w', 'Np', 'wips_Te', 'eV'] 

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
    """
    if not os.path.exists(BASE_MODEL_PATH):
        return []
    
    # 최신 폴더가 위에 오도록 정렬
    model_versions = sorted([
        d for d in os.listdir(BASE_MODEL_PATH) 
        if os.path.isdir(os.path.join(BASE_MODEL_PATH, d))
    ], reverse=True)
    
    return model_versions

@st.cache_resource(show_spinner="저장된 모델 및 매개변수 로드 중...")
def load_model_for_prediction(model_version_name: str) -> Tuple[tf.keras.Model | None, Dict | None, Dict | None, Tuple[int, int, int]]:
    """
    선택된 Multi-Output 모델 버전의 모델 파일과 정규화 매개변수를 로드합니다.
    """
    # 로드 실패 시 반환할 None 튜플 (Unpacking 에러 방지)
    ERROR_RETURN: Tuple[None, None, None, Tuple[int, int, int]] = (None, None, None, (0, 0, 0)) 

    MODEL_PATH = os.path.join(BASE_MODEL_PATH, model_version_name, MODEL_FILE_NAME)
    PARAMS_PATH = os.path.join(BASE_MODEL_PATH, model_version_name, PARAMS_FILE_NAME)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PARAMS_PATH):
        st.error(f"오류: 선택된 모델 버전 '{model_version_name}'에서 모델 파일 또는 매개변수 파일이 발견되지 않았습니다.")
        st.info("DNN 모델 학습 및 저장 페이지에서 모델을 먼저 학습하고 저장해주세요.")
        return ERROR_RETURN

    try:
        # Keras 모델 로드
        model = load_model(MODEL_PATH, compile=False)
        
        # 정규화 매개변수 로드
        params = np.load(PARAMS_PATH)
        # Multi-Output이므로 배열로 로드된 mean/std를 다시 딕셔너리로 매핑
        Y_train_mean_array = params['Y_train_mean']
        Y_train_std_array = params['Y_train_std']
        
        # Target 이름 순서에 맞게 딕셔너리로 재구성
        Y_train_mean = {TARGETS[i]: Y_train_mean_array[i].item() for i in range(len(TARGETS))}
        Y_train_std = {TARGETS[i]: Y_train_std_array[i].item() for i in range(len(TARGETS))}
        
        # 데이터 크기 로드
        train_size = params['train_size'].item()
        test_size = params['test_size'].item()
        val_size = params.get('val_size', 0).item()
        
        st.success(f"Multi-Output 모델 버전 '{model_version_name}' 로드 완료. (총 학습 데이터 포인트: {train_size + val_size + test_size})")
        
        return model, Y_train_mean, Y_train_std, (train_size, test_size, val_size)
        
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return ERROR_RETURN

def to_excel_download_link(df: pd.DataFrame, file_name: str) -> BytesIO:
    """DataFrame을 Excel(xlsx) 파일로 변환하고 BytesIO 객체를 반환합니다."""
    output = BytesIO()
    # 'writer.close()' 대신 'with' 구문을 사용하면 더 안전합니다.
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicted_Results') 
    output.seek(0)
    return output

def run_dnn_prediction_wips_page():
    """
    Streamlit 페이지: 저장된 DNN 모델을 로드하여 EEPF, Ne, Te를 추론합니다.
    """
    st.set_page_config(layout="wide", page_title="EEPF Multi-Output DNN 예측기")
    st.title("DNN 모델 로드 및 Multi-Output 추론")
    st.write("학습된 8-특성 모델을 사용하여 입력 조건에 해당하는 EEPF, Ne, Te를 예측합니다.")

    available_models = get_available_models()

    if not available_models:
        st.error("오류: 'model' 폴더에 학습된 모델 버전이 없습니다.")
        st.info("DNN 모델 학습 및 저장 페이지에서 모델을 먼저 학습하고 저장해주세요.")
        return
        
    # 모델 선택 UI
    st.subheader("모델 버전 선택")
    selected_model_version = st.selectbox(
        "사용할 모델 버전(하이퍼파라미터 폴더)을 선택하세요:",
        available_models,
        help="가장 최근 모델이 상단에 표시됩니다."
    )
    st.markdown("---")

    # 모델 로드 (캐싱 적용)
    model, Y_train_mean, Y_train_std, sizes = load_model_for_prediction(selected_model_version)

    if model is None:
        st.stop()
        
    train_size, test_size, val_size = sizes
    st.markdown(f"**로드된 모델 요약:** 버전 **`{selected_model_version}`**")
    st.markdown(f"총 데이터 포인트: **{train_size + val_size + test_size}** (Train: {train_size}, Validation: {val_size}, Test: {test_size})")
    st.markdown("---")
    
    st.subheader("8-특성 Multi-Output 예측 입력값")
    st.write("아래 **7가지 입력값**($\text{pressure, power, ion\_flux, i\_1w, i\_2w, Np, wips\_Te}$)과 $\text{eV}$를 기반으로 $\text{EEPF, Ne, Te}$가 추론됩니다.")

    # [수정된 부분] 7개의 고정 입력 Feature 모두 받기 (wips_Te 추가)
    col1, col2, col3 = st.columns(3)
    with col1:
        pressure_input = st.number_input("1. 압력 (pressure, mTorr)", value=50.0, step=1.0)
        power_input = st.number_input("2. 파워 (power, W)", value=500.0, step=10.0)
    with col2:
        ion_flux_input = st.number_input("3. 이온 플럭스 (ion_flux, #/m^2s)", value=5.0e16, step=1.0e15, format="%.2e")
        i_1w_input = st.number_input("4. i_1w (A)", value=5.0, step=0.1, format="%.1f")
    with col3:
        i_2w_input = st.number_input("5. i_2w (A)", value=5.0, step=0.1, format="%.1f")
        np_input = st.number_input("6. Np (WIPS 전자 밀도, #/m^3)", value=5.0e17, step=1e16, format="%.2e")
    
    # 누락된 7번째 고정 입력값 (wips_Te)을 별도 섹션에 추가
    st.markdown("---")
    wips_Te_input = st.number_input("7. wips_Te (WIPS 전자 온도, eV)", min_value=0.1, value=3.5, step=0.1, format="%.1f", help="이 값은 8개 특성 중 하나로 사용됩니다.")
    st.markdown("---")

    if st.button("Multi-Output 추론 실행 (EEPF, Ne, Te)", type="primary"):
        with st.spinner('Multi-Output 추론 중...'):
            # 1. eV 스펙트럼 생성
            ev_values = np.arange(EV_MIN, EV_MAX + EV_STEP, EV_STEP)
            
            # 2. 8개 Feature 순서에 맞게 2D array 생성
            # FEATURES = ['pressure', 'power', 'ion_flux', 'i_1w', 'i_2w', 'Np', 'wips_Te', 'eV'] 
            custom_inputs = np.column_stack((
                np.full(ev_values.shape, pressure_input), 
                np.full(ev_values.shape, power_input), 
                np.full(ev_values.shape, ion_flux_input),
                np.full(ev_values.shape, i_1w_input), 
                np.full(ev_values.shape, i_2w_input), 
                np.full(ev_values.shape, np_input), 
                np.full(ev_values.shape, wips_Te_input), # <--- 7번째 고정 Feature 추가 완료
                ev_values                              # <--- 8번째 변수 Feature (eV)
            ))

            # Perform batch prediction (출력: [N, 3] 형태의 정규화된 EEPF, Ne, Te)
            predicted_norms = model.predict(custom_inputs, verbose=0) # Shape: (N, 3)
            
            # 3. Denormalize the predicted values for all 3 targets
            predicted_eepfs = predicted_norms[:, 0] * Y_train_std['EEPF'] + Y_train_mean['EEPF']
            predicted_nes = predicted_norms[:, 1] * Y_train_std['Ne'] + Y_train_mean['Ne']
            predicted_tes = predicted_norms[:, 2] * Y_train_std['Te'] + Y_train_mean['Te']
            
            # Ne, Te는 모든 eV 포인트에서 동일해야 하므로, 첫 번째 값만 대표값으로 사용
            predicted_ne_avg = predicted_nes[0] 
            predicted_te_avg = predicted_tes[0]

            # 4. 예측 결과를 DataFrame으로 변환 (다운로드용)
            predicted_df = pd.DataFrame({
                'Energy (eV)': ev_values,
                'Predicted EEPF': predicted_eepfs,
                'Predicted Ne': predicted_nes, 
                'Predicted Te': predicted_tes,
                'Input_Pressure': pressure_input,
                'Input_Power': power_input,
                'Input_ion_flux': ion_flux_input,
                'Input_i_1w': i_1w_input,
                'Input_i_2w': i_2w_input,
                'Input_Np': np_input,
                'Input_wips_Te': wips_Te_input # 입력값도 다운로드 데이터에 포함
            })
        
        # 5. 결과 출력
        st.success("Multi-Output 추론 완료")
        
        col_metrics, col_plot = st.columns([1, 2])
        
        with col_metrics:
            # 추론된 Ne, Te 결과 요약 출력
            st.subheader("추론 결과 요약")
            ne_formatted = f"{predicted_ne_avg:.2e}"
            te_formatted = f"{predicted_te_avg:.3f}"
            
            st.metric(
                label="전자 밀도 (Predicted $N_e$)", 
                value=f"{ne_formatted} #/m³",
                delta_color="off"
            )
            st.metric(
                label="전자 온도 (Predicted $T_e$)", 
                value=f"{te_formatted} eV",
                delta_color="off"
            )
            st.markdown("---")
            st.info(f"입력된 WIPS $T_e$: **{wips_Te_input:.1f} eV**")
            st.info(f"추론된 Langmuir Probe $T_e$: **{te_formatted} eV**")
            
            # 엑셀 다운로드 버튼
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file_name = f"MultiOut_Pred_P{pressure_input:.0f}W{power_input:.0f}_{timestamp}.xlsx"
            
            excel_data = to_excel_download_link(predicted_df, excel_file_name)
            
            st.download_button(
                label="📊 Export (Excel 다운로드)",
                data=excel_data,
                file_name=excel_file_name,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                help="추론된 EEPF, Ne, Te 데이터를 엑셀 파일로 다운로드합니다."
            )


        with col_plot:
            # Plot the predicted EEPF spectrum
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(ev_values, predicted_eepfs, label='Predicted EEPF', color='blue')
            ax.set_yscale("log")
            ax.set_xlabel("Energy [eV]", fontsize=12)
            ax.set_ylabel(r"EEPF [eV$^{-3/2}$ cm$^{-3}$]", fontsize=12)
            
            ax.set_title(f"Predicted EEPF Spectrum (P={pressure_input:.0f}, W={power_input:.0f}, $N_e$={ne_formatted}, $T_e$={te_formatted})", fontsize=14)
            ax.grid(True, which="both", ls="--")
            ax.legend()
            st.pyplot(fig)
            
        st.markdown("---")
        st.subheader("추론된 EEPF 데이터 테이블 (일부)")
        st.dataframe(predicted_df[['Energy (eV)', 'Predicted EEPF']].head(20).style.format({'Energy (eV)': "{:.3f}", 'Predicted EEPF': "{:.4e}"}))

