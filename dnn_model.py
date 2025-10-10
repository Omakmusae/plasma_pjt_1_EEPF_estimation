import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st
import sqlite3
import json
import os # 경로 처리를 위해 os 모듈 추가
from tensorflow.keras.models import load_model
from datetime import datetime
from io import BytesIO # Excel 다운로드를 위해 BytesIO 임포트


# 모델 및 매개변수 파일 경로 설정
MODEL_PATH = "eepf_dnn_model.h5"
PARAMS_PATH = "eepf_norm_params.npz"

HISTORY_PATH = "eepf_history.json"

EV_MIN = 0.0
EV_MAX = 17.01
EV_STEP = 0.045
TEST_SIZE_RATIO = 0.15
VALIDATION_SIZE_RATIO = 0.15 # 전체 데이터셋 대비 최종 비율

# EEPF 데이터베이스에서 데이터를 로드하고 전처리하는 함수
#@st.cache_resource
def load_and_train_model():
    """
    DB에서 데이터를 로드, DNN 모델을 학습하고, 학습된 모델과 정규화 매개변수를 반환합니다.
    (1) 저장된 모델 파일이 있으면 로드하고, 없으면 DB에서 로드 후 학습 및 저장합니다.
    반환 값 순서: model, history, Y_train_mean, Y_train_std, len(X_train), len(X_test) (총 6개)
    """
    # 에러 발생 시 반환할 6개의 None 튜플 (Unpacking 에러 방지)
    ERROR_RETURN = (None, None, None, None, None, None)

    # -----------------------------
    # 1. Saved Model Load (빠른 로딩)
    # -----------------------------
    if os.path.exists(MODEL_PATH) and os.path.exists(PARAMS_PATH):
        with st.spinner("기존 학습된 모델 파일을 로드합니다..."):
            
            try:
                # 모델 로드 성공 시 터미널 출력
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 기존 학습된 모델을 로드했습니다. (학습 건너뛰기)")
        
                # Keras 모델 로드
                model = load_model(MODEL_PATH, compile=False)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                              loss='mean_squared_error',
                              metrics=['mean_absolute_error'])
                params = np.load(PARAMS_PATH)
                Y_train_mean = params['Y_train_mean'].item()
                Y_train_std = params['Y_train_std'].item()
                train_size = params['train_size'].item()
                test_size = params['test_size'].item()

                # ✅ 학습 이력 (History) 로드
                history_data = {}
                if os.path.exists(HISTORY_PATH):
                    with open(HISTORY_PATH, 'r') as f:
                        history_data = json.load(f)
                    
                    # 로드된 딕셔너리를 Keras History 객체처럼 사용할 수 있도록 구조화
                    history = tf.keras.callbacks.History()
                    history.history = history_data 

                st.success(f"저장된 모델 및 매개변수를 로드했습니다. (학습 건너뛰기)")
                # 학습 이력(history)이 로드되지 않았을 경우 history_data만 빈 상태로 반환됨
                return model, history, Y_train_mean, Y_train_std, train_size, test_size
                
            except Exception as e:
                st.error(f"저장된 모델/매개변수 로드 중 오류 발생: {e}. DB에서 데이터를 로드하여 재학습을 시도합니다.")
                # 오류 발생 시 아래 DB 로드 및 학습 로직을 실행
    
    # DB 파일 경로 설정: 스크립트가 excel_parsing 폴더 안에 있고, DB 파일은 상위 폴더에 있음
    db_file_path = os.path.join(".\EEPF_estimation.db")

    # -----------------------------
    # 2. DB Data Loading and Transformation (모델 파일이 없거나 로드 실패 시)
    # -----------------------------
    df_final = pd.DataFrame()
    
    # 데이터베이스 연결
    try:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        
        # eepf_graph 테이블과 eepf_data 테이블을 조인하여 데이터 로드
        cursor.execute(
            """
            SELECT
                G.id,
                G.pressure,
                G.power,
                G.eepf_json,
                D.Np 
            FROM 
                eepf_graph AS G
            INNER JOIN 
                eepf_data AS D
            ON
                G.pressure = D.pressure AND G.power = D.power
            """
            # WHERE G.pressure = 5 OR G.pressure = 10 # 이 조건은 주석 처리 또는 제거하여 유연하게 사용
            )
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            st.error(f"오류: 데이터베이스 '{db_file_path}'에서 조건에 맞는 데이터가 발견되지 않았습니다.")
            return ERROR_RETURN

        all_data = []
        for eepf_id, pressure, power, eepf_json_str, Ne_value in records:
            try:
                if Ne_value is None or not isinstance(Ne_value, (float, int)):
                    st.warning(f"경고: pressure={pressure}, power={power} 조건의 Plasma Density(Np) 값이 유효하지 않습니다. 이 레코드를 건너킵니다.")
                    continue
                
                eepf_data = json.loads(eepf_json_str)
                eV_list = eepf_data.get('eV', [])
                EEPF_list = eepf_data.get('EEPF', [])
                
                if len(eV_list) == len(EEPF_list) and len(eV_list) > 0:
                    for eV, EEPF in zip(eV_list, EEPF_list):
                        all_data.append({
                            'eepf_id': eepf_id,
                            'pressure': pressure,
                            'power': power,
                            'Ne': Ne_value, 
                            'eV': eV,
                            'EEPF': EEPF
                        })

            except json.JSONDecodeError:
                st.error(f"오류: pressure={pressure}, power={power} 조건의 eepf_json 데이터 파싱에 실패했습니다.")
                continue
            except Exception as inner_e:
                st.error(f"데이터 변환 중 알 수 없는 오류 발생 (pressure={pressure}, power={power}): {inner_e}")
                continue


        if not all_data:
            st.error("오류: 데이터베이스에서 유효한 EEPF 데이터를 추출하지 못했습니다.")
            return ERROR_RETURN
            
        df_final = pd.DataFrame(all_data)
        
        # ### 디버깅 코드 시작: 특정 (P, W) 조합의 샘플 수 확인 ###
        # 확인하고 싶은 Pressure (P_test)와 Power (W_test)를 설정하세요.
        P_test = 5.0  # 예시 값
        W_test = 100.0 # 예시 값
        
        # 1. 특정 조합의 샘플 수 확인
        test_samples = df_final[
            (df_final['pressure'] == P_test) & 
            (df_final['power'] == W_test)
        ]
        test_count = 0

        for eepf_id, pressure, power, eepf_json_str, Ne_value in records:
            if pressure == P_test and power == W_test:
                test_count += 1
        print(f"[디버그] DB 조인 결과 (records)에서 P={P_test}, W={W_test} 행의 개수: {test_count}개")
        # 2. 결과 출력 (터미널 또는 Streamlit)
        print(f"\n[디버그] P={P_test}, W={W_test} 조건의 df_final 샘플 수: {len(test_samples)}개")
        # st.info(f"[디버그] P={P_test}, W={W_test} 조건의 df_final 샘플 수: {len(test_samples)}개")
        
        # 3. 샘플의 Np 값 확인 (Np가 1개인지 확인)
        unique_np = test_samples['Ne'].unique() 
        print(f"[디버그] 해당 조건의 고유 Np(Ne) 값: {unique_np.tolist()}")
        # ### 디버깅 코드 끝 ###

        for col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce') 
        
        df_final.dropna(subset=['pressure', 'power', 'Ne', 'eV', 'EEPF'], inplace=True)
        
    except sqlite3.Error as e:
        st.error(f"SQLite 오류 발생: {e}. DB 파일 경로('{db_file_path}')를 확인해주세요.")
        return ERROR_RETURN
    except Exception as e:
        st.error(f"데이터 로드 및 변환 중 일반 오류 발생: {e}")
        return ERROR_RETURN
        
    if df_final.empty:
        st.error("오류: 모든 데이터를 로드하고 전처리한 후 유효한 데이터가 남아있지 않아 학습을 진행할 수 없습니다.")
        return ERROR_RETURN
        
    st.success(f"DB 로드 완료: 총 {len(df_final)}개 데이터 포인트 확보")

    # -----------------------------
    # 3. Model Training
    # -----------------------------

    GROUP_KEYS = ['pressure', 'power', 'eepf_id']
    FEATURES = ['pressure', 'power', 'Ne', 'eV'] # X_data의 컬럼과 일치
    TARGET = 'EEPF' # Y_data의 컬럼과 일치
    # 1) 고유 그룹 식별: (pressure, power, eepf_id) 조합을 그룹으로 인식
    groups = df_final[GROUP_KEYS].drop_duplicates().reset_index(drop=True)
    
    # 2) 고유 그룹을 Train/Test 그룹으로 무작위 분할 (test_size=0.2)
    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=TEST_SIZE_RATIO, # 0.15
        random_state=42,
        # Pressure를 기준으로 분할하기 위해 'pressure'를 기준으로 계층적 샘플링
        stratify=groups['pressure'] 
    )

    # Train (70%) 그룹과 Validation (15%) 그룹 분리
    # train_val_groups에서 Validation이 전체의 15%가 되려면
    # 15% / 85% = 약 0.17647 의 비율로 분할해야 함.
    val_split_ratio = VALIDATION_SIZE_RATIO / (1 - TEST_SIZE_RATIO) # 0.15 / 0.85 ≈ 0.17647
    
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_split_ratio,
        random_state=42,
        # Pressure를 기준으로 분할하기 위해 'pressure'를 기준으로 계층적 샘플링
        stratify=train_val_groups['pressure']
    )

    # 요구사항 1. Train, Validation, Test 그룹 정보 추출 (리스트로 변환)
    train_groups_list = train_groups.to_dict('records')
    val_groups_list = val_groups.to_dict('records')
    test_groups_list = test_groups.to_dict('records')

    # 3) 분할된 그룹에 해당하는 행 추출
    X_train_df = df_final.merge(train_groups, on=GROUP_KEYS, how='inner')
    X_val_df = df_final.merge(val_groups, on=GROUP_KEYS, how='inner')
    X_test_df = df_final.merge(test_groups, on=GROUP_KEYS, how='inner')
    
    # 4) 특징(X)과 타겟(Y) 분리
    X_train = X_train_df[FEATURES]
    Y_train = X_train_df[TARGET]
    X_val = X_val_df[FEATURES]
    Y_val = X_val_df[TARGET]
    X_test = X_test_df[FEATURES]
    Y_test = X_test_df[TARGET] 

    train_size = len(X_train)
    val_size = len(X_val) # 추가된 검증 데이터 크기
    test_size = len(X_test)

    # 디버깅 정보 출력
    st.info(f"데이터 그룹 분할 (Pressure 기반):")
    st.info(f"  총 그룹 수: {len(groups)}")
    st.info(f"  훈련 그룹 수: {len(train_groups)}, 데이터 수: {train_size} ({train_size/len(df_final)*100:.1f}%)")
    st.info(f"  검증 그룹 수: {len(val_groups)}, 데이터 수: {val_size} ({val_size/len(df_final)*100:.1f}%)")
    st.info(f"  테스트 그룹 수: {len(test_groups)}, 데이터 수: {test_size} ({test_size/len(df_final)*100:.1f}%)")
    
    # Data Normalization
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train)) 

    Y_train_mean = Y_train.mean()
    Y_train_std = Y_train.std()

    Y_train_norm = (Y_train - Y_train_mean) / Y_train_std
    Y_val_norm = (Y_val - Y_train_mean) / Y_train_std # 검증 데이터도 정규화
    
    # Y_test 정규화
    Y_test_norm = (Y_test - Y_train_mean) / Y_train_std

    # Build Model
    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    
    X_train_array = np.array(X_train)
    X_val_array = np.array(X_val) # 검증 데이터 배열 준비
    # X_test 배열 준비
    X_test_array = np.array(X_test)
    start_time = datetime.now()

    # Train Model
    with st.spinner('모델 학습 중... 잠시만 기다려주세요.'):
        print(f"\n--- 모델 학습 시작 ---")
        print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        history = model.fit(
            X_train_array,
            Y_train_norm,
            epochs=100,
            # Validation Split 대신 X_val, Y_val을 명시적으로 사용
            validation_data=(X_val_array, Y_val_norm), 
            verbose=0
        )
        end_time = datetime.now()
        training_duration = end_time - start_time
        # 터미널에 출력
        print(f"완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 소요 시간: {training_duration}")
        print(f"--- 모델 학습 완료 ---\n")
    
    # Test Data 평가 및 History에 추가
    with st.spinner('테스트 데이터 성능 평가 중...'):
        test_loss, test_mae = model.evaluate(X_test_array, Y_test_norm, verbose=0)
        st.success(f"테스트 결과 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")
    
    # Test 결과를 history.history 딕셔너리에 추가 (그래프 출력을 위해 epoch 수만큼 반복)
    history_data = history.history.copy()
    history_data['test_loss'] = [test_loss] * len(history.epoch)
    history_data['test_mean_absolute_error'] = [test_mae] * len(history.epoch)

    # ✅ 요구사항 2. Train/Validation/Test 데이터셋 그룹 정보를 history_data에 추가
    history_data['train_groups'] = train_groups_list
    history_data['val_groups'] = val_groups_list
    history_data['test_groups'] = test_groups_list

    # -----------------------------
    # 4. Save Model and Parameters (학습 후 저장)
    # -----------------------------
    try:
        with st.spinner('학습된 모델 저장 중...'):
            # 모델 전체 저장 (Normalization Layer 포함)
            model.save(MODEL_PATH)

            # 정규화 매개변수 및 데이터 크기 저장 (train/val/test 크기 저장)
            np.savez(PARAMS_PATH, 
                      Y_train_mean=Y_train_mean, 
                      Y_train_std=Y_train_std, 
                      train_size=train_size, 
                      test_size=test_size,
                      val_size=val_size)
            
            # history_data (Test 지표 및 그룹 정보 포함) 저장
            with open(HISTORY_PATH, 'w') as f:
                # pandas DataFrame의 to_dict('records')는 JSON으로 바로 저장이 가능합니다.
                json.dump(history_data, f, indent=4) # indent=4 추가로 가독성 향상

            st.success(f"새롭게 학습된 모델과 매개변수가 '{MODEL_PATH}'와 '{PARAMS_PATH}'에 저장되었습니다.")
    except Exception as e:
           st.error(f"모델 저장 중 오류 발생: {e}")
    
    # history 객체의 history 속성을 업데이트된 데이터로 변경
    history.history = history_data

    return model, history, Y_train_mean, Y_train_std, train_size, test_size


def run_dnn_model_page():
    """
    Streamlit page for DNN model training and prediction.
    """
    
    # --- 엑셀 파일로 변환하는 헬퍼 함수 ---
    def to_excel_download_link(df: pd.DataFrame, file_name: str) -> BytesIO:
        """DataFrame을 Excel(xlsx) 파일로 변환하고 BytesIO 객체를 반환합니다."""
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        # 데이터프레임을 Excel 시트에 씁니다.
        df.to_excel(writer, index=False, sheet_name='Predicted_EEPF')
        # writer 객체를 저장합니다.
        # save()를 호출해야만 BytesIO 객체에 내용이 기록됩니다.
        writer.close()
        # 파일 포인터를 처음으로 되돌립니다.
        output.seek(0)
        return output
    # -----------------------------------
    
    st.title("DNN 모델 학습 및 EEPF 예측")
    st.write("딥러닝 모델을 로드/학습하고, 새로운 입력값에 대한 EEPF 값을 예측합니다.")

    # Load and train the model using caching
    model, history, Y_train_mean, Y_train_std, train_size, test_size = load_and_train_model()

    if model is None:
        st.stop()
        
    
    st.markdown("---")
    st.subheader("모델 학습 정보")
    st.write(f"총 데이터 포인트 수: {train_size + test_size}")
    st.write(f"학습 데이터 수: {train_size}")
    st.write(f"테스트 데이터 수: {test_size}")

    # Plot the training history
    def plot_loss(history):
        # Test 결과가 history.history에 포함되어 있는지 확인합니다.
        test_loss_available = 'test_loss' in history.history and history.history['test_loss']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        
        if test_loss_available:
            # Test Loss는 단일 값이므로 수평선으로 표시
            test_loss_value = history.history['test_loss'][0] 
            ax.axhline(y=test_loss_value, color='r', linestyle='--', label=f'Test Loss ({test_loss_value:.4f})')
        
        ax.set_title('Training, Validation, and Test Loss Over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    def plot_mae(history):
        # Test 결과가 history.history에 포함되어 있는지 확인합니다.
        test_mae_available = 'test_mean_absolute_error' in history.history and history.history['test_mean_absolute_error']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['mean_absolute_error'], label='Training MAE')
        ax.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        
        if test_mae_available:
            # Test MAE는 단일 값이므로 수평선으로 표시
            test_mae_value = history.history['test_mean_absolute_error'][0]
            ax.axhline(y=test_mae_value, color='r', linestyle='--', label=f'Test MAE ({test_mae_value:.4f})')
        
        ax.set_title('Training, Validation, and Test MAE Over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean Absolute Error')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # 저장된 모델 로드 시 history가 None이므로, 학습된 경우에만 그래프 출력
    if history is not None:
        # 모델 학습 결과 확인 (클릭)
        with st.expander("모델 학습 결과 확인 (클릭)", expanded=False):
            plot_loss(history)
            plot_mae(history)
            
        # 데이터 셋 현황 확인 (클릭) 추가
        with st.expander("데이터 셋 현황 확인 (클릭)", expanded=False):
            if 'train_groups' in history.history and history.history['train_groups']:
                
                # Train/Validation/Test 그룹 정보를 DataFrame으로 변환
                df_train_groups = pd.DataFrame(history.history['train_groups']).set_index('pressure')
                df_val_groups = pd.DataFrame(history.history['val_groups']).set_index('pressure')
                df_test_groups = pd.DataFrame(history.history['test_groups']).set_index('pressure')
                
                st.markdown("**1. 훈련 데이터셋 (Train Groups) - 학습에 사용**")
                st.write(df_train_groups)
                
                st.markdown("**2. 검증 데이터셋 (Validation Groups) - 학습 중 성능 모니터링에 사용**")
                st.write(df_val_groups)
                
                st.markdown("**3. 테스트 데이터셋 (Test Groups) - 최종 모델 성능 평가에 사용**")
                st.write(df_test_groups)
                
                st.info("표의 각 행은 특정 Power 및 Pressure 조건의 EEPF 그래프 데이터를 나타냅니다.")

            else:
                st.info("데이터셋 분할 정보가 학습 이력 파일에 기록되어 있지 않습니다. 모델을 재학습해야 합니다.")
    else:
        st.info("모델이 저장된 파일에서 로드되었으므로, 학습 이력(History) 및 데이터셋 정보가 제공되지 않습니다.")

    st.markdown("---")
    st.subheader("EEPF 예측하기")
    st.write("아래 입력값을 변경하고 버튼을 누르면, eV 0부터 17.01까지의 EEPF 스펙트럼이 추론됩니다.")

    # User input for prediction
    col1, col2 = st.columns(2)
    with col1:
        pressure_input = st.number_input("압력 (pressure)", min_value=0.1, value=5.0, step=0.1)
        power_input = st.number_input("파워 (power)", min_value=1.0, value=110.0, step=1.0)
    with col2:
        ne_input = st.number_input("플라즈마 밀도 (Ne)", min_value=1e9, value=1.5e10, step=1e9, format="%.1e")

    if st.button("EEPF 추론 실행"):
        with st.spinner('EEPF 추론 중...'):
            # Generate eV values from 0 to 17.01 with a step of 0.045
            ev_values = np.arange(EV_MIN, EV_MAX + EV_STEP, EV_STEP)
            
            # Create a 2D array of inputs for batch prediction
            custom_inputs = np.column_stack((
                np.full(ev_values.shape, pressure_input),
                np.full(ev_values.shape, power_input),
                np.full(ev_values.shape, ne_input),
                ev_values
            ))

            # Perform batch prediction
            # Keras 모델은 Normalization Layer를 포함하고 있으므로, 정규화되지 않은 입력값 사용
            predicted_norms = model.predict(custom_inputs).flatten()
            
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
        ax.set_title(f"Predicted EEPF Spectrum (Pressure={pressure_input:.1f}, Power={power_input:.1f}, Ne={ne_input:.2e})", fontsize=14)
        ax.grid(True, which="both", ls="--")
        ax.legend()
        st.pyplot(fig)
        
        st.write("---")
        
        # Excel 다운로드 버튼
        
        # 파일 이름 정의
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file_name = f"EEPF_Prediction_P{pressure_input:.1f}_W{power_input:.1f}_{timestamp}.xlsx"
        
        # Excel 파일 내용 생성
        excel_data = to_excel_download_link(predicted_df, excel_file_name)
        
        # Streamlit 다운로드 버튼 표시
        st.download_button(
            label="📊 Export (Excel 다운로드)",
            data=excel_data,
            file_name=excel_file_name,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            help="추론된 EEPF 데이터를 엑셀 파일로 다운로드합니다."
        )
        
        st.write("---")