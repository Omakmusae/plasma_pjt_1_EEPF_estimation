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

# 모델 및 매개변수 파일 경로 설정
MODEL_PATH = "eepf_dnn_model.h5"
PARAMS_PATH = "eepf_norm_params.npz"

HISTORY_PATH = "eepf_history.json"

EV_MIN = 0.0
EV_MAX = 17.01
EV_STEP = 0.045
TEST_SIZE_RATIO = 0.15
VALIDATION_SIZE_RATIO = 0.15 # 전체 데이터셋 대비 최종 비율


@st.cache_resource
def load_and_train_sample_model(file_path):
    """
    Load data, train the DNN model, and return the trained model and normalization parameters.
    This function is cached to prevent retraining on every user interaction.
    """
    # 1. Data parsing and concatenation
    try:
        df_5_100_raw = pd.read_excel(file_path, sheet_name='5_100', header=None, usecols='A:C,E:F,H:I')
        input_vals1_100 = df_5_100_raw.iloc[1, 0:3].values
        output_df1_100 = df_5_100_raw.iloc[1:, [3, 4]].copy()
        output_df1_100.columns = ['eV', 'EEPF']
        output_df1_100.dropna(inplace=True)
        output_df1_100['pressure'] = input_vals1_100[0]
        output_df1_100['power'] = input_vals1_100[1]
        output_df1_100['Ne'] = input_vals1_100[2]
        df_5_100_set1 = output_df1_100[['pressure', 'power', 'Ne', 'eV', 'EEPF']]
        input_vals2_100 = df_5_100_raw.iloc[2, 0:3].values
        output_df2_100 = df_5_100_raw.iloc[1:, [5, 6]].copy()
        output_df2_100.columns = ['eV', 'EEPF']
        output_df2_100.dropna(inplace=True)
        output_df2_100['pressure'] = input_vals2_100[0]
        output_df2_100['power'] = input_vals2_100[1]
        output_df2_100['Ne'] = input_vals2_100[2]
        df_5_100_set2 = output_df2_100[['pressure', 'power', 'Ne', 'eV', 'EEPF']]
        
        df_5_110_raw = pd.read_excel(file_path, sheet_name='5_110', header=None, usecols='A:C,E:F,H:I')
        input_vals1_110 = df_5_110_raw.iloc[1, 0:3].values
        output_df1_110 = df_5_110_raw.iloc[1:, [3, 4]].copy()
        output_df1_110.columns = ['eV', 'EEPF']
        output_df1_110.dropna(inplace=True)
        output_df1_110['pressure'] = input_vals1_110[0]
        output_df1_110['power'] = input_vals1_110[1]
        output_df1_110['Ne'] = input_vals1_110[2]
        df_5_110_set1 = output_df1_110[['pressure', 'power', 'Ne', 'eV', 'EEPF']]
        input_vals2_110 = df_5_110_raw.iloc[2, 0:3].values
        output_df2_110 = df_5_110_raw.iloc[1:, [5, 6]].copy()
        output_df2_110.columns = ['eV', 'EEPF']
        output_df2_110.dropna(inplace=True)
        output_df2_110['pressure'] = input_vals2_110[0]
        output_df2_110['power'] = input_vals2_110[1]
        output_df2_110['Ne'] = input_vals2_110[2]
        df_5_110_set2 = output_df2_110[['pressure', 'power', 'Ne', 'eV', 'EEPF']]
        
        df_final = pd.concat([df_5_100_set1, df_5_100_set2, df_5_110_set1, df_5_110_set2], ignore_index=True)
        for col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        df_final.dropna(inplace=True)
        
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the correct directory.")
        return None, None, None, None, None, None

    X_data = df_final[['pressure', 'power', 'Ne', 'eV']]
    Y_data = df_final['EEPF']
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # 2. Model building and training
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))

    Y_train_mean = Y_train.mean()
    Y_train_std = Y_train.std()

    Y_train_norm = (Y_train - Y_train_mean) / Y_train_std
    Y_test_norm = (Y_test - Y_train_mean) / Y_train_std

    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    
    with st.spinner('모델 학습 중... 잠시만 기다려주세요.'):
        history = model.fit(
            X_train,
            Y_train_norm,
            epochs=100,
            validation_split=0.2,
            verbose=0
        )
    
    return model, history, Y_train_mean, Y_train_std, len(X_train), len(X_test)


def run_dnn_sample_model_page(file_path):
    """
    Streamlit page for DNN model training and prediction.
    """
    st.title("DNN 모델 학습 및 EEPF 예측")
    st.write("엑셀 파일 데이터를 기반으로 딥러닝 모델을 학습하고, 새로운 입력값에 대한 EEPF 값을 예측합니다.")

    # Load and train the model using caching
    model, history, Y_train_mean, Y_train_std, train_size, test_size = load_and_train_sample_model(file_path)

    if model is None:
        st.stop()
        return
    
    st.markdown("---")
    st.subheader("모델 학습 정보")
    st.write(f"총 데이터 포인트 수: {train_size + test_size}")
    st.write(f"학습 데이터 수: {train_size}")
    st.write(f"테스트 데이터 수: {test_size}")

    # Plot the training history
    def plot_loss(history):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Training and Validation Loss Over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    def plot_mae(history):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['mean_absolute_error'], label='Training MAE')
        ax.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        ax.set_title('Training and Validation MAE Over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean Absolute Error')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with st.expander("모델 학습 결과 확인 (클릭)", expanded=False):
        plot_loss(history)
        plot_mae(history)

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
            predicted_norms = model.predict(custom_inputs).flatten()
            
            # Denormalize the predicted EEPF values
            predicted_eepfs = predicted_norms * Y_train_std + Y_train_mean

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
                if os.path.exists(HISTORY_PATH):
                    with open(HISTORY_PATH, 'r') as f:
                        history_data = json.load(f)
                    
                    # 로드된 딕셔너리를 Keras History 객체처럼 사용할 수 있도록 구조화
                    # Keras의 History 객체는 아니지만, plot 함수에서 dict 형태로 사용 가능
                    history = tf.keras.callbacks.History()
                    history.history = history_data 

                st.success(f"저장된 모델 및 매개변수를 로드했습니다. (학습 건너뛰기)")
                # 학습 이력(history)은 로드되지 않으므로 None 반환
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
        for pressure, power, eepf_json_str, Ne_value in records:
            try:
                if Ne_value is None or not isinstance(Ne_value, (float, int)):
                     st.warning(f"경고: pressure={pressure}, power={power} 조건의 Plasma Density(Np) 값이 유효하지 않습니다. 이 레코드를 건너뜁니다.")
                     continue
                
                eepf_data = json.loads(eepf_json_str)
                eV_list = eepf_data.get('eV', [])
                EEPF_list = eepf_data.get('EEPF', [])
                
                if len(eV_list) == len(EEPF_list) and len(eV_list) > 0:
                    for eV, EEPF in zip(eV_list, EEPF_list):
                        all_data.append({
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

    GROUP_KEYS = ['pressure', 'power']
    FEATURES = ['pressure', 'power', 'Ne', 'eV'] # X_data의 컬럼과 일치
    TARGET = 'EEPF' # Y_data의 컬럼과 일치
    # 1) 고유 그룹(압력, 파워 조합) 식별
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

    # ✅ 디버깅 정보 출력
    st.info(f"데이터 그룹 분할 (Pressure 기반):")
    st.info(f"  총 그룹 수: {len(groups)}")
    st.info(f"  훈련 그룹 수: {len(train_groups)}, 데이터 수: {train_size} ({train_size/len(df_final)*100:.1f}%)")
    st.info(f"  검증 그룹 수: {len(val_groups)}, 데이터 수: {val_size} ({val_size/len(df_final)*100:.1f}%)")
    st.info(f"  테스트 그룹 수: {len(test_groups)}, 데이터 수: {test_size} ({test_size/len(df_final)*100:.1f}%)")
    
    # Data Normalization
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train)) 

    Y_train_mean = Y_train.mean()
    Y_train_std = Y_train.std()

    Y_train_norm = (Y_train - Y_train_mean) / Y_train_std
    Y_val_norm = (Y_val - Y_train_mean) / Y_train_std # 검증 데이터도 정규화
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
    X_test_array = np.array(X_test)
    
    # Train Model
    with st.spinner('모델 학습 중... 잠시만 기다려주세요.'):
        history = model.fit(
            X_train_array,
            Y_train_norm,
            epochs=100,
            # Validation Split 대신 X_val, Y_val을 명시적으로 사용
            validation_data=(X_val_array, Y_val_norm), 
            verbose=0
        )
    
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
            
            with open(HISTORY_PATH, 'w') as f:
                json.dump(history.history, f)

            st.success(f"새롭게 학습된 모델과 매개변수가 '{MODEL_PATH}'와 '{PARAMS_PATH}'에 저장되었습니다.")
    except Exception as e:
         st.error(f"모델 저장 중 오류 발생: {e}")

    return model, history, Y_train_mean, Y_train_std, train_size, test_size


def run_dnn_model_page():
    """
    Streamlit page for DNN model training and prediction.
    """
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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Training and Validation Loss Over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    def plot_mae(history):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['mean_absolute_error'], label='Training MAE')
        ax.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        ax.set_title('Training and Validation MAE Over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean Absolute Error')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # 저장된 모델 로드 시 history가 None이므로, 학습된 경우에만 그래프 출력
    if history is not None:
        with st.expander("모델 학습 결과 확인 (클릭)", expanded=False):
            plot_loss(history)
            plot_mae(history)
    else:
        st.info("모델이 저장된 파일에서 로드되었으므로, 학습 이력(History)은 제공되지 않습니다.")

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