import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st
import sqlite3
import json
import os 
from tensorflow.keras.models import load_model
from datetime import datetime

# --- 고정 변수 설정 ---
BASE_MODEL_PATH = "model" # 모든 모델 버전이 저장될 기본 디렉토리

# --- 학습 하이퍼파라미터 정의 (모델 버전 관리를 위해 사용) ---
# NOTE: 이 값들은 아래 model.compile 및 model.fit에 사용되는 실제 값입니다.
DNN_LAYERS = '32x2' # 모델 구조를 식별하기 위한 문자열 (Dense(64) -> Dense(64))
LEARNING_RATE = 0.001 # model.compile에 사용되는 학습률
EPOCHS = 100 # model.fit에 사용되는 에폭 수
TEST_SIZE_RATIO = 0.15
VALIDATION_SIZE_RATIO = 0.15 # 전체 데이터셋 대비 최종 비율

@st.cache_resource(show_spinner=False)
def train_model_from_db(post_fix=None):
    """
    DB에서 데이터를 로드, DNN 모델을 학습하고, 학습된 모델과 정규화 매개변수 및 학습 이력을 반환합니다.
    학습된 모델은 하이퍼파라미터를 포함한 폴더 내에 저장됩니다.
    
    Args:
        post_fix (str, optional): 모델 버전 폴더명에 추가할 선택적 접미사. 예: 'Drop01'. Defaults to None.
    """
    st.info("새로운 모델 학습을 시작합니다. DB에서 데이터를 로드하고 DNN 모델을 훈련합니다.")
    
    # 에러 발생 시 반환할 6개의 None 튜플 (Unpacking 에러 방지)
    ERROR_RETURN = (None, None, None, None, None, None)
    
    # DB 파일 경로 설정
    db_file_path = os.path.join(".\EEPF_estimation.db")

    # -----------------------------
    # 1. DB Data Loading and Transformation
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
            WHERE G.pressure = 5 
            """
            )
            #WHERE G.pressure = 5 
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            st.error(f"오류: 데이터베이스 '{db_file_path}'에서 조건에 맞는 데이터가 발견되지 않았습니다. 학습을 진행할 수 없습니다.")
            return ERROR_RETURN

        all_data = []
        for eepf_id, pressure, power, eepf_json_str, Ne_value in records:
            try:
                if Ne_value is None or not isinstance(Ne_value, (float, int)):
                    print(f"경고: pressure={pressure}, power={power} 조건의 Plasma Density(Np) 값이 유효하지 않아 레코드를 건너킵니다.")
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
                print(f"오류: pressure={pressure}, power={power} 조건의 eepf_json 데이터 파싱에 실패했습니다.")
                continue
            except Exception as inner_e:
                print(f"데이터 변환 중 알 수 없는 오류 발생 (pressure={pressure}, power={power}): {inner_e}")
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
    # 2. Model Training
    # -----------------------------

    GROUP_KEYS = ['pressure', 'power', 'eepf_id']
    FEATURES = ['pressure', 'power', 'Ne', 'eV'] # X_data의 컬럼과 일치
    TARGET = 'EEPF' # Y_data의 컬럼과 일치
    
    # 1) 고유 그룹 식별: (pressure, power, eepf_id) 조합을 그룹으로 인식
    groups = df_final[GROUP_KEYS].drop_duplicates().reset_index(drop=True)
    
    # 2) 고유 그룹을 Train/Test/Validation 그룹으로 분할 (Pressure 기반 계층적 샘플링)
    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=TEST_SIZE_RATIO, 
        random_state=42,
        stratify=groups['pressure'] 
    )

    val_split_ratio = VALIDATION_SIZE_RATIO / (1 - TEST_SIZE_RATIO) 
    
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_split_ratio,
        random_state=42,
        stratify=train_val_groups['pressure']
    )

    # Train/Validation/Test 그룹 정보 추출 (리스트로 변환)
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
    val_size = len(X_val) 
    test_size = len(X_test)
    
    # Streamlit에 분할 정보 출력
    st.info(f"데이터 그룹 분할 (Pressure 기반):")
    st.code(f"""
    총 그룹 수: {len(groups)}
    훈련 그룹 수: {len(train_groups)}, 데이터 수: {train_size} ({train_size/len(df_final)*100:.1f}%)
    검증 그룹 수: {len(val_groups)}, 데이터 수: {val_size} ({val_size/len(df_final)*100:.1f}%)
    테스트 그룹 수: {len(test_groups)}, 데이터 수: {test_size} ({test_size/len(df_final)*100:.1f}%)
    """)
    
    # Data Normalization
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train)) 

    Y_train_mean = Y_train.mean()
    Y_train_std = Y_train.std()

    Y_train_norm = (Y_train - Y_train_mean) / Y_train_std
    Y_val_norm = (Y_val - Y_train_mean) / Y_train_std 
    Y_test_norm = (Y_test - Y_train_mean) / Y_train_std

    # Build Model
    model = keras.Sequential([
        normalizer,
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), # 0.001 고정
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    
    X_train_array = np.array(X_train)
    X_val_array = np.array(X_val) 
    X_test_array = np.array(X_test)
    start_time = datetime.now()

    # Train Model
    with st.spinner('모델 학습 중... 잠시만 기다려주세요.'):
        print(f"\n--- 모델 학습 시작 ---")
        print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        history = model.fit(
            X_train_array,
            Y_train_norm,
            epochs=EPOCHS, # 100 고정
            validation_data=(X_val_array, Y_val_norm), 
            verbose=0
        )
        end_time = datetime.now()
        training_duration = end_time - start_time
        print(f"총 소요 시간: {training_duration}")
        
    # Test Data 평가 및 History에 추가
    with st.spinner('테스트 데이터 성능 평가 중...'):
        test_loss, test_mae = model.evaluate(X_test_array, Y_test_norm, verbose=0)
        st.success(f"테스트 결과 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")
    
    # Test 결과를 history.history 딕셔너리에 추가
    history_data = history.history.copy()
    history_data['test_loss'] = [test_loss] * len(history.epoch)
    history_data['test_mean_absolute_error'] = [test_mae] * len(history.epoch)

    # Train/Validation/Test 데이터셋 그룹 정보를 history_data에 추가
    history_data['train_groups'] = train_groups_list
    history_data['val_groups'] = val_groups_list
    history_data['test_groups'] = test_groups_list

    # -----------------------------
    # 3. Save Model and Parameters (학습 후 저장)
    # -----------------------------
    
    # 1. 모델 버전 폴더 경로 생성 (요구사항 반영)
    # 이름 형식: Dnn{Layers}_LR{LR}_Epo{EPOCHS}[_POSTFIX]_t{MM}
    
    # DNN_LAYERS (노드 및 레이어 정보) 포함
    dnn_str = f"Dnn{DNN_LAYERS}" 
    
    # LEARNING_RATE 문자열화 (소수점 제거)
    lr_str = str(LEARNING_RATE).replace('.', '')
    
    # Post Fix 처리 (선택적으로 추가)
    postfix_str = f"_{post_fix}" if post_fix else ""
    
    # 짧은 2자리 timestamp (현재 분, MM)를 맨 끝에 추가.
    # 초(SS)는 충돌 가능성이 높아 분(MM)을 사용합니다.
    timestamp_2digit = datetime.now().strftime("%M") 

    # 최종 폴더명 조합
    MODEL_VERSION_NAME = (
        f"{dnn_str}_LR{lr_str}_Epo{EPOCHS}"
        f"{postfix_str}_t{timestamp_2digit}"
    )

    SAVE_PATH = os.path.join(BASE_MODEL_PATH, MODEL_VERSION_NAME)
    
    # 모델 구성 요소 파일 이름은 고정 (폴더 내에서 관리)
    MODEL_FILE_NAME = "eepf_dnn_model.h5"
    PARAMS_FILE_NAME = "eepf_norm_params.npz"
    HISTORY_FILE_NAME = "eepf_history.json"
    
    MODEL_FULL_PATH = os.path.join(SAVE_PATH, MODEL_FILE_NAME)
    PARAMS_FULL_PATH = os.path.join(SAVE_PATH, PARAMS_FILE_NAME)
    HISTORY_FULL_PATH = os.path.join(SAVE_PATH, HISTORY_FILE_NAME)
    
    try:
        # 모델 저장 폴더 생성
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"\n모델 저장 디렉토리 생성: {SAVE_PATH}")
        
        with st.spinner('학습된 모델 저장 중...'):
            # 모델 전체 저장
            model.save(MODEL_FULL_PATH)

            # 정규화 매개변수 및 데이터 크기 저장
            np.savez(PARAMS_FULL_PATH, 
                     Y_train_mean=Y_train_mean, 
                     Y_train_std=Y_train_std, 
                     train_size=train_size, 
                     test_size=test_size,
                     val_size=val_size)
            
            # history_data (Test 지표 및 그룹 정보 포함) 저장
            with open(HISTORY_FULL_PATH, 'w') as f:
                json.dump(history_data, f, indent=4)

        st.success(f"새롭게 학습된 모델과 매개변수가 **{MODEL_VERSION_NAME}** 폴더에 저장되었습니다. 이제 'DNN 모델 로드 및 추론' 페이지에서 사용할 수 있습니다.")
    except Exception as e:
        st.error(f"모델 저장 중 오류 발생: {e}")
    
    # history 객체의 history 속성을 업데이트된 데이터로 변경하여 반환
    history.history = history_data

    return model, history, Y_train_mean, Y_train_std, train_size, test_size, val_size


def run_dnn_training_page():
    """
    Streamlit 페이지: DNN 모델 학습 및 학습 결과 시각화
    """
    st.title("딥러닝 모델 학습 및 저장")
    st.write("DB에 저장된 데이터를 사용하여 딥러닝 모델을 학습하고 결과를 저장합니다.")
    
    # Postfix 입력 필드 추가 (사용자 요구사항)
    post_fix_input = st.text_input(
        "선택적 Postfix 추가 (예: Drop01, BN_Added)",
        value="",
        help="모델 버전 폴더명에 추가할 선택적 정보입니다 (예: Dropout 사용 여부, Batch Normalization 등)"
    )
    
    post_fix_to_use = post_fix_input.strip().replace(" ", "_") if post_fix_input else None
    
    # 학습 실행 버튼
    if st.button("모델 학습 시작 (DB에서 데이터 로드)", type="primary"):
        # 실행 전 BASE_MODEL_PATH ('model') 폴더가 없으면 생성
        os.makedirs(BASE_MODEL_PATH, exist_ok=True)
        
        # 캐시된 데이터를 무시하고 강제로 재학습
        st.cache_resource.clear() 
        model, history, Y_train_mean, Y_train_std, train_size, test_size, val_size = train_model_from_db(post_fix=post_fix_to_use)

        if model is None:
            st.stop()
            
        st.markdown("---")
        st.subheader("모델 학습 정보 요약")
        st.write(f"총 데이터 포인트 수: **{train_size + val_size + test_size}**")
        st.write(f"훈련 데이터 수: **{train_size}**")
        st.write(f"검증 데이터 수: **{val_size}**")
        st.write(f"테스트 데이터 수: **{test_size}**")
        
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

        # 모델 학습 결과 확인 (클릭)
        with st.expander("모델 학습 결과 확인 (클릭)", expanded=True):
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
                st.info("데이터셋 분할 정보가 학습 이력 파일에 기록되어 있지 않습니다.")
