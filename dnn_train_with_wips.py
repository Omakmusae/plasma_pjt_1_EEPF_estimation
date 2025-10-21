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
from pathlib import Path

# --- 고정 변수 설정 ---
BASE_MODEL_PATH = Path("model") 

# --- 학습 하이퍼파라미터 정의 (Multi-Output 모델에 사용했던 값들) ---
DNN_LAYERS = '64x7' # 노드 수와 레이어 개수 표시 (예: '64x3' -> 64개 노드, 3개 레이어)
DROPOUT_RATE = 0.2 # Dropout 비율 (0.0: 사용 안함)
USE_BATCH_NORM = True # Batch Normalization 사용 여부 (True/False)
PATIENCE_COUNT = 18
LEARNING_RATE = 0.001 
EPOCHS = 100
TEST_SIZE_RATIO = 0.15
VALIDATION_SIZE_RATIO = 0.15

# Streamlit의 캐싱 기능을 사용하여 DB 로드 및 학습 결과를 저장합니다.
@st.cache_resource(show_spinner=False)
def train_model_from_db(post_fix=None):
  """
  Multi-Output DNN 모델을 학습하고 저장합니다. (Targets: EEPF, Ne, Te)
  
  Returns: (model, history, Y_train_mean, Y_train_std, train_size, test_size, val_size)
  """
  st.info("새로운 Multi-Output 모델 학습을 시작합니다. DB에서 데이터를 로드하고 DNN 모델을 훈련합니다.")
  
  # 에러 발생 시 반환할 7개의 None 튜플 (Unpacking 에러 방지)
  ERROR_RETURN = (None, None, None, None, None, None, None)
  # DB 파일 경로 설정 (실행 환경에 맞게 조정 필요)
  db_file_path = Path("./EEPF_estimation.db") 

  # -----------------------------
  # 1. DB Data Loading and Transformation
  # -----------------------------
  df_final = pd.DataFrame()
  
  if not db_file_path.exists():
    st.error(f"오류: 데이터베이스 파일 '{db_file_path}'이(가) 존재하지 않습니다. 학습을 진행할 수 없습니다.")
    return ERROR_RETURN

  try:
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # [수정된 쿼리] Target Ne, Te는 eepf_data(D)에서, Input Feature Te(wips_Te)와 Np는 wips_data(W)에서 가져옴
    cursor.execute(
      """
      SELECT
        G.id, G.pressure, G.power, G.eepf_json,
        D.Np, 
        D.Te AS Te_Target, 
        W.ion_flux, W.i_1w, W.i_2w, 
        W.Np,            
        W.Te AS wips_Te       
      FROM 
        eepf_graph AS G
      INNER JOIN 
        eepf_data AS D
      ON
        G.pressure = D.pressure AND G.power = D.power
      INNER JOIN 
        wips_data AS W
      ON
        G.pressure = W.pressure AND G.power = W.power
    
      """
    )
    #WHERE G.pressure = 5
    records = cursor.fetchall()
    conn.close()
    
    if not records:
      st.error(f"오류: 데이터베이스 '{db_file_path}'에서 데이터가 발견되지 않았습니다. 학습을 진행할 수 없습니다.")
      return ERROR_RETURN

    all_data = []
    # 레코드 형식 (11개): (eepf_id, pressure, power, eepf_json_str, D.Ne, Te_Target, ion_flux, i_1w, i_2w, Np, wips_Te)
    for eepf_id, pressure, power, eepf_json_str, Ne_value, Te_target_value, ion_flux_value, i_1w_value, i_2w_value, Np_value, wips_Te_value in records:
      try:
        # [유효성 검사] 모든 핵심 값이 유효한지 확인
        if any(v is None for v in [Ne_value, Te_target_value, ion_flux_value, i_1w_value, i_2w_value, Np_value, wips_Te_value]):
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
              'Ne': Ne_value,    # Target 2
              'Te': Te_target_value, # Target 3
              'ion_flux': ion_flux_value, 
              'i_1w': i_1w_value, 
              'i_2w': i_2w_value, 
              'Np': Np_value,    # Input Feature 6 (복구)
              'wips_Te': wips_Te_value, # Input Feature 7 (추가)
              'eV': eV,       # Input Feature 8
              'EEPF': EEPF     # Target 1
            })

      except Exception as inner_e:
        print(f"데이터 변환 중 알 수 없는 오류 발생 (pressure={pressure}, power={power}): {inner_e}")
        continue

    if not all_data:
      st.error("오류: 데이터베이스에서 유효한 EEPF 데이터를 추출하지 못했습니다.")
      return ERROR_RETURN
      
    df_final = pd.DataFrame(all_data)
    # 모든 컬럼을 숫자로 강제 변환 (NaN 발생 가능)
    for col in df_final.columns:
      df_final[col] = pd.to_numeric(df_final[col], errors='coerce') 
    
    # [필수] 결측치가 있는 행 제거 - 8개 Feature 및 3개 Target 모두 포함
    df_final.dropna(subset=['pressure', 'power', 'Ne', 'Te', 'EEPF', 
               'ion_flux', 'i_1w', 'i_2w', 'Np', 'wips_Te', 'eV'], inplace=True)
    
  except sqlite3.Error as e:
    st.error(f"SQLite 오류 발생: {e}. DB 파일 경로('{db_file_path}')를 확인해주세요.")
    return ERROR_RETURN
  except Exception as e:
    st.error(f"데이터 로드 및 변환 중 일반 오류 발생: {e}")
    return ERROR_RETURN
    
  if df_final.empty:
    st.error("오류: 모든 데이터를 로드하고 전처리한 후 유효한 데이터가 남아있지 않아 학습을 진행할 수 없습니다.")
    return ERROR_RETURN
    
  st.success(f"DB 로드 완료: 총 **{len(df_final):,}개** 데이터 포인트 확보")

  # -----------------------------
  # 2. Model Training (Multi-Output에 맞게 FEATURES/TARGETS 및 정규화)
  # -----------------------------

  GROUP_KEYS = ['pressure', 'power', 'eepf_id']
  
  # [수정] 8개 Feature 정의: 기존 7개 (Np 포함)에 wips_Te 추가
  FEATURES = ['pressure', 'power', 'ion_flux', 'i_1w', 'i_2w', 'Np', 'wips_Te', 'eV'] 
  
  # [유지] 3개 Target 정의
  TARGETS = ['EEPF', 'Ne', 'Te'] 
  
  # 데이터 분할 로직 (Pressure 기반 층화 추출)
  groups = df_final[GROUP_KEYS].drop_duplicates().reset_index(drop=True)
  
  # 그룹을 훈련+검증, 테스트 셋으로 분할
  train_val_groups, test_groups = train_test_split(
    groups, test_size=TEST_SIZE_RATIO, random_state=42, stratify=groups['pressure']
  )
  # 훈련+검증 셋을 다시 훈련, 검증 셋으로 분할
  val_split_ratio = VALIDATION_SIZE_RATIO / (1 - TEST_SIZE_RATIO) 
  train_groups, val_groups = train_test_split(
    train_val_groups, test_size=val_split_ratio, random_state=42, stratify=train_val_groups['pressure']
  )
  
  # 분할된 그룹 기준으로 실제 데이터프레임 매핑
  X_train_df = df_final.merge(train_groups, on=GROUP_KEYS, how='inner')
  X_val_df = df_final.merge(val_groups, on=GROUP_KEYS, how='inner')
  X_test_df = df_final.merge(test_groups, on=GROUP_KEYS, how='inner')
  
  # X와 Y 데이터 추출
  X_train = X_train_df[FEATURES]
  X_val = X_val_df[FEATURES]
  X_test = X_test_df[FEATURES]
  
  Y_train = X_train_df[TARGETS]
  Y_val = X_val_df[TARGETS]
  Y_test = X_test_df[TARGETS] 

  train_size = len(X_train); val_size = len(X_val); test_size = len(X_test)
  
  st.info(f"데이터 그룹 분할 (Pressure 기반):")
  st.code(f"""
  총 그룹 수: {len(groups)}
  훈련 그룹 수: {len(train_groups)}, 데이터 수: {train_size:,} ({train_size/len(df_final)*100:.1f}%)
  검증 그룹 수: {len(val_groups)}, 데이터 수: {val_size:,} ({val_size/len(df_final)*100:.1f}%)
  테스트 그룹 수: {len(test_groups)}, 데이터 수: {test_size:,} ({test_size/len(df_final)*100:.1f}%)
  """)

  # Data Normalization (Input Features)
  normalizer = layers.Normalization(axis=-1)
  normalizer.adapt(np.array(X_train)) 

  # [Multi-Output 정규화] Target DataFrame의 각 컬럼에 대해 mean/std 계산
  Y_train_mean = Y_train.mean().to_dict() # {Target_name: mean}
  Y_train_std = Y_train.std().to_dict() # {Target_name: std}
  
  # 정규화된 Y 데이터 생성
  Y_train_norm = (Y_train - Y_train.mean()) / Y_train.std()
  Y_val_norm = (Y_val - Y_train.mean()) / Y_train.std() 
  Y_test_norm = (Y_test - Y_train.mean()) / Y_train.std()
  
  # Build Model (동적 레이어, BN, Dropout 적용)
  try:
    node_count = int(DNN_LAYERS.split('x')[0]) 
    layer_count = int(DNN_LAYERS.split('x')[1]) 
    num_outputs = len(TARGETS) # [유지] 출력 노드 수는 Target 개수와 일치 (3개)
  except:
    st.error(f"오류: DNN_LAYERS 형식('{DNN_LAYERS}')이 올바르지 않습니다. '노드수x레이어수' 형식으로 설정해주세요 (예: '64x3').")
    return ERROR_RETURN

  # -----------------------------
  # Model Architecture Definition
  # -----------------------------
  model_layers = [normalizer]
  
  for i in range(layer_count):
    model_layers.append(layers.Dense(node_count, activation='relu'))
    if USE_BATCH_NORM:
      model_layers.append(layers.BatchNormalization())
    # 마지막 레이어에는 Dropout을 적용하지 않음
    if DROPOUT_RATE > 0.0 and i < layer_count - 1:
      model_layers.append(layers.Dropout(DROPOUT_RATE))

  # [Multi-Output] 최종 출력 레이어: 노드 수를 num_outputs(3)로 변경
  model_layers.append(layers.Dense(num_outputs))

  model = keras.Sequential(model_layers)

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
         # Multi-Output의 경우에도 기본 Loss는 MSE로 설정
         loss='mean_squared_error',
         metrics=['mean_absolute_error'])
  
  # Keras에 넣기 위해 NumPy Array로 변환
  X_train_array = np.array(X_train)
  X_val_array = np.array(X_val) 
  X_test_array = np.array(X_test)
  
  Y_train_norm_array = np.array(Y_train_norm)
  Y_val_norm_array = np.array(Y_val_norm)
  Y_test_norm_array = np.array(Y_test_norm)
  
  start_time = datetime.now()

  # Train Model
  with st.spinner('모델 학습 중... 잠시만 기다려주세요.'):
    print(f"\n--- Multi-Output 모델 학습 시작 (Epochs: {EPOCHS}) ---")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # 관찰할 지표 (검증 손실)
        patience=PATIENCE_COUNT,        # 성능 개선이 없을 때 몇 Epoch을 더 기다릴지
        restore_best_weights=True # 가장 좋았던 가중치로 복원할지 여부
    )

    history = model.fit(
      X_train_array,
      Y_train_norm_array,
      epochs=EPOCHS, 
      validation_data=(X_val_array, Y_val_norm_array),
      callbacks=[early_stopping],
      verbose=0
    )
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"총 소요 시간: {training_duration}")

  # Test Data 평가 및 History에 추가
  with st.spinner('테스트 데이터 성능 평가 중...'):
    # model.evaluate는 multi-output의 평균 loss/metric을 반환합니다.
    test_loss, test_mae = model.evaluate(X_test_array, Y_test_norm_array, verbose=0) 
    st.success(f"테스트 결과 (Multi-Output 평균) - Loss: **{test_loss:.4f}**, MAE: **{test_mae:.4f}**")
  
  # History Data 업데이트
  history_data = history.history.copy()
  history_data['test_loss'] = [test_loss] * len(history.epoch)
  history_data['test_mean_absolute_error'] = [test_mae] * len(history.epoch)
  
  # [수정] 정규화 파라미터를 history_data에 딕셔너리 형태로 저장 (TARGETS 정보 유지)
  history_data['norm_params'] = {
    'targets': TARGETS,
    'means': Y_train_mean,
    'stds': Y_train_std
  }
  
  # 데이터 분할 정보를 JSON에 저장하기 위해 리스트로 변환
  history_data['train_groups'] = train_groups.to_dict('records')
  history_data['val_groups'] = val_groups.to_dict('records')
  history_data['test_groups'] = test_groups.to_dict('records')

  # -----------------------------
  # 3. Save Model and Parameters (Multi-Output에 맞게 저장)
  # -----------------------------
  
  drop_str = f"_D{str(DROPOUT_RATE).replace('.', '')}" if DROPOUT_RATE > 0 else ""
  bn_str = "_BN" if USE_BATCH_NORM else ""
  
  dnn_str = f"Dnn{DNN_LAYERS}{drop_str}{bn_str}_MultiOut" # [유지] Multi-Output 명시
  
  lr_str = str(LEARNING_RATE).replace('.', '')
  postfix_str = f"_{post_fix}" if post_fix else ""
  # 모델 폴더명에 분/초를 추가하여 충돌 방지
  timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S") 

  MODEL_VERSION_NAME = (
    f"{dnn_str}_LR{lr_str}_Epo{EPOCHS}"
    f"{postfix_str}_{timestamp_str}"
  )

  SAVE_PATH = BASE_MODEL_PATH / MODEL_VERSION_NAME
  
  MODEL_FILE_NAME = "eepf_dnn_model.h5"
  PARAMS_FILE_NAME = "eepf_norm_params.npz"
  HISTORY_FILE_NAME = "eepf_history.json"
  
  MODEL_FULL_PATH = SAVE_PATH / MODEL_FILE_NAME
  PARAMS_FULL_PATH = SAVE_PATH / PARAMS_FILE_NAME
  HISTORY_FULL_PATH = SAVE_PATH / HISTORY_FILE_NAME
  
  try:
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\n모델 저장 디렉토리 생성: {SAVE_PATH}")
    
    with st.spinner('학습된 모델 저장 중...'):
      model.save(MODEL_FULL_PATH)
      
      # [유지] np.savez에 mean/std를 array 형태로 저장 (순서는 TARGETS=['EEPF', 'Ne', 'Te']와 일치)
      Y_train_mean_array = np.array([Y_train_mean[t] for t in TARGETS])
      Y_train_std_array = np.array([Y_train_std[t] for t in TARGETS])
      
      np.savez(PARAMS_FULL_PATH, 
          Y_train_mean=Y_train_mean_array, 
          Y_train_std=Y_train_std_array, 
          train_size=train_size, 
          test_size=test_size, 
          val_size=val_size)
      
      with open(HISTORY_FULL_PATH, 'w') as f:
        # json.dump는 Path 객체를 처리하지 못하므로, Path는 os.path.join()으로 변환해야 함
        json.dump(history_data, f, indent=4)

    st.success(f"새롭게 학습된 Multi-Output 모델과 매개변수가 **{MODEL_VERSION_NAME}** 폴더에 저장되었습니다.")
  except Exception as e:
    st.error(f"모델 저장 중 오류 발생: {e}")
  
  history.history = history_data

  # [유지] Y_train_mean/std를 dictionary 형태로 반환하여 타겟 이름 확인 가능하도록 함.
  return model, history, Y_train_mean, Y_train_std, train_size, test_size, val_size


def run_dnn_training_wips_page():
  """
  Streamlit 페이지: DNN 모델 학습 및 학습 결과 시각화
  """
  st.title("Multi-Output 딥러닝 모델 학습 및 저장")
  
  st.markdown("### 현재 모델 구조 및 Target 설정")
  # [수정] Feature 개수를 8개로 명시
  st.code(f"""
    [Input Features]: pressure, power, ion_flux, i_1w, i_2w, Np, wips_Te, eV (총 8개)
    [Target Variables]: EEPF, Ne, Te (총 3개)
    --------------------------------------------------
    DNN_LAYERS: {DNN_LAYERS}
    DROPOUT_RATE: {DROPOUT_RATE}
    USE_BATCH_NORM: {USE_BATCH_NORM}
    LEARNING_RATE: {LEARNING_RATE}
    EPOCHS: {EPOCHS}
      """)
  st.write("모델 구조를 변경하려면 코드 상단의 변수 값을 수정하세요.")
  
  # 모델 폴더 생성
  if not BASE_MODEL_PATH.exists():
    BASE_MODEL_PATH.mkdir(exist_ok=True)
    st.warning(f"모델 저장 폴더 '{BASE_MODEL_PATH}'를 생성했습니다.")

  post_fix_input = st.text_input(
    "선택적 Postfix 추가 (예: Custom_Run)",
    value="",
    help="모델 버전 폴더명에 추가할 선택적 정보입니다."
  )
  
  post_fix_to_use = post_fix_input.strip().replace(" ", "_") if post_fix_input else None
  
  # 학습 실행 버튼
  if st.button("모델 학습 시작 (DB에서 데이터 로드)", type="primary"):
    # 캐시 클리어 후 학습 시작
    st.cache_resource.clear()
    
    # [유지] 7개 반환값을 받도록 수정
    model, history, Y_train_mean, Y_train_std, train_size, test_size, val_size = train_model_from_db(post_fix=post_fix_to_use)

    if model is None:
      st.stop()
      
    st.markdown("---")
    st.subheader("모델 학습 정보 요약")
    st.write(f"총 데이터 포인트 수: **{train_size + val_size + test_size:,}**")
    st.write(f"훈련 데이터 수: **{train_size:,}**")
    st.write(f"검증 데이터 수: **{val_size:,}**")
    st.write(f"테스트 데이터 수: **{test_size:,}**")
    
    # Plot the training history (Loss)
    def plot_loss(history):
      test_loss_available = 'test_loss' in history.history and history.history['test_loss']
      
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.plot(history.history['loss'], label='Training Loss (EEPF + Ne + Te 평균)')
      ax.plot(history.history['val_loss'], label='Validation Loss (EEPF + Ne + Te 평균)')
      
      if test_loss_available:
        # Test Loss는 한 값만 존재하므로, 수평선으로 표시
        test_loss_value = history.history['test_loss'][0] 
        ax.axhline(y=test_loss_value, color='r', linestyle='--', label=f'Test Loss ({test_loss_value:.4f})')
      
      ax.set_title('Training, Validation, and Test Loss Over Epochs (Multi-Output Average)')
      ax.set_xlabel('Epochs')
      ax.set_ylabel('Loss (MSE)')
      ax.legend()
      ax.grid(True)
      st.pyplot(fig)

    # Plot the training history (MAE)
    def plot_mae(history):
      test_mae_available = 'test_mean_absolute_error' in history.history and history.history['test_mean_absolute_error']
      
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.plot(history.history['mean_absolute_error'], label='Training MAE (EEPF + Ne + Te 평균)')
      ax.plot(history.history['val_mean_absolute_error'], label='Validation MAE (EEPF + Ne + Te 평균)')
      
      if test_mae_available:
        # Test MAE는 한 값만 존재하므로, 수평선으로 표시
        test_mae_value = history.history['test_mean_absolute_error'][0]
        ax.axhline(y=test_mae_value, color='r', linestyle='--', label=f'Test MAE ({test_mae_value:.4f})')
      
      ax.set_title('Training, Validation, and Test MAE Over Epochs (Multi-Output Average)')
      ax.set_xlabel('Epochs')
      ax.set_ylabel('Mean Absolute Error (MAE)')
      ax.legend()
      ax.grid(True)
      st.pyplot(fig)

    # 모델 학습 결과 확인
    with st.expander("모델 학습 결과 확인 (클릭)", expanded=True):
      st.info("다중 출력 모델이므로, Loss와 MAE는 EEPF, Ne, Te 세 타겟에 대한 평균 지표입니다.")
      plot_loss(history)
      plot_mae(history)
      
    # 데이터 셋 현황 확인
    with st.expander("데이터 셋 현황 확인 (클릭)", expanded=False):
      if 'train_groups' in history.history and history.history['train_groups']:
        
        # Train/Validation/Test 그룹 정보를 DataFrame으로 변환
        df_train_groups = pd.DataFrame(history.history['train_groups']).set_index('eepf_id')
        df_val_groups = pd.DataFrame(history.history['val_groups']).set_index('eepf_id')
        df_test_groups = pd.DataFrame(history.history['test_groups']).set_index('eepf_id')
        
        st.markdown("**1. 훈련 데이터셋 (Train Groups) - 학습에 사용**")
        st.dataframe(df_train_groups[['pressure', 'power']])
        
        st.markdown("**2. 검증 데이터셋 (Validation Groups) - 학습 중 성능 모니터링에 사용**")
        st.dataframe(df_val_groups[['pressure', 'power']])
        
        st.markdown("**3. 테스트 데이터셋 (Test Groups) - 최종 모델 성능 평가에 사용**")
        st.dataframe(df_test_groups[['pressure', 'power']])
        
        st.info("표의 각 행은 특정 Power 및 Pressure 조건의 EEPF 그래프 데이터 (EEPField ID 기준)를 나타냅니다.")

      else:
        st.info("데이터셋 분할 정보가 학습 이력 파일에 기록되어 있지 않습니다.")


