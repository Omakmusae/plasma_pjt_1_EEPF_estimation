import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_resource
def load_and_train_model(file_path):
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


def run_dnn_model_page(file_path):
    """
    Streamlit page for DNN model training and prediction.
    """
    st.title("DNN 모델 학습 및 EEPF 예측")
    st.write("엑셀 파일 데이터를 기반으로 딥러닝 모델을 학습하고, 새로운 입력값에 대한 EEPF 값을 예측합니다.")

    # Load and train the model using caching
    model, history, Y_train_mean, Y_train_std, train_size, test_size = load_and_train_model(file_path)

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
            ev_values = np.arange(0, 17.01 + 0.045, 0.045)
            
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