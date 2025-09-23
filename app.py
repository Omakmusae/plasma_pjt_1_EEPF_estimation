import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dnn_model import run_dnn_model_page

def maxwellian_eepf(energy, Te, ne):
    """
    Maxwellian 가정 하의 EEPF 계산 (단위: eV^-3/2 * cm^-3)
    """
    # eV, Te의 단위는 eV, ne는 cm-3
    # EEPF의 단위는 eV^(-3/2) * cm^(-3) 입니다.
    coef = ne / (Te**1.5)
    return coef * np.sqrt(energy) * np.exp(-energy / Te)

def show_maxwellian_eepf():
    """
    기존 Maxwellian EEPF 시각화 페이지를 보여주는 함수
    """
    st.title("Maxwellian EEPF 시각화 (로그 스케일 + 단위 반영)")
    Te = st.number_input("전자 온도 Te [eV]", min_value=0.1, value=3.0, step=0.1)
    ne = st.number_input("전자 밀도 ne [cm⁻³]", min_value=1e8, value=1e10, step=1e9, format="%.1e")

    # 에너지 축 정의
    energy = np.linspace(0.1, 30, 500)
    eepf = maxwellian_eepf(energy, Te, ne)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(energy, eepf, color='red', label="Fitted EEPF (Maxwellian)")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [eV]", fontsize=12)
    # y축 레이블의 단위는 LaTeX 수식으로 올바르게 표기
    ax.set_ylabel(r"EEPF [eV$^{-3/2}$ cm$^{-3}$]", fontsize=12)
    
    # 제목의 과학적 표기법을 Matplotlib가 수식으로 해석하지 않도록 변경
    # 'e+10'과 같은 표기 대신 직접 10의 지수승으로 변환
    if ne >= 1e9:
        ne_formatted = f"{ne / 1e9:.1f} x 10⁹"
    else:
        ne_formatted = f"{ne:.1e}"
    
    ax.set_title(f"Maxwellian EEPF (Te = {Te:.2f} eV, ne = {ne_formatted} cm⁻³)", fontsize=14)

    ax.grid(True)
    ax.legend()
    
    st.pyplot(fig)

def main():
    """
    메인 Streamlit 애플리케이션
    """
    st.sidebar.title("시각화 선택")
    
    # 사이드바에서 탭 선택
    selection = st.sidebar.radio(
        "어떤 기능을 사용하시겠습니까?",
        ("Maxwellian EEPF 시각화", "DNN 모델 학습 및 예측")
    )

    if selection == "Maxwellian EEPF 시각화":
        show_maxwellian_eepf()
    elif selection == "DNN 모델 학습 및 예측":
        # 데이터 파일 경로 설정
        file_path = 'plasma_ne_sample.xlsx' 
        run_dnn_model_page(file_path)

if __name__ == "__main__":
    main()
