import os
import sqlite3
import json
import pandas as pd
import re
import numpy as np

# -----------------------------
# 1. CSV 파일들이 들어있는 폴더 설정
# -----------------------------
data_folder = "EEPF_graph_dataset" 
csv_files = [] 

try:
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
except Exception as e:
    print(f"[오류] 파일 목록을 가져오는 중 오류 발생: {e}. '{data_folder}' 폴더 경로를 확인해주세요.")

print("발견된 파일 목록:")
if not csv_files:
    print("[경고] CSV 파일이 data_folder에 없습니다. 폴더 경로와 파일 존재 여부를 확인해주세요.")
else:
    for file in csv_files:
        print(file)

# -----------------------------
# 2. SQLite 연결 및 테이블 생성
# -----------------------------
conn = sqlite3.connect("eepf_graph.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS eepf_graph (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pressure REAL,
    power REAL,
    eepf_json TEXT
)
""")

# -----------------------------
# 3. 마커에서 조건 추출 함수 (변동 없음)
# -----------------------------
def parse_marker(marker_str):
    """A열 마커 문자열에서 파워와 압력을 추출합니다 (예: '400w_15mTorr')."""
    if not isinstance(marker_str, str):
        return None, None
    
    # 띄어쓰기 유무에 관계없이 (w_? 혹은 w) 숫자(w) + _? + 숫자(mTorr) 패턴 검색
    match = re.search(r'(\d+)\s*w_?(\d+)\s*mTorr', marker_str, re.IGNORECASE)
    
    if match:
        power = float(match.group(1))
        pressure = float(match.group(2))
        return power, pressure
    return None, None


# -----------------------------
# 4. 조건별 데이터 파싱 함수 (변동 없음)
# -----------------------------
def parse_condition(df, data_start_row, power, pressure, file_name):
    data_end = data_start_row + 1000 

    def safe_get(idx, start, end):
        if idx < df.shape[1]:
            col_data = df.iloc[start:end, idx]
            return col_data.replace([np.inf, -np.inf], np.nan).fillna(0).tolist()
        else:
            return [0.0] * 1000

    # EEPF1 데이터 추출: G열(6), H열(7)
    ev1 = safe_get(6, data_start_row, data_end) 
    eepf1 = safe_get(7, data_start_row, data_end) 
    
    # EEPF2 데이터 추출: K열(10), L열(11)
    ev2 = safe_get(10, data_start_row, data_end)
    eepf2 = safe_get(11, data_start_row, data_end)

    # DB 저장 로직 (2개의 레코드 저장)
    eepf1_json_data = {"Measurement_ID": 1, "eV": ev1, "EEPF": eepf1}
    cursor.execute("INSERT INTO eepf_graph (pressure, power, eepf_json) VALUES (?, ?, ?)", 
                   (pressure, power, json.dumps(eepf1_json_data)))

    eepf2_json_data = {"Measurement_ID": 2, "eV": ev2, "EEPF": eepf2}
    cursor.execute("INSERT INTO eepf_graph (pressure, power, eepf_json) VALUES (?, ?, ?)", 
                   (pressure, power, json.dumps(eepf2_json_data)))
    
    print(f"  → DB 저장 완료: P={pressure}mTorr, W={power}W (2개 레코드)")


# -----------------------------
# 5. 메인 처리 루프 (마커 위치 수정)
# -----------------------------
block_size = 1008
total_parsed_conditions = 0

for file in csv_files:
    file_path = os.path.join(data_folder, file)

    print(f"\n▶ 파일 처리 시작: {file}")

    try:
        # [파일 읽기 유지] skiprows=1과 sep=','를 사용하여 파일 읽기 오류를 해결
        df = pd.read_csv(file_path, header=None, skiprows=1, sep=',')
    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {file}. {e}")
        continue
    
    # 총 파싱 가능한 조건 개수 계산: DF 전체 행 수를 블록 크기(1008)로 나눔
    # (원래 파일 A1이 skip되었으므로, DF의 행 수는 원래 파일 행 수 - 1)
    # block_size가 1008이면, 각 블록은 1008행으로 취급하여 다음 블록의 시작점을 계산
    num_conditions = len(df) // block_size
    num_conditions+=1
    if num_conditions == 0:
        print(f"  → [경고] {file}: 파일 크기가 너무 작아 파싱할 데이터 블록이 없습니다.")
        continue

    print(f"  → 파일 크기를 기반으로 총 {num_conditions}개 조건 블록 파싱 예정.")

    for i in range(num_conditions+1):
        # 데이터 시작 행 (A2 -> 0, A1010 -> 1008, A2018 -> 2016 ...)
        data_start_row = i * block_size 
        
        # [핵심 수정]: 마커 행 인덱스 = 데이터 시작 행 + 1006
        # A2(idx 0)에서 A1008(Power Marker)까지는 1007행이므로, 인덱스는 1006입니다.
        marker_row_index = data_start_row + 1006 
        
        if marker_row_index < len(df):
            # A열의 마커 값 추출 (df.iloc[행, 0])
            marker_str = df.iloc[marker_row_index, 0] 
            power, pressure = parse_marker(str(marker_str))

            if power is not None and pressure is not None:
                # 데이터 파싱 및 DB 저장 실행
                parse_condition(df, data_start_row, power, pressure, file)
                total_parsed_conditions += 1
            else:
                print(f"  → [경고] {file}: 블록 {i+1}의 마커 '{marker_str}'에서 조건 추출 실패. (마커 문자열이 'W_mTorr' 형식이 아닙니다.)")
        else:
            print(f"  → [경고] {file}: 블록 {i+1}에 해당하는 마커 행 ({marker_row_index}번째)이 파일 길이를 초과하여 건너뜀.")
            
    print(f"  → {file} 처리 완료. {total_parsed_conditions}개 조건 파싱됨.")


# -----------------------------
# 6. 저장 후 종료
# -----------------------------
conn.commit()
conn.close()
print(f"\n✅ 모든 CSV 파일 데이터 처리 완료. 총 {total_parsed_conditions}개 조건 (총 {total_parsed_conditions * 2}개 EEPF 레코드)이 DB에 저장되었습니다.")