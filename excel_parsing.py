import os
import sqlite3
import json
import pandas as pd

# -----------------------------
# 1. CSV 파일들이 들어있는 폴더
# -----------------------------
data_folder = os.path.join(os.path.dirname(__file__), "EEPF_graph_dataset")

csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

print("발견된 파일 목록:")
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
# 3. 조건별 데이터 파싱 함수
# -----------------------------
def parse_condition(df, start_row, power, pressure):
    """
    df: 전체 DataFrame
    start_row: 조건 데이터 시작 행 (0-based index)
    power: 파워 값 (int)
    pressure: 압력 값 (float)
    """
    # 데이터 시작/끝 인덱스 (1000개 데이터)
    data_start = start_row + 1
    data_end = data_start + 999

    ev1 = df.iloc[data_start:data_end, 6].tolist()    # G열
    eepf1 = df.iloc[data_start:data_end, 7].tolist()  # H열
    ev2 = df.iloc[data_start:data_end, 10].tolist()   # K열
    eepf2 = df.iloc[data_start:data_end, 11].tolist() # L열

    # null/0 값 필터링
    first = [{"eV": e, "EEPF": v} for e, v in zip(ev1, eepf1) if pd.notnull(e) and pd.notnull(v) and v != 0]
    second = [{"eV": e, "EEPF": v} for e, v in zip(ev2, eepf2) if pd.notnull(e) and pd.notnull(v) and v != 0]

    # 종료 마커 체크 (예: "100w_5mTorr")
    end_marker_row = data_end + 7  # A1008 같은 위치
    end_marker = str(df.iloc[end_marker_row, 0]).strip()
    expected_marker = f"{power}w_{int(pressure)}mTorr"
    if end_marker != expected_marker:
        print(f"[경고] {power}W 조건 종료 마커 불일치! ({end_marker} vs {expected_marker})")

    # JSON 데이터 구성
    eepf_data = {
        "condition": expected_marker,
        "first": first,
        "second": second
    }

    # DB 저장
    cursor.execute("""
        INSERT INTO eepf_graph (pressure, power, eepf_json)
        VALUES (?, ?, ?)
    """, (pressure, power, json.dumps(eepf_data)))


# -----------------------------
# 4. 각 CSV 파일 처리
# -----------------------------
for file in csv_files:
    file_path = os.path.join(data_folder, file)

    # 파일명에서 압력 추출
    base_name = os.path.splitext(file)[0]  # "5mTorr_100_550w"
    parts = base_name.split("_")
    pressure_str = parts[0]  # "5mTorr"
    pressure_val = float(pressure_str.replace("mTorr", ""))

    print(f"\n▶ 파일 처리 시작: {file} (압력={pressure_val} mTorr)")

    # CSV 읽기
    df = pd.read_csv(file_path)

    # 각 조건 블록 처리 (100W~550W, step 10)
    block_size = 1008
    start_row = 0
    for idx, power_val in enumerate(range(100, 560, 10)):
        row = start_row + idx * block_size
        parse_condition(df, row, power_val, pressure_val)

    print(f"  → {file} 처리 완료")

# -----------------------------
# 5. 저장 후 종료
# -----------------------------
conn.commit()
conn.close()
print(" 모든 CSV 파일 데이터가 eepf_graph 테이블에 저장되었습니다.")
