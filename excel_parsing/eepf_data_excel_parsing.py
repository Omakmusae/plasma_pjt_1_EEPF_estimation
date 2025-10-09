import os
import sqlite3
import pandas as pd
import re
import numpy as np

# -----------------------------
# 1. 파일 및 폴더 설정
# -----------------------------
excel_file_name = "EEPF_data.xlsx"
data_folder = ".."
excel_file_path = os.path.join(excel_file_name)
sheet_name = "Sheet1" 

print(f"▶ 처리할 엑셀 파일: {excel_file_name} ({sheet_name})")

if not os.path.exists(excel_file_path):
    print(f"\n[경고] '{excel_file_name}' 파일을 찾을 수 없습니다. 파일을 스크립트와 같은 폴더에 놓아주세요.")

# -----------------------------
# 2. SQLite 연결
# -----------------------------

db_file_path = os.path.join(".", "EEPF_estimation.db")
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# [수정] 테이블 생성 DDL 삭제 (사용자 요청 반영)
print("\n[DB] 기존 'eepf_data' 테이블에 연결했습니다.")

# -----------------------------
# 3. 마커에서 조건 추출 함수
# -----------------------------
def parse_power_pressure_marker(marker_str):
    """C2 셀 등의 마커 문자열에서 파워와 압력을 추출합니다 (예: '100w_5mTorr')."""
    if not isinstance(marker_str, str):
        return None, None
    
    # 숫자(w) + _ + 숫자(mTorr) 패턴 검색
    match = re.search(r'(\d+)\s*w_?(\d+)\s*mTorr', marker_str, re.IGNORECASE)
    
    if match:
        power = float(match.group(1))
        pressure = float(match.group(2))
        return power, pressure
    return None, None

def insert_data(cursor, te, np_val, vp, vf, pressure_val, current_power):
    """DB에 데이터를 삽입합니다."""
    cursor.execute("""
        INSERT INTO eepf_data (Te, Np, Vp, Vf, pressure, power)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (te, np_val, vp, vf, pressure_val, current_power))

# -----------------------------
# 4. 메인 파싱 함수 (데이터 저장 오류 수정 로직 적용됨)
# -----------------------------
def parse_eepf_data(excel_path, sheet_name, db_cursor):
    """
    EEPF_data.xlsx 파일의 Sheet1을 읽고 데이터를 파싱하고 DB에 저장합니다.
    """
    total_parsed_conditions = 0
    
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    except FileNotFoundError:
        print(f"[오류] 엑셀 파일 '{excel_file_name}'을 찾을 수 없습니다.")
        return 0
    except Exception as e:
        print(f"[오류] 엑셀 파일 로드 중 오류 발생: {e}")
        return 0

    print(f"  → 총 {df.shape[0]}행, {df.shape[1]}열 데이터 로드됨.")

    # 마커 셀(C2, L2, U2, ...)을 찾기 위한 열 인덱스 리스트 생성 (Index 1)
    marker_cols = []
    if df.shape[0] > 1:
        for col_idx in range(df.shape[1]):
            marker_cell_raw = df.iloc[1, col_idx]
            
            # [수정된 로직] 셀 내용을 문자열로 강제 변환하여 데이터 타입 오류 방지
            marker_str = str(marker_cell_raw).strip() 
            marker_str_lower = marker_str.lower()
            
            # 마커 조건 확인
            if marker_str_lower != 'nan' and 'w' in marker_str_lower and 'mtorr' in marker_str_lower:
                marker_cols.append(col_idx)

    if not marker_cols:
        print("  → [경고] 데이터 블록을 구분하는 마커(예: '100w_5mTorr')를 찾을 수 없습니다.")
        return 0
    
    print(f"  → {len(marker_cols)}개의 데이터 블록 마커 열 인덱스 발견: {marker_cols}")


    # 각 데이터 블록(마커 열)을 순회하며 데이터 추출
    for marker_col in marker_cols:
        marker_str = str(df.iloc[1, marker_col]).strip()
        fixed_power_val, pressure_val = parse_power_pressure_marker(marker_str)
        
        if pressure_val is None:
            print(f"  → [경고] 마커 '{marker_str}'에서 압력 값을 추출할 수 없습니다. 건너뜀.")
            continue

        print(f"  → 조건 블록 파싱 시작: P={pressure_val}mTorr (마커 셀 인덱스: {marker_col})")
        
        # 데이터 열 정의: 마커 셀을 기준으로 파생됨
        power_col = marker_col - 1 
        te_col = marker_col + 1
        np_col = marker_col + 2
        vp_col = marker_col + 3
        vf_col = marker_col + 4
        
        if power_col < 0 or vf_col >= df.shape[1]:
            print(f"  → [경고] 마커 {marker_str}의 주변 데이터 열 인덱스가 유효하지 않습니다. 건너뜀.")
            continue
            
        current_block_count = 0
        
        # -----------------------------
        # 4-1. 100W 데이터 분리 파싱 (Index 1)
        # -----------------------------
        row_idx_100w = 1 
        try:
            row = df.iloc[row_idx_100w]
            
            # Power 값을 안전하게 추출
            if pd.notna(row[power_col]):
                current_power = float(row[power_col])
            else:
                # Power 값이 없으면 해당 행을 파싱할 필요가 없음
                raise ValueError("Power value is missing in 100W row.") 

            if current_power == fixed_power_val: 
                # Te, Np, Vp, Vf 값을 안전하게 추출 및 0.0으로 대체
                te = float(row[te_col]) if pd.notna(row[te_col]) else 0.0
                np_val = float(row[np_col]) if pd.notna(row[np_col]) else 0.0
                vp = float(row[vp_col]) if pd.notna(row[vp_col]) else 0.0
                vf = float(row[vf_col]) if pd.notna(row[vf_col]) else 0.0

                insert_data(db_cursor, te, np_val, vp, vf, pressure_val, current_power)
                current_block_count += 1
                total_parsed_conditions += 1
                
        except (ValueError, TypeError, IndexError) as e:
            print(f"  → [경고] 100W 데이터 파싱 오류 (Index 1, 마커 {marker_str}): {e}. 건너뜀.")
            pass


        # -----------------------------
        # 4-2. 110W 이상 데이터 파싱 (Index 2부터)
        # -----------------------------
        data_start_row = 2 

        for row_idx in range(data_start_row, df.shape[0]):
            row = df.iloc[row_idx]
            
            # Power 열에 값이 없거나 NaN이면 데이터 블록의 끝으로 간주
            if pd.isna(row[power_col]):
                break

            try:
                # Power 값을 안전하게 추출
                current_power = float(row[power_col])
                
                # 나머지 데이터도 안전하게 추출
                te = float(row[te_col]) if pd.notna(row[te_col]) else 0.0
                np_val = float(row[np_col]) if pd.notna(row[np_col]) else 0.0
                vp = float(row[vp_col]) if pd.notna(row[vp_col]) else 0.0
                vf = float(row[vf_col]) if pd.notna(row[vf_col]) else 0.0

                insert_data(db_cursor, te, np_val, vp, vf, pressure_val, current_power)
                
                current_block_count += 1
                total_parsed_conditions += 1
                
            except (ValueError, TypeError, IndexError) as e:
                # 데이터가 중간에 잘못된 경우 (해당 행만 건너뜀)
                print(f"  → [경고] 110W 이상 데이터 파싱 오류 (Index {row_idx}, 마커 {marker_str}): {e}. 해당 행 건너뜀.")
                pass
        
        print(f"  → '{marker_str}' 블록 처리 완료. 총 {current_block_count}개 레코드 저장됨.")

    return total_parsed_conditions

# -----------------------------
# 5. 메인 실행
# -----------------------------
# DB에 중복 데이터가 쌓이는 것을 방지하기 위해 기존 테이블 데이터를 먼저 삭제합니다.
cursor.execute("DELETE FROM eepf_data")
conn.commit() 
print("\n[DB] 중복 방지를 위해 기존 데이터를 모두 삭제했습니다.")

# 데이터 파싱 시작
total_parsed_conditions = parse_eepf_data(excel_file_path, sheet_name, cursor)


# -----------------------------
# 6. 저장 후 종료
# -----------------------------
conn.commit()
conn.close()
print(f"\n✅ '{excel_file_name}' 파일 처리 완료. 총 {total_parsed_conditions}개 조건이 'eepf_data' 테이블에 저장되었습니다.")