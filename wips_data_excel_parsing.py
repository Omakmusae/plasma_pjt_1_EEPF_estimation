import os
import sqlite3
import json
import pandas as pd
import re
import numpy as np

# -----------------------------
# 1. 파일 및 폴더 설정
# -----------------------------
# [수정] WIPS 엑셀 파일 이름 지정
excel_file_name = "wips_data.xlsx"
data_folder = "./"
excel_file_path = os.path.join(data_folder, excel_file_name)

print(f"▶ 처리할 엑셀 파일: {excel_file_name}")

if not os.path.exists(excel_file_path):
    # [경고] 파일이 존재하지 않는 경우 (CSV로 변환된 파일을 업로드한 경우를 대비해 안내)
    print(f"\n[경고] '{excel_file_name}' 파일을 찾을 수 없습니다. \n만약 '{excel_file_name}' 파일을 이미 CSV 파일들로 변환하여 업로드했다면, \n이전 답변의 CSV 파싱 코드를 사용하는 것이 더 적절합니다.")
    print("현재는 엑셀 파일을 직접 읽는 코드로 진행합니다.")

# -----------------------------
# 2. SQLite 연결 및 테이블 생성
# -----------------------------
conn = sqlite3.connect("EEPF_estimation.db")
cursor = conn.cursor()



# -----------------------------
# 3. 데이터 로드 및 파싱 함수
# -----------------------------

def parse_wips_data(excel_path):
    """
    wips_data.xlsx 파일의 각 시트를 읽고, 두 데이터 블록에서 
    Power, Te, ion_flux, Np, i_1w, i_2w 데이터를 추출하여 DB에 저장합니다.
    """
    total_parsed_conditions = 0
    
    try:
        # Excel 파일의 모든 시트를 DataFrame으로 읽어옵니다.
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
    except FileNotFoundError:
        # 파일이 없을 경우 이미 위에서 경고했으므로 함수를 종료합니다.
        return 0
    except Exception as e:
        print(f"[오류] 엑셀 파일 로드 중 오류 발생: {e}")
        return 0


    for sheet_name in sheet_names:
        print(f"\n▶ 시트 처리 시작: {sheet_name}")
        
        # 시트 이름에서 압력 추출 (예: '5mTorr' -> 5.0)
        pressure_match = re.search(r'(\d+)\s*mTorr', sheet_name, re.IGNORECASE)
        if pressure_match:
            pressure_val = float(pressure_match.group(1))
        else:
            print(f"  → [경고] 시트 이름 '{sheet_name}'에서 압력 정보를 추출할 수 없어 건너뜁니다.")
            continue
        
        try:
            # 헤더 없이 모든 데이터를 DataFrame으로 읽어옵니다.
            df = xls.parse(sheet_name, header=None)
        except Exception as e:
            print(f"  → [오류] 시트 '{sheet_name}' 읽기 실패: {e}")
            continue

        # [핵심 로직] 데이터는 Power, Te, ion_flux, Np, i_1w, i_2w 순서로 저장되어 있습니다.
        # A열(인덱스 0)은 Power, F열(인덱스 5)은 i_2w입니다.
        # H열(인덱스 7)은 Power, M열(인덱스 12)은 i_2w입니다.
        
        # A3부터 시작하는 두 개의 데이터 블록을 처리합니다.
        # A3은 DF의 인덱스 2입니다.
        start_row_index = 2
        
        # 1. 좌측 데이터 블록 (A:F, 인덱스 0:5)
        # Power(0), Te(1), ion_flux(2), Np(3), i_1w(4), i_2w(5)
        parsed_left = parse_block(df, start_row_index, 0, 6, pressure_val, sheet_name)
        total_parsed_conditions += parsed_left
        
        # 2. 우측 데이터 블록 (H:M, 인덱스 7:12)
        # Power(7), Te(8), ion_flux(9), Np(10), i_1w(11), i_2w(12)
        parsed_right = parse_block(df, start_row_index, 7, 13, pressure_val, sheet_name)
        total_parsed_conditions += parsed_right
        
        print(f"  → {sheet_name} 처리 완료. {parsed_left + parsed_right}개 조건이 DB에 추가됨.")

    return total_parsed_conditions


def parse_block(df, start_row, start_col, end_col, pressure, sheet_name):
    """
    지정된 DF 영역에서 데이터를 추출하여 wips_data 테이블에 저장합니다.
    """
    count = 0
    # 데이터는 A3(인덱스 2)부터 시작합니다.
    for i in range(start_row, len(df)):
        row = df.iloc[i]
        
        # Power 열(첫 번째 인덱스)에 값이 없거나 NaN이면 데이터 블록의 끝으로 간주하고 중단
        if pd.isna(row[start_col]) or row[start_col] == '':
            break

        try:
            power = float(row[start_col])
            # Te, ion_flux, Np, i_1w, i_2w는 다음 5개 열입니다.
            # DDL 순서: Te, Np, ion_flux, i_1w, i_2w, pressure, power

            # 엑셀 데이터 순서: Power(0), Te(1), ion_flux(2), Np(3), i_1w(4), i_2w(5)
            # DF 인덱스: start_col + 1, start_col + 2, ..., start_col + 5
            
            # [매핑] DF 인덱스 -> DDL 변수
            # Te: [start_col + 1]
            # Np: [start_col + 3]
            # ion_flux: [start_col + 2]
            # i_1w: [start_col + 4]
            # i_2w: [start_col + 5]
            
            # 파싱할 지표 5개의 인덱스
            # Power 열 인덱스를 start_col이라고 할 때,
            # Te: start_col + 1 (Te)
            # ion_flux: start_col + 2 (ion_flux)
            # Np: start_col + 3 (Np)
            # i_1w: start_col + 4 (i_1w)
            # i_2w: start_col + 5 (i_2w)
            
            # DDL 순서에 맞게 값을 배열
            metrics_to_save = [
                float(row[start_col + 1]) if pd.notnull(row[start_col + 1]) else 0.0, # Te
                float(row[start_col + 3]) if pd.notnull(row[start_col + 3]) else 0.0, # Np
                float(row[start_col + 2]) if pd.notnull(row[start_col + 2]) else 0.0, # ion_flux
                float(row[start_col + 4]) if pd.notnull(row[start_col + 4]) else 0.0, # i_1w
                float(row[start_col + 5]) if pd.notnull(row[start_col + 5]) else 0.0  # i_2w
            ]
            
            # DB 저장
            cursor.execute("""
                INSERT INTO wips_data (Te, Np, ion_flux, i_1w, i_2w, pressure, power)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (*metrics_to_save, pressure, power))
            count += 1
            
        except (ValueError, TypeError, IndexError) as e:
            # 데이터 변환 오류 또는 인덱스 오류 발생 시 해당 레코드 건너뛰기
            # print(f"  → [경고] {sheet_name} 블록 파싱 오류 (행 {i+1}): {e}") # 디버깅 시에만 사용
            pass
            
    return count

# -----------------------------
# 5. 메인 실행
# -----------------------------
total_parsed_conditions = parse_wips_data(excel_file_path)


# -----------------------------
# 6. 저장 후 종료
# -----------------------------
conn.commit()
conn.close()
print(f"\n✅ '{excel_file_name}' 파일 처리 완료. 총 {total_parsed_conditions}개 조건이 'wips_data' 테이블에 저장되었습니다.")