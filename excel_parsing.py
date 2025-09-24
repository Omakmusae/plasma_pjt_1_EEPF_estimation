import os
import pandas as pd

# CSV 파일들이 들어있는 폴더
data_folder = os.path.join(os.path.dirname(__file__), "EEPF_graph_dataset")

# 폴더 안의 모든 csv 파일 가져오기
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

print("발견된 파일 목록:")
for file in csv_files:
    print(file)

# 예시: 첫 번째 파일을 pandas로 읽기
if csv_files:
    sample_file = os.path.join(data_folder, csv_files[0])
    df = pd.read_csv(sample_file)
    print("\n샘플 데이터 미리보기:")
    print(df.head())
