import sqlite3

# DB 파일 연결 (없으면 자동 생성)
conn = sqlite3.connect("EEPF_estimation.db")
cursor = conn.cursor()

# 테이블 생성
cursor.execute("""
CREATE TABLE IF NOT EXISTS wips_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Te REAL,
    Np REAL,
    ion_flux REAL,
    i_1w REAL,
    i_2w REAL,
    pressure REAL,
    power REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS eepf_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Te REAL,
    Np REAL,
    Vp REAL,
    Vf REAL,
    pressure REAL,
    power REAL
)
""")


conn.commit()
conn.close()
