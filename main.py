import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    resolve_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolve_path
#
if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--server.port=8502",  # 포트를 8502으로 설정
        "--global.developmentMode=false"
    ]
    sys.exit(stcli.main())
    

