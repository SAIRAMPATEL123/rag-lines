@echo off
set PYTHON="C:\Users\sairam.bathula\Desktop\install p\python.exe"

echo Starting RAG-Lines...
start cmd /k %PYTHON% main.py api
timeout /t 3
start cmd /k %PYTHON% -m streamlit run src\ui\streamlit_app.py

echo Done! Open http://localhost:8501
