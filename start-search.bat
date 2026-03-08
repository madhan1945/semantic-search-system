@echo off
cd D:\Projects\semantic-search-system
call .venv\Scripts\activate.bat
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
pause