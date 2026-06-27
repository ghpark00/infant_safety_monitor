@echo off
call "C:\Users\427s2\anaconda3\condabin\conda.bat" activate cuda_torch1
uvicorn ai_backend.api.main:app --reload
pause
