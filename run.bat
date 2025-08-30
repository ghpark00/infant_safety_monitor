@echo off
call "C:\Users\427s2\anaconda3\condabin\conda.bat" activate cuda_torch1
uvicorn main:app --reload
pause
