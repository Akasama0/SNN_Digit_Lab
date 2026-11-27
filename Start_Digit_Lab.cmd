@echo off
cd /d "%~dp0"
if exist "%USERPROFILE%\anaconda3\python.exe" goto anaconda
python server.py --host 0.0.0.0
goto finish
:anaconda
"%USERPROFILE%\anaconda3\python.exe" server.py --host 0.0.0.0
:finish
pause
