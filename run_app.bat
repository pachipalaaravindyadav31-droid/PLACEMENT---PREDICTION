@echo off
title PlacementPredict Unified Launcher
echo ==============================================
echo  Starting PlacementPredict ML Application...
echo ==============================================
if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
    "%LOCALAPPDATA%\Python\bin\python.exe" app.py
) else (
    python app.py
)
pause
