@echo off
title LUMI Vision - Healthcare Companion
cd /d "%~dp0"

echo ====================================================
echo      LUMI Vision - Real-time Webcam Feed
echo ====================================================
echo Controls:
echo   [Q] or [ESC] : Quit application
echo   [E]          : Enrol face visible in camera
echo   [L]          : List enrolled identities
echo   [D]          : Delete an identity
echo   [S]          : Toggle facial landmarks
echo ====================================================
echo.

".venv\Scripts\python.exe" -m luminax_vision run

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
