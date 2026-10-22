@echo off
title LUMI Vision - Healthcare Companion
cd /d "%~dp0"

echo ====================================================
echo      LUMI Vision - Real-time Vision & Fall Detection
echo ====================================================
echo Interactive Controls:
echo   [Q] or [ESC] : Quit application
echo   [E]          : Enrol face visible in camera
echo   [L]          : List enrolled identities
echo   [D]          : Delete an identity
echo   [S]          : Toggle skeleton & landmarks overlay
echo   [M]          : Toggle monitoring target person
echo   [A]          : Acknowledge active emergency alerts
echo ====================================================
echo Caregiver API: http://127.0.0.1:8000
echo WebSocket:    ws://127.0.0.1:8000/ws/events
echo ====================================================
echo.

".venv\Scripts\python.exe" -m luminax_vision run

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
