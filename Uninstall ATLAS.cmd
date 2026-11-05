@echo off
rem Double-click to uninstall ATLAS. Asks before deleting your data (settings, voices, agents, memories).
cd /d "%TEMP%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %*
echo.
echo This window closes in a few seconds.
ping -n 4 127.0.0.1 >nul
