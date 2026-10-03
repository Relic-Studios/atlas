@echo off
rem Double-click to install ATLAS. Safe to run again: finished steps are skipped.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
