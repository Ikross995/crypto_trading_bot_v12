@echo off
echo ======================================================================
echo    Enhanced Trading Dashboard Server
echo ======================================================================
echo.
echo Starting web server...
echo.

cd /d "%~dp0"

REM Try different Python commands
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found: python
    python dashboard_server.py
    goto :end
)

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found: python3
    python3 dashboard_server.py
    goto :end
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found: py
    py dashboard_server.py
    goto :end
)

REM If Python not found, open file directly
echo.
echo WARNING: Python not found in PATH!
echo Opening dashboard file directly in browser...
echo.
start "" "data\learning_reports\enhanced_dashboard.html"

:end
echo.
pause
