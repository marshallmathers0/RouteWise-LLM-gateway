@echo off
rem ===========================================================================
rem  RouteWise one-click demo (Windows). Double-click to start:
rem    1. the gateway (FastAPI)            -> http://localhost:8000  (API docs at /docs)
rem    2. the playground + dashboard UI    -> http://localhost:8501
rem  and open the UI in your browser. Each server runs in its own window;
rem  close those windows (or run stop_demo.bat) to stop the demo.
rem  Optional: "start_demo.bat --no-browser" skips opening the browser.
rem ===========================================================================
setlocal
title RouteWise demo launcher
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
set "CURL=%SystemRoot%\System32\curl.exe"
set "PYTHONWARNINGS=ignore"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" (
    echo [RouteWise] Python environment not found in .venv. One-time setup:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
    echo     .venv\Scripts\pip install -r requirements-dev.txt
    pause
    exit /b 1
)
if not exist ".env" (
    echo [RouteWise] .env not found. Copy .env.example to .env and add GROQ_API_KEY and GEMINI_API_KEY.
    pause
    exit /b 1
)

rem ---- 1. Gateway on :8000 (reuse it if it is already running) --------------
call :is_up http://localhost:8000/health
if errorlevel 1 (
    echo [RouteWise] Starting the gateway on http://localhost:8000 ...
    start "RouteWise gateway (port 8000)" cmd /k ""%PY%" -m uvicorn routewise.main:app --port 8000"
) else (
    echo [RouteWise] Gateway already running on http://localhost:8000
)

echo [RouteWise] Waiting for the gateway (first start loads the ML models, ~20-60 s) ...
set /a TRIES=0
:wait_gateway
call :is_up http://localhost:8000/health
if not errorlevel 1 goto gateway_ready
set /a TRIES+=1
if %TRIES% geq 180 (
    echo [RouteWise] The gateway did not start within 3 minutes. Check its window for errors.
    pause
    exit /b 1
)
ping -n 2 127.0.0.1 >nul
goto wait_gateway
:gateway_ready
echo [RouteWise] Gateway is up.

rem ---- 2. Playground + dashboard on :8501 -------------------------------------
call :is_up http://localhost:8501/_stcore/health
if errorlevel 1 (
    echo [RouteWise] Starting the playground + dashboard on http://localhost:8501 ...
    start "RouteWise playground (port 8501)" cmd /k ""%PY%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"
) else (
    echo [RouteWise] Playground already running on http://localhost:8501
)

set /a TRIES=0
:wait_ui
call :is_up http://localhost:8501/_stcore/health
if not errorlevel 1 goto ui_ready
set /a TRIES+=1
if %TRIES% geq 90 (
    echo [RouteWise] The playground did not start. Check its window for errors.
    pause
    exit /b 1
)
ping -n 2 127.0.0.1 >nul
goto wait_ui
:ui_ready

rem ---- 3. Open the browser ----------------------------------------------------
if /i not "%~1"=="--no-browser" start "" http://localhost:8501

echo.
echo [RouteWise] Demo is running:
echo     Playground + dashboard : http://localhost:8501
echo     Gateway API docs       : http://localhost:8000/docs   (Authorize with rw-dev-key)
echo     Stop everything        : close the two server windows, or run stop_demo.bat
echo.
ping -n 9 127.0.0.1 >nul
exit /b 0

rem ---- helper: exit code 0 if the URL answers HTTP 200 -------------------------
:is_up
"%CURL%" -s -f -o nul --max-time 2 "%~1" >nul 2>&1
exit /b %errorlevel%
