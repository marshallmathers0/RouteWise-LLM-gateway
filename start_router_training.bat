@echo off
rem ===========================================================================
rem  RouteWise router training, unattended. Double-click to start or resume.
rem    collect answers -> LLM judge -> labels -> train (MLflow) -> evaluate
rem    -> write the real results into README / REPORT / DEVLOG + resume bullets
rem  Every step resumes where it stopped, so it is safe to close this window or
rem  restart the PC and double-click again. It waits out Groq's free daily quota
rem  by itself; expect 1-2 days. Progress: data\router\pipeline.log
rem ===========================================================================
setlocal
title RouteWise router training
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "PYTHONWARNINGS=ignore"

if not exist "%PY%" (
    echo [RouteWise] .venv not found - see README "One-time setup".
    pause
    exit /b 1
)

set /a RUNS=0
:run
set /a RUNS+=1
echo [RouteWise] Router pipeline, attempt %RUNS% (log: data\router\pipeline.log)
"%PY%" -u scripts\run_router_pipeline.py
if not errorlevel 1 goto finished
if %RUNS% geq 30 (
    echo [RouteWise] Stopped 30 times - giving up. Check the end of data\router\pipeline.log.
    pause
    exit /b 1
)
echo [RouteWise] The pipeline stopped unexpectedly; restarting in 60 seconds ...
ping -n 61 127.0.0.1 >nul
goto run

:finished
echo [RouteWise] Training and evaluation finished. Writing results into the docs ...
"%PY%" scripts\finalize_router_results.py
echo.
echo [RouteWise] DONE. Next steps:
echo   1. Restart the demo (stop_demo.bat, then start_demo.bat) to use the trained router.
echo   2. git add . ^&^& git commit -m "Add trained router and results" ^&^& git push
echo      (with the GitHub-to-Hugging-Face workflow set up, this also updates your Space)
echo.
pause
exit /b 0
