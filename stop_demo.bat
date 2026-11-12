@echo off
rem Stops the RouteWise demo windows started by start_demo.bat (gateway + playground).
rem Only windows with these titles are closed; nothing else is touched.
taskkill /FI "WINDOWTITLE eq RouteWise gateway (port 8000)*" /T /F >nul 2>&1 && echo [RouteWise] Gateway stopped. || echo [RouteWise] Gateway window not found.
taskkill /FI "WINDOWTITLE eq RouteWise playground (port 8501)*" /T /F >nul 2>&1 && echo [RouteWise] Playground stopped. || echo [RouteWise] Playground window not found.
ping -n 4 127.0.0.1 >nul
