@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not exist logs mkdir logs
uv run research >> logs\research_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
