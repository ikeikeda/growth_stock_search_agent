@echo off
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
uv run research >> logs\research_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
