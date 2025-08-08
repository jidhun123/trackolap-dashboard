@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "C:\Users\JIDHUN K M\Desktop\6branch" || (echo CD failed & exit /b 1)

REM Show current branch and remote
echo === Git context ===
git rev-parse --abbrev-ref HEAD || exit /b 1
git remote -v

REM Make sure we’re on main and up to date
git fetch origin || exit /b 1
git switch main || exit /b 1
git pull --rebase origin main || exit /b 1

REM Run ETL (fail fast if it errors)
echo === Running ETL ===
python etl_task_report.py
if errorlevel 1 (
  echo ETL failed with exit code %errorlevel%
  exit /b %errorlevel%
)

REM Force a detectable change every run
for /f %%A in ('powershell -NoProfile -Command "(Get-Date).ToString(\"s\")"') do set TS=%%A
echo %TS%> timestamp.txt

REM Stage changes (all, in case schema/files change)
git add -A

REM Skip commit if nothing actually changed
git diff --cached --quiet
if %errorlevel%==0 (
  echo No changes to commit. Exiting cleanly.
  exit /b 0
)

REM Commit with timestamp
git commit -m "Hourly update of live_data.csv @ %TS%" || exit /b 1

REM Push current HEAD to origin/main
git push -u origin HEAD || exit /b 1

echo === Done ===
exit /b 0
