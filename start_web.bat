@echo off
REM Start Torp Next.js UI on http://localhost:3000
cd /d "%~dp0web"
if not exist node_modules (
  echo Installing npm packages...
  call npm install
)
call npm run dev
