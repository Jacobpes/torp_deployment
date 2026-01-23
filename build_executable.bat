@echo off
REM Windows batch script to build the executable
REM Double-click this file or run from command prompt

echo ========================================
echo Building Windows Executable
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Step 1: Installing dependencies...
pip install -r requirements_executable.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo.
echo Step 2: Building executable...
python build_windows_executable.py
if errorlevel 1 (
    echo ERROR: Failed to build executable
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executable location: dist\torp_report_generator.exe
echo.
echo Don't forget to include these files with the executable:
echo   - id_ed25519 (SSH key, if using SFTP download)
echo   - data\parametrar\Beställningsfrekvens.csv
echo   - data\parametrar\Leveransfrekvens.csv
echo.
pause








