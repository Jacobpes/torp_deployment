#!/bin/bash
# Build script for macOS executable
# Usage: ./build_executable.sh

set -e

echo "=========================================="
echo "Building macOS Executable"
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Check if PyInstaller is installed, install if missing
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller is not installed. Installing..."
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyInstaller"
        exit 1
    fi
fi

# Check if required files exist
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found"
    exit 1
fi

if [ ! -f "data/parametrar/Leveransfrekvens.csv" ]; then
    echo "Error: data/parametrar/Leveransfrekvens.csv not found"
    exit 1
fi

if [ ! -f "data/parametrar/Beställningsfrekvens.csv" ]; then
    echo "Error: data/parametrar/Beställningsfrekvens.csv not found"
    exit 1
fi

# Run the build script
python3 build_macos_executable.py

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Executable location: dist/torp_report_generator"
echo ""
echo "To run: ./dist/torp_report_generator"
