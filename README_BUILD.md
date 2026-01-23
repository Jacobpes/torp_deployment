# Building Executables - Quick Guide

This project includes build scripts for both **macOS** and **Windows** executables.

## Quick Start

### macOS
```bash
./build_executable.sh
```
or
```bash
python3 build_macos_executable.py
```

### Windows
```batch
build_executable.bat
```
or
```batch
python build_windows_executable.py
```

## What Gets Built

Both executables include:
- ✅ All Python scripts (`main.py` and all scripts in `scripts/`)
- ✅ All data files (`data/parametrar/Beställningsfrekvens.csv`, `data/parametrar/Leveransfrekvens.csv`)
- ✅ Empty `data/downloads/` directory structure
- ✅ All Python dependencies (pandas, numpy, sklearn, matplotlib, paramiko, etc.)

## Output Files

- **macOS**: `dist/torp_report_generator`
- **Windows**: `dist/torp_report_generator.exe`

## Distribution

The executables are self-contained. You only need to include:
- The executable file itself
- Optional: `id_ed25519` (SSH key file, if using SFTP download feature)

All other files are embedded in the executable.

## Detailed Instructions

See `BUILD_INSTRUCTIONS.md` for complete documentation.







