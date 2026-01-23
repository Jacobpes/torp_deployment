# Build Instructions for Torp Report Generator

This document explains how to build executables for both macOS and Windows.

## Prerequisites

1. **Python 3.8+** installed
2. **All dependencies** installed:
   ```bash
   pip install -r requirements_executable.txt
   ```
3. **PyInstaller** installed:
   ```bash
   pip install pyinstaller
   ```

## Required Files

Make sure these files exist before building:
- `main.py` - Main orchestrator script
- `scripts/1download.py` - SFTP download script
- `scripts/2generate_picking_list.py` - Picking list generator
- `scripts/3generate_order_list.py` - Order list generator
- `scripts/4generate_graphical_picking_list.py` - Graphical picking list generator
- `data/parametrar/Beställningsfrekvens.csv` - Supplier order frequencies
- `data/parametrar/Leveransfrekvens.csv` - Store delivery frequencies
- `data/downloads/` - Directory for downloaded data (can be empty)

## Building for macOS

### Option 1: Using the build script (Recommended)
```bash
./build_executable.sh
```

### Option 2: Using Python script
```bash
python3 build_macos_executable.py
```

### Option 3: Using PyInstaller directly
```bash
pyinstaller torp_report_generator_macos.spec --clean
```

**Output:** `dist/torp_report_generator`

**To run:**
```bash
./dist/torp_report_generator
```

## Building for Windows

### Option 1: Using the batch file (Recommended)
```batch
build_executable.bat
```

### Option 2: Using Python script
```batch
python build_windows_executable.py
```

### Option 3: Using PyInstaller directly
```batch
pyinstaller torp_report_generator.spec --clean
```

**Output:** `dist/torp_report_generator.exe`

**To run:**
Double-click `dist/torp_report_generator.exe` or run from command prompt.

## Build Options

### Console vs Windowed (Windows only)

The Windows executable can be built in two modes:

1. **Console version** (default): Shows output window with progress
2. **Windowed version**: No console window (silent execution)

To build windowed version, edit `torp_report_generator.spec` and change:
```python
console=False,  # Set to False for windowed version
```

Or use the build script and choose option 2 when prompted.

## What's Included

The executable includes:
- All Python scripts (`main.py` and all scripts in `scripts/`)
- All data files (`data/parametrar/Beställningsfrekvens.csv`, `data/parametrar/Leveransfrekvens.csv`)
- Empty `data/downloads/` directory structure
- All Python dependencies (pandas, numpy, sklearn, matplotlib, paramiko, etc.)

## Distribution

### Files to distribute:

1. **Executable file:**
   - macOS: `dist/torp_report_generator`
   - Windows: `dist/torp_report_generator.exe`

2. **Optional files** (if using SFTP download):
   - `id_ed25519` - SSH private key file
   - Place it in the same directory as the executable

### Note:
- The executable is self-contained and includes all dependencies
- Data files are embedded in the executable
- The `data/downloads/` directory will be created automatically when the executable runs

## Troubleshooting

### macOS Issues

1. **"App is damaged" error:**
   ```bash
   xattr -cr dist/torp_report_generator
   ```

2. **Gatekeeper blocking:**
   - Right-click the executable → Open
   - Or: System Preferences → Security & Privacy → Allow

### Windows Issues

1. **Antivirus false positive:**
   - Some antivirus software may flag PyInstaller executables
   - Add exception or use a code signing certificate

2. **Missing DLL errors:**
   - Ensure Visual C++ Redistributable is installed
   - Download from Microsoft's website

### General Issues

1. **Import errors:**
   - Check that all dependencies are in `requirements_executable.txt`
   - Rebuild with `--clean` flag

2. **File not found errors:**
   - Ensure all data files exist before building
   - Check that paths in spec file are correct

3. **Large executable size:**
   - This is normal (100-200 MB) due to included libraries
   - Consider using `--onedir` instead of `--onefile` for smaller size

## Advanced Configuration

### Custom Icon

1. **macOS:** Create `icon.icns` file
2. **Windows:** Create `icon.ico` file
3. Update spec file:
   ```python
   icon='icon.ico',  # or 'icon.icns' for macOS
   ```

### Code Signing (macOS)

To sign the executable:
```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/torp_report_generator
```

### Code Signing (Windows)

Use `signtool.exe` from Windows SDK to sign the executable.

## Build Output Structure

```
dist/
├── torp_report_generator      # macOS executable
└── torp_report_generator.exe  # Windows executable

build/                         # Temporary build files (can be deleted)
```

## Clean Build

To clean previous builds:
```bash
# macOS/Linux
rm -rf build dist __pycache__

# Windows
rmdir /s /q build dist __pycache__
```

Then rebuild from scratch.







