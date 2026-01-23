"""
Build script for creating Windows executable using PyInstaller
Run this script to build the executable: python build_windows_executable.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Check if PyInstaller is installed, install if missing
try:
    import PyInstaller.__main__
except ImportError:
    print("PyInstaller is not installed. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        import PyInstaller.__main__
        print("PyInstaller installed successfully!")
    except Exception as e:
        print(f"ERROR: Failed to install PyInstaller: {e}")
        print("Please install manually with: pip install pyinstaller")
        sys.exit(1)

def build_executable():
    """Build Windows executable using PyInstaller"""
    
    script_dir = Path(__file__).parent
    spec_file = script_dir / 'torp_report_generator.spec'
    
    # Check if spec file exists (preferred method)
    if spec_file.exists():
        print("Using existing spec file: torp_report_generator.spec")
        print("(Edit the spec file to customize build options)")
        print("\nBuilding Windows executable...")
        print("This may take several minutes...\n")
        
        try:
            PyInstaller.__main__.run([
                str(spec_file),
                '--clean',
            ])
        except Exception as e:
            print(f"\nERROR building executable: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Build without spec file
        print("No spec file found. Building with command-line options...")
        print("(Consider using torp_report_generator.spec for better control)\n")
        
        # Ask user if they want console window
        print("Choose build type:")
        print("1. Console version (shows output, recommended)")
        print("2. Windowed version (no console)")
        choice = input("Enter choice (1 or 2, default 1): ").strip() or "1"
        
        # PyInstaller options
        options = [
            'main.py',  # Main script
            '--name=torp_report_generator',  # Executable name
            '--onefile',  # Create a single executable file
            '--clean',  # Clean cache before building
            '--add-data=data/downloads;data/downloads',  # Include data directory
            '--add-data=data/parametrar/Beställningsfrekvens.csv;data',  # Include beställningsfrekvens
            '--add-data=data/parametrar/Leveransfrekvens.csv;data',  # Include leveransfrekvens
            '--hidden-import=pandas',
            '--hidden-import=numpy',
            '--hidden-import=sklearn',
            '--hidden-import=sklearn.neighbors',
            '--hidden-import=sklearn.neighbors._base',
            '--hidden-import=sklearn.neighbors._classification',
            '--hidden-import=sklearn.neighbors._regression',
            '--hidden-import=matplotlib',
            '--hidden-import=matplotlib.backends.backend_pdf',
            '--hidden-import=matplotlib.backends.backend_agg',
            '--hidden-import=paramiko',
            '--hidden-import=paramiko.ed25519key',
            '--hidden-import=paramiko.rsakey',
            '--hidden-import=paramiko.ecdsakey',
            '--collect-all=sklearn',
            '--collect-all=matplotlib',
        ]
        
        # Add console or windowed option
        if choice == "2":
            options.append('--windowed')  # No console window
            print("Building windowed version (no console)...")
        else:
            print("Building console version (shows output)...")
        
        # Add scripts directory if it exists
        scripts_dir = script_dir / 'scripts'
        if scripts_dir.exists():
            options.append(f'--add-data=scripts;scripts')
        
        print("Building Windows executable...")
        print(f"Options: {' '.join(options)}")
        print("\nThis may take several minutes...\n")
        
        try:
            PyInstaller.__main__.run(options)
        except Exception as e:
            print(f"\nERROR building executable: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "="*80)
    print("BUILD COMPLETE!")
    print("="*80)
    print(f"\nExecutable location: {script_dir / 'dist' / 'torp_report_generator.exe'}")
    print("\nNote: The executable includes all dependencies.")
    print("You can distribute this single .exe file along with:")
    print("  - id_ed25519 (SSH key file, if using SFTP download)")
    print("  - data/parametrar/Beställningsfrekvens.csv")
    print("  - data/parametrar/Leveransfrekvens.csv")

if __name__ == "__main__":
    build_executable()




