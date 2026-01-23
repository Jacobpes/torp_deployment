#!/usr/bin/env python3
"""
Main orchestrator script that:
1. Downloads data from SFTP server
2. Generates picking lists
3. Generates order lists per supplier
4. Generates graphical picking lists

This script can be packaged as a Windows executable.
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Determine the base directory
# When running as executable (PyInstaller), find where .exe is located
if getattr(sys, 'frozen', False):
    # Running as executable - sys.executable points to .exe file
    script_dir = Path(sys.executable).parent.resolve()
else:
    # Running as script - use __file__ location
    script_dir = Path(__file__).parent.resolve()

scripts_dir = script_dir / 'scripts'
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(scripts_dir))

# Change to script directory to ensure relative paths work
# This ensures all files are saved in the same folder as .exe
os.chdir(script_dir)

# Setup logging
LOG_FILE = script_dir / 'torp_report_generator.log'
ERROR_LOG_FILE = script_dir / 'torp_report_generator_errors.log'

# Clear log files at start of each run
if LOG_FILE.exists():
    LOG_FILE.unlink()
if ERROR_LOG_FILE.exists():
    ERROR_LOG_FILE.unlink()

# Configure main logger (logs everything)
main_logger = logging.getLogger('main')
main_logger.setLevel(logging.DEBUG)
main_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
main_handler.setLevel(logging.DEBUG)
main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
main_handler.setFormatter(main_formatter)
main_logger.addHandler(main_handler)

# Configure error logger (logs only warnings and errors)
error_logger = logging.getLogger('errors')
error_logger.setLevel(logging.WARNING)
error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding='utf-8')
error_handler.setLevel(logging.WARNING)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

# Also log to console (keep original stdout)
# Use errors='replace' to handle Unicode characters that can't be encoded to console
console_handler = logging.StreamHandler(sys.__stdout__)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
# Set encoding to handle Unicode characters
if hasattr(console_handler.stream, 'reconfigure'):
    try:
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass
main_logger.addHandler(console_handler)

class LoggerWriter:
    """Wrapper to redirect print statements to logger while also printing to console"""
    def __init__(self, logger, error_logger, level):
        self.logger = logger
        self.error_logger = error_logger
        self.level = level
        self.original_stdout = sys.__stdout__
        
    def write(self, message):
        if message.strip():
            # Log to main log
            self.logger.log(self.level, message.rstrip())
            # Also log errors/warnings to error log
            if self.level >= logging.WARNING:
                self.error_logger.log(self.level, message.rstrip())
            # Also print to console
            self.original_stdout.write(message)
            self.original_stdout.flush()
    
    def flush(self):
        self.original_stdout.flush()

# Redirect stdout and stderr to logger (but keep console output)
sys.stdout = LoggerWriter(main_logger, error_logger, logging.INFO)
sys.stderr = LoggerWriter(error_logger, error_logger, logging.ERROR)

def print_header(text):
    """Print a formatted header"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = f"\n{'='*80}\n{text}\n{'='*80}\n"
    main_logger.info(header)
    error_logger.warning(header)  # Also log headers to error log for context

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print_header(f"Running: {description}")
    
    script_path = Path(script_path)
    
    # When running as executable, scripts are in PyInstaller's temp directory
    is_frozen = getattr(sys, 'frozen', False)
    
    if is_frozen:
        # PyInstaller extracts files to sys._MEIPASS
        # Scripts are in the 'scripts' subdirectory there
        meipass = Path(sys._MEIPASS)
        possible_paths = [
            meipass / 'scripts' / script_path.name,  # scripts/1download.py in temp dir
            meipass / script_path.name,  # Direct in temp dir
        ]
    else:
        # When running as script, try normal locations
        possible_paths = [
            script_path,
            script_dir / script_path.name,
            scripts_dir / script_path.name,
            script_dir / script_path,
        ]
    
    actual_path = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            actual_path = path.resolve()
            break
    
    if actual_path is None:
        error_msg = f"ERROR: Script not found. Tried:\n"
        for path in possible_paths:
            error_msg += f"  - {path}\n"
        if is_frozen:
            error_msg += f"\nPyInstaller temp dir: {sys._MEIPASS}\n"
        main_logger.error(error_msg)
        error_logger.error(error_msg)
        return False
    
    if is_frozen:
        # When running as executable, import and run directly (avoid subprocess loop)
        try:
            # Create a unique module name to avoid conflicts
            module_name = f"script_{actual_path.stem}_{id(actual_path)}"
            spec = importlib.util.spec_from_file_location(module_name, str(actual_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load script: {actual_path}")
            
            # Add script directory to path temporarily
            script_parent = str(actual_path.parent)
            if script_parent not in sys.path:
                sys.path.insert(0, script_parent)
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Call the main function if it exists (for scripts that use if __name__ == "__main__")
            # Try common function names
            if hasattr(module, 'main'):
                module.main()
            elif hasattr(module, 'download_files'):
                module.download_files()
            elif hasattr(module, 'run'):
                module.run()
            
            main_logger.info(f"\n[OK] Successfully completed: {description}")
            return True
        except Exception as e:
            error_msg = f"\n[ERROR] Error running {description}: {e}"
            main_logger.error(error_msg)
            error_logger.error(error_msg)
            import traceback
            tb_str = traceback.format_exc()
            main_logger.error(tb_str)
            error_logger.error(tb_str)
            return False
    else:
        # When running as normal Python script, use subprocess
        try:
            result = subprocess.run(
                [sys.executable, str(actual_path)],
                cwd=str(script_dir),
                check=True,
                capture_output=False,
                text=True
            )
            main_logger.info(f"\n[OK] Successfully completed: {description}")
            return True
        except FileNotFoundError:
            # If subprocess fails, try importing and running directly
            try:
                module_name = f"script_{actual_path.stem}_{id(actual_path)}"
                spec = importlib.util.spec_from_file_location(module_name, str(actual_path))
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load script: {actual_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                main_logger.info(f"\n[OK] Successfully completed: {description}")
                return True
            except Exception as e:
                error_msg = f"\n[ERROR] Error running {description}: {e}"
                main_logger.error(error_msg)
                error_logger.error(error_msg)
                import traceback
                tb_str = traceback.format_exc()
                main_logger.error(tb_str)
                error_logger.error(tb_str)
                return False
        except subprocess.CalledProcessError as e:
            error_msg = f"\n✗ Error running {description}: {e}"
            main_logger.error(error_msg)
            error_logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"\n✗ Unexpected error running {description}: {e}"
            main_logger.error(error_msg)
            error_logger.error(error_msg)
            import traceback
            tb_str = traceback.format_exc()
            main_logger.error(tb_str)
            error_logger.error(tb_str)
            return False

def main():
    """Main orchestrator function"""
    start_time = datetime.now()
    main_logger.info("="*80)
    main_logger.info("TORP SHOP - AUTOMATED REPORT GENERATOR")
    main_logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main_logger.info(f"Working directory: {script_dir}")
    main_logger.info(f"Log file: {LOG_FILE}")
    main_logger.info(f"Error log file: {ERROR_LOG_FILE}")
    main_logger.info("="*80)
    
    # Step 1: Download data from SFTP
    print_header("STEP 1: Downloading data from SFTP server")
    download_script = Path('scripts/1download.py')
    
    success = run_script(download_script, "Download data from SFTP")
    if not success:
        warning_msg = "\nWARNING: Download failed or script not found.\nContinuing with existing data files...\n(This is OK if you're using pre-downloaded data)"
        main_logger.warning(warning_msg)
        error_logger.warning(warning_msg)
    
    # Step 2: Generate picking list
    print_header("STEP 2: Generating picking list")
    picking_script = Path('scripts/2generate_picking_list.py')
    success = run_script(picking_script, "Generate picking list")
    if not success:
        error_msg = "\nERROR: Failed to generate picking list. Stopping."
        main_logger.error(error_msg)
        error_logger.error(error_msg)
        return False
    
    # Step 3: Generate order lists per supplier
    print_header("STEP 3: Generating order lists per supplier")
    order_script = Path('scripts/3generate_order_list.py')
    success = run_script(order_script, "Generate order lists per supplier")
    if not success:
        warning_msg = "\nWARNING: Failed to generate order lists. Continuing..."
        main_logger.warning(warning_msg)
        error_logger.warning(warning_msg)
    
    # Step 4: Generate graphical picking lists
    print_header("STEP 4: Generating graphical picking lists")
    graphical_script = Path('scripts/4generate_graphical_picking_list.py')
    success = run_script(graphical_script, "Generate graphical picking lists")
    if not success:
        warning_msg = "\nWARNING: Failed to generate graphical picking lists. Continuing..."
        main_logger.warning(warning_msg)
        error_logger.warning(warning_msg)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    print_header("PROCESS COMPLETE")
    main_logger.info(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main_logger.info(f"Total duration: {duration}")
    main_logger.info("\nGenerated files:")
    main_logger.info(f"  - Picking lists: {script_dir / 'plocklistor'}")
    main_logger.info(f"  - Order lists: {script_dir / 'orderlistor'}")
    main_logger.info(f"  - Graphical picking lists: {script_dir / 'picking_list_graphical'}")
    main_logger.info(f"  - Log file: {LOG_FILE}")
    main_logger.info(f"  - Error log file: {ERROR_LOG_FILE}")
    main_logger.info("\n" + "="*80)
    
    # Keep window open on Windows if running as executable
    if sys.platform == 'win32' and getattr(sys, 'frozen', False):
        input("\nPress Enter to exit...")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        error_msg = "\n\nProcess interrupted by user."
        main_logger.error(error_msg)
        error_logger.error(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"\n\nFATAL ERROR: {e}"
        main_logger.error(error_msg)
        error_logger.error(error_msg)
        import traceback
        tb_str = traceback.format_exc()
        main_logger.error(tb_str)
        error_logger.error(tb_str)
        if sys.platform == 'win32' and getattr(sys, 'frozen', False):
            input("\nPress Enter to exit...")
        sys.exit(1)








