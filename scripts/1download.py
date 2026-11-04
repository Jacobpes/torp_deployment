#!/usr/bin/env python3
"""
SFTP Download Script
Downloads all files from the remote server to the local data/downloads directory.
"""

import os
import re
import shutil
import sys
import stat
import time
import paramiko
from pathlib import Path

# Configuration
SFTP_HOST = "64.226.94.227"
SFTP_USER = "torpshop_dl"
SFTP_REMOTE_PATH = "/prod/export"
SSH_KEY_PATH = "../id_ed25519"  # Relative to scripts/ directory
LOCAL_DOWNLOAD_PATH = "../data/downloads"  # Relative to scripts/ directory


def _project_paths():
    """Return (project_root, download_dir, stock_history_dir).

    Hanterar både scriptkörning och PyInstaller-bundle (sys.frozen).
    """
    if getattr(sys, 'frozen', False):
        project_root = Path(sys.executable).parent.resolve()
        download_dir = project_root / 'data' / 'nedladdningar'
    else:
        script_dir = Path(__file__).parent
        project_root = (script_dir.parent if script_dir.name == 'scripts'
                        else script_dir)
        download_dir = (project_root / 'data' / 'nedladdningar').resolve()
    stock_history_dir = project_root / 'system_data' / 'stock_history'
    return project_root, download_dir, stock_history_dir


def archive_stock_reports(download_dir, stock_history_dir):
    """
    Kopiera alla stock_report_*.csv från download_dir till
    stock_history_dir så vi bygger upp en saldo-historik över tid.

    Detta är grunden för att ML-träningen ska kunna se historiskt saldo
    och behandla "0 försäljning vid 0 lager" som ett missing-data-fall
    snarare än ett genuint "låg efterfrågan"-signal.

    Idempotent: om filen redan finns i arkivet hoppar vi över den
    (skriver inte över - första kopian vinner, då den motsvarar
    snapshotten det datumet faktiskt togs).
    """
    stock_history_dir.mkdir(parents=True, exist_ok=True)
    stock_files = sorted(download_dir.glob('stock_report_*.csv'))
    if not stock_files:
        print("[ARCHIVE] Inga stock_report-filer att arkivera.")
        return

    archived = 0
    skipped = 0
    failed = 0
    for src in stock_files:
        # Validera filnamn så vi inte arkiverar konstig filer.
        if not re.match(r'^stock_report_\d{4}-\d{2}-\d{2}\.csv$', src.name):
            continue
        dest = stock_history_dir / src.name
        if dest.exists():
            skipped += 1
            continue
        try:
            shutil.copy2(str(src), str(dest))
            archived += 1
        except OSError as e:
            print(f"  [WARNING] Could not archive {src.name}: {e}")
            failed += 1

    msg_parts = []
    if archived:
        msg_parts.append(f"{archived} nya")
    if skipped:
        msg_parts.append(f"{skipped} redan arkiverade")
    if failed:
        msg_parts.append(f"{failed} misslyckades")
    print(
        f"[ARCHIVE] Stock-historik: {', '.join(msg_parts) if msg_parts else 'inget att göra'} "
        f"({stock_history_dir})"
    )


def _import_closers():
    """Load the shared file-unlock helpers without a circular import."""
    script_dir = Path(__file__).parent.resolve()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from _common import close_all_excel_applications, release_locks_on_paths
    return close_all_excel_applications, release_locks_on_paths


def _is_local_lock_error(err):
    """True when a local file is open in another program or the fd cap is hit."""
    if isinstance(err, PermissionError):
        return True
    if isinstance(err, OSError) and getattr(err, 'errno', None) in (13, 16, 32):
        return True
    msg = str(err).lower()
    return (
        'permission denied' in msg
        or 'being used by another process' in msg
        or 'access is denied' in msg
        or 'too many open files' in msg
        or 'för många' in msg
    )


def _unlink_download_files(paths):
    """Delete files. Returns (removed_count, still_present)."""
    removed = 0
    failed = []
    for file_path in paths:
        file_path = Path(file_path)
        if not file_path.is_file():
            continue
        try:
            file_path.unlink(missing_ok=True)
            removed += 1
        except OSError as e:
            failed.append(file_path)
            print(f"  [WARNING] Could not delete {file_path.name}: {e}")
    return removed, failed


def _close_quietly(obj):
    if obj is None:
        return
    try:
        obj.close()
    except Exception:
        pass


def _clean_download_dir(download_dir):
    """
    Remove existing downloads so the next SFTP get is a fresh file.

    Files left open in Excel are closed and the delete is retried. A locked
    CSV used to survive into the download step, the overwrite failed, and
    sys.exit() then stopped the exe before any lists were written.
    """
    print("Cleaning download directory...")
    existing = [p for p in Path(download_dir).glob('*') if p.is_file()]
    if not existing:
        print("Download directory is already empty.\n")
        return

    removed, failed = _unlink_download_files(existing)
    if failed:
        print(
            f"  [INFO] {len(failed)} fil(er) är öppna och kunde inte raderas. "
            f"Stänger dem och försöker igen..."
        )
        try:
            close_all, release = _import_closers()
            close_all()
            release(failed)
            time.sleep(1.0)
        except Exception as e:
            print(f"  [VARNING] Kunde inte stänga öppna filer: {e}")
        removed_retry, failed = _unlink_download_files(failed)
        removed += removed_retry

    msg = f"Removed {removed} existing file(s) from download directory."
    if failed:
        msg += f" Skipped {len(failed)} file(s) that could not be deleted."
    print(msg + "\n")


def download_files():
    """Connect to SFTP server and download all files."""
    project_root, download_dir, stock_history_dir = _project_paths()
    
    key_path = (project_root / 'id_ed25519').resolve()
    
    # Ensure download directory exists
    download_dir.mkdir(parents=True, exist_ok=True)
    _clean_download_dir(download_dir)
    
    print(f"Connecting to {SFTP_HOST} as {SFTP_USER}...")
    print(f"Using SSH key: {key_path}")
    print(f"Download destination: {download_dir}")

    transport = None
    sftp = None
    try:
        # Load the private key (try Ed25519, then RSA, then ECDSA)
        private_key = None
        for key_class in [paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey]:
            try:
                private_key = key_class.from_private_key_file(str(key_path))
                break
            except (paramiko.ssh_exception.SSHException, Exception):
                continue
        
        if private_key is None:
            raise Exception("Unable to load SSH key. Unsupported key type.")
        
        # Create SSH client
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USER, pkey=private_key)
        
        # Create SFTP client
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        print(f"\nConnected! Changing to: {SFTP_REMOTE_PATH}")
        sftp.chdir(SFTP_REMOTE_PATH)
        
        # List all files in the remote directory
        print("\nListing remote files...")
        files = sftp.listdir()
        
        if not files:
            print("No files found in remote directory.")
            return
        
        print(f"\nFound {len(files)} file(s). Starting download...\n")
        
        # Download each file
        downloaded_count = 0
        for filename in files:
            remote_file = sftp.stat(filename)
            
            # Skip directories
            if not stat.S_ISREG(remote_file.st_mode):
                print(f"Skipping directory: {filename}")
                continue
            
            local_file = download_dir / filename
            
            print(f"Downloading: {filename} ({remote_file.st_size:,} bytes)...")
            
            try:
                sftp.get(filename, str(local_file))
            except Exception as e:
                if _is_local_lock_error(e):
                    print(f"  [INFO] {filename} är öppen. Stänger och försöker igen...")
                    try:
                        close_all, release = _import_closers()
                        close_all()
                        release([local_file])
                        time.sleep(1.0)
                        sftp.get(filename, str(local_file))
                    except Exception as retry_err:
                        # Plain ASCII tags so the Windows cp1252 console doesn't
                        # crash trying to render checkmarks/cross glyphs.
                        print(f"  [FAIL] Error downloading {filename}: {retry_err}")
                        continue
                else:
                    print(f"  [FAIL] Error downloading {filename}: {e}")
                    continue
            downloaded_count += 1
            print(f"  [OK] Saved to: {local_file}")
        
        print(f"\n{'='*60}")
        print(f"Download complete! {downloaded_count} file(s) downloaded.")
        print(f"Files saved to: {download_dir}")
        print(f"{'='*60}")

        # Arkivera stock_report-snapshots så vi bygger saldohistorik
        # framåt. Idempotent - kör utan biverkningar även om man kör
        # download.py flera gånger samma dag.
        archive_stock_reports(download_dir, stock_history_dir)
        
    except paramiko.ssh_exception.SSHException as e:
        print(f"SSH Error: {e}")
        raise
    except FileNotFoundError:
        print(f"Error: SSH key not found at {key_path}")
        print("Please ensure the id_ed25519 file exists in the project root.")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        # Always release the SSH session. An error path used to leave it
        # open, and sys.exit() then aborted the exe before list generation.
        _close_quietly(sftp)
        _close_quietly(transport)


if __name__ == "__main__":
    try:
        download_files()
    except Exception:
        sys.exit(1)

