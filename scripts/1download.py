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


def download_files():
    """Connect to SFTP server and download all files."""
    project_root, download_dir, stock_history_dir = _project_paths()
    
    key_path = (project_root / 'id_ed25519').resolve()
    
    # Ensure download directory exists
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove all existing files in the download directory. Tolerate
    # files that vanish between listing and unlink (race condition) or
    # are locked by another process - we just keep going and let the
    # subsequent download overwrite what remains.
    print("Cleaning download directory...")
    existing_files = list(download_dir.glob("*"))
    if existing_files:
        removed_count = 0
        failed_count = 0
        for file_path in existing_files:
            if not file_path.is_file():
                continue
            try:
                file_path.unlink(missing_ok=True)
                removed_count += 1
            except OSError as e:
                # File locked / permission denied / disappeared with an
                # error other than FileNotFoundError. Log and continue.
                failed_count += 1
                print(f"  [WARNING] Could not delete {file_path.name}: {e}")
        msg = f"Removed {removed_count} existing file(s) from download directory."
        if failed_count:
            msg += f" Skipped {failed_count} file(s) that could not be deleted."
        print(msg + "\n")
    else:
        print("Download directory is already empty.\n")
    
    print(f"Connecting to {SFTP_HOST} as {SFTP_USER}...")
    print(f"Using SSH key: {key_path}")
    print(f"Download destination: {download_dir}")
    
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
            remote_path = f"{SFTP_REMOTE_PATH}/{filename}"
            
            print(f"Downloading: {filename} ({remote_file.st_size:,} bytes)...")
            
            try:
                sftp.get(filename, str(local_file))
                downloaded_count += 1
                print(f"  [OK] Saved to: {local_file}")
            except Exception as e:
                # Plain ASCII tags so the Windows cp1252 console doesn't
                # crash trying to render checkmarks/cross glyphs.
                print(f"  [FAIL] Error downloading {filename}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Download complete! {downloaded_count} file(s) downloaded.")
        print(f"Files saved to: {download_dir}")
        print(f"{'='*60}")
        
        # Close connections
        sftp.close()
        transport.close()

        # Arkivera stock_report-snapshots så vi bygger saldohistorik
        # framåt (gör efter download lyckats men före några andra
        # exceptions kan hända). Idempotent - kör utan biverkningar
        # även om man kör download.py flera gånger samma dag.
        archive_stock_reports(download_dir, stock_history_dir)
        
    except paramiko.ssh_exception.SSHException as e:
        print(f"SSH Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: SSH key not found at {key_path}")
        print("Please ensure the id_ed25519 file exists in the project root.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_files()

