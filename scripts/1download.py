#!/usr/bin/env python3
"""
SFTP Download Script
Downloads all files from the remote server to the local data/downloads directory.
"""

import os
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


def download_files():
    """Connect to SFTP server and download all files."""
    
    # Get absolute paths
    # When running as executable, __file__ points to temp directory
    # Use sys.executable to find where .exe is located (same as main.py)
    if getattr(sys, 'frozen', False):
        # Running as executable - use sys.executable to find where .exe file is located
        project_root = Path(sys.executable).parent.resolve()
        # Save downloads in data/nedladdningar subdirectory
        download_dir = project_root / 'data' / 'nedladdningar'
    else:
        # Running as script - use script directory and go up to project root
        script_dir = Path(__file__).parent
        if script_dir.name == 'scripts':
            project_root = script_dir.parent
        else:
            project_root = script_dir
        download_dir = (project_root / 'data' / 'nedladdningar').resolve()
    
    key_path = (project_root / 'id_ed25519').resolve()
    
    # Ensure download directory exists
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove all existing files in the download directory
    print("Cleaning download directory...")
    existing_files = list(download_dir.glob("*"))
    if existing_files:
        removed_count = 0
        for file_path in existing_files:
            if file_path.is_file():
                file_path.unlink()
                removed_count += 1
        print(f"Removed {removed_count} existing file(s) from download directory.\n")
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
                print(f"  ✓ Saved to: {local_file}")
            except Exception as e:
                print(f"  ✗ Error downloading {filename}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Download complete! {downloaded_count} file(s) downloaded.")
        print(f"Files saved to: {download_dir}")
        print(f"{'='*60}")
        
        # Close connections
        sftp.close()
        transport.close()
        
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

