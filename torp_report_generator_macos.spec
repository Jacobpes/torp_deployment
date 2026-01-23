# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for torp_report_generator (macOS)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data/downloads', 'data/downloads'),
        ('data/parametrar/Beställningsfrekvens.csv', 'data'),
        ('data/parametrar/Leveransfrekvens.csv', 'data'),
        ('scripts', 'scripts'),
    ],
    hiddenimports=[
        'pandas',
        'numpy',
        'sklearn',
        'sklearn.neighbors',
        'sklearn.neighbors._base',
        'sklearn.neighbors._classification',
        'sklearn.neighbors._regression',
        'matplotlib',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_agg',
        'paramiko',
        'paramiko.ed25519key',
        'paramiko.rsakey',
        'paramiko.ecdsakey',
        'cryptography',
        'cryptography.hazmat',
        'cryptography.hazmat.primitives',
        'cryptography.hazmat.backends',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='torp_report_generator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one: 'icon.icns'
)
