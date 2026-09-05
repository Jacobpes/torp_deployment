# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for torp_report_generator
#
# Parameter-CSV-filerna bundlas som FALLBACK. Vid körning letar scripten
# (via _common.get_param_file_path) FÖRST i <exe-mapp>/data/parametrar/,
# vilket är den användareditbara platsen som operatören kan uppdatera utan
# rebuild. Om den inte finns används den bundlade kopian under sys._MEIPASS.
# Detta gör att man kan släppa en ny version av .exe-filen utan att tappa
# anpassade leverans-/beställningsfrekvenser - men också att .exe:n fungerar
# på en helt tom installation, eftersom defaultvärdena alltid följer med.

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Fallback-värden inbäddade i .exe; körtiden föredrar
        # <exe-mapp>/data/parametrar/* om dessa finns.
        ('data/parametrar/Beställningsfrekvens.csv', 'data/parametrar'),
        ('data/parametrar/Leveransfrekvens.csv', 'data/parametrar'),
        ('data/parametrar/format.xlsx', 'data/parametrar'),
        # Hela scripts/-mappen, inklusive _common.py som alla scripts importerar.
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
        'sklearn.linear_model',
        'sklearn.linear_model._base',
        'sklearn.linear_model._ridge',
        'sklearn.ensemble',
        'sklearn.ensemble._forest',
        'sklearn.ensemble._gb',
        'sklearn.ensemble._gradient_boosting',
        'sklearn.metrics',
        'sklearn.metrics._regression',
        'sklearn.tree',
        'sklearn.tree._classes',
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
        'openpyxl',
        'openpyxl.utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Uteslut tunga deps som dras in transitivt via Anaconda men som
    # vi INTE använder. Utan dessa excludes hänger PyInstaller-analysen
    # i 15+ minuter och produserar en flera GB stor .exe. Vi kör
    # matplotlib enbart i headless-läge (Agg + PDF) så GUI-backends
    # är onödiga.
    excludes=[
        'tensorflow', 'tensorflow_core', 'tf', 'keras',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'tkinter', '_tkinter', 'Tkinter',
        'IPython', 'jupyter', 'jupyter_client', 'jupyter_core',
        'notebook', 'jupyterlab', 'qtconsole', 'nbconvert', 'nbformat',
        'ipykernel', 'ipywidgets',
        'pytest', 'unittest',
        'sphinx', 'docutils', 'pygments',
        'sqlalchemy', 'psycopg2', 'MySQLdb', 'pymongo',
        'cv2', 'PIL.ImageQt', 'PIL.ImageTk',
        'bokeh', 'holoviews', 'panel', 'altair', 'plotly',
        'sympy', 'numba',
        'tables', 'pyarrow',
        'boto3', 'botocore', 's3transfer',
        'redis', 'pika',
    ],
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
    console=True,  # Set to False for windowed version
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one: 'icon.ico'
)








