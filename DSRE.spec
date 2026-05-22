# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

for mod in (
    "numpy",
    "scipy",
    "librosa",
    "numba",
    "llvmlite",
    "resampy",
    "soundfile",
    "send2trash",
    "audioread",
    "pooch",
    "soxr",
    "joblib",
    "sklearn",
    "threadpoolctl",
    "lazy_loader",
    "msgpack",
    "decorator",
    "cffi",
    "pyloudnorm",
    "future",
    "mutagen",
):
    d, b, h = collect_all(mod)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += [
    "scipy.signal",
    "scipy.signal._peak_finding_utils",
    "scipy.special.cython_special",
    "soundfile",
    "_soundfile",
]

a = Analysis(
    ['DSRE.py'],
    pathex=[],
    binaries=binaries,
    datas=datas + [
        ('logo.ico', '.'),
        ('_internal/ffmpeg/fpcalc.exe', 'ffmpeg'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DSRE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon='logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='DSRE',
    contents_directory='_internal',
)
