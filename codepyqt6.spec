# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

a = Analysis(
    ['codepyqt6.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('creditcard.csv', '.'),
        ('design.qss', '.'),
        ('icon.jpeg', '.'),
        ('image.webp', '.')
    ],
    hiddenimports=collect_submodules('PyQt6') + collect_submodules('Classification_Technics'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide2', 'PySide6'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='codepyqt6',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='codepyqt6',
)
