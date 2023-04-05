#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings

@author: jussi
"""
from pathlib import Path
import platform

# the path where the necessary data files reside
DATA_PATH = Path.home() / 'repos/spatialfreq_ms_code'

# path to the montage binary
if platform.system() == 'Linux':
    MONTAGE_PATH = '/usr/bin/montage'
else:
    MONTAGE_PATH = 'C:/Program Files/ImageMagick-7.0.10-Q16/montage.exe'
