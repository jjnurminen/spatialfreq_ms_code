#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misc utilities.

"""
from pathlib import Path
import subprocess
import os
import tempfile
import numpy as np

from paths import MONTAGE_PATH


def _named_tempfile(suffix=None):
    """Return a name for a temporary file.
    Does not open the file. Cross-platform. Replaces tempfile.NamedTemporaryFile
    which behaves strangely on Windows.
    """
    if suffix is None:
        suffix = ''
    elif suffix[0] != '.':
        raise ValueError('Invalid suffix, must start with dot')
    return os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + suffix)


def _montage_figs(fignames, montage_fn, ncols_max=None):
    """Montages a bunch of figures into montage_fname.

    fignames is a list of figure filenames.
    montage_fn is the resulting montage name.
    ncols_max defines max number of columns for the montage.
    """
    if ncols_max is None:
        ncols_max = 4
    if not Path(MONTAGE_PATH).exists():
        raise RuntimeError('montage binary not found, cannot montage files')
    # set montage geometry
    nfigs = len(fignames)
    geom_cols = ncols_max
    geom_rows = int(np.ceil(nfigs / geom_cols))  # figure out how many rows we need
    geom_str = f'{geom_cols}x{geom_rows}'
    MONTAGE_ARGS = ['-geometry', '+0+0', '-tile', geom_str]
    # compose a list of arguments
    montage_fn = str(montage_fn)  # so we can accept pathlib args too
    theargs = [MONTAGE_PATH] + MONTAGE_ARGS + fignames + [montage_fn]
    print('running montage command %s' % ' '.join(theargs))
    subprocess.call(theargs)  # use call() to wait for completion


def _ipython_setup(enable_reload=False):
    """Set up some IPython parameters, if we're running in IPython"""
    try:
        __IPYTHON__
    except NameError:
        return
    from IPython import get_ipython

    ip = get_ipython()
    ip.magic("gui qt5")  # needed for mayavi plots
    # ip.magic("matplotlib qt")  # do mpl plots in separate windows
    if enable_reload:
        ip.magic("reload_ext autoreload")  # these will enable module autoreloading
        ip.magic("autoreload 2")
