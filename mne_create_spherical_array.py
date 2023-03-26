#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create sensor info for spherical pointlike array. Coregistration is taken from the mne
"sample" dataset.

@author: jussi
"""

# %% init
import numpy as np
import pathlib
import mne
from mne.io.constants import FIFF
from mne.transforms import invert_transform, apply_trans, combine_transforms
from mne.preprocessing.maxwell import _sss_basis, _sss_basis_basic, _prep_mf_coils
from mne.forward import _create_meg_coils
import pickle
from mayavi import mlab

from megsimutils.viz import _mlab_points3d, _mlab_quiver3d
from misc import _spherepts_golden, _random_unit, _normalize_columns
from megsimutils.envutils import _ipython_setup
from forward_comp import _sensordata_to_ch_dicts, _sss_basis_nvecs


def _coil_name(coil_type):
    """Return a name for a coil"""
    cstr = str(coil_type)
    ind1 = cstr.find('FIFFV_COIL_') + 11
    ind2 = cstr.find(')')
    return cstr[ind1:ind2]


_ipython_setup()

# use 'sample' for coreg + mri
data_path = pathlib.Path(mne.datasets.sample.data_path())
raw_file = data_path / 'MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_file)
info['bads'] = list()  # reset bads since we do not use measured data

NSENSORS = 1000
ARRAY_RADIUS = 0.12  # array radius in m
FLIP_SENSORS = 0  # how many sensors to flip (90 deg)
COVERAGE_ANGLE = 4 * np.pi  # solid angle coverage of the array
COIL_TYPE = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
# COIL_TYPE = FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2
# COIL_TYPE = FIFF.FIFFV_COIL_VV_MAG_T4
# COIL_TYPE = FIFF.FIFFV_COIL_CTF_GRAD
# COIL_TYPE = FIFF.FIFFV_COIL_BABY_MAG


Sc = _spherepts_golden(NSENSORS, angle=COVERAGE_ANGLE)
Sn = Sc.copy()
Sc *= ARRAY_RADIUS

# perform 90 degree flips for a subset of sensor normals
if FLIP_SENSORS:
    print(f'*** flipping {FLIP_SENSORS} sensors')
    to_flip = np.random.choice(NSENSORS, FLIP_SENSORS, replace=False)
    for k in to_flip:
        flipvec = _random_unit(3)
        Sn[k, :] = np.cross(Sn[k, :], flipvec)

# head-device translation
HEAD_SHIFT = np.array([0, -2e-2, 0.0e-2])
if np.any(HEAD_SHIFT):
    print('*** performing head shift')
head_dev_trans = invert_transform(info['dev_head_t'])
head_dev_trans['trans'][:3, 3] += HEAD_SHIFT
headpos = head_dev_trans['trans'][:3, 3]
dev_head_trans = invert_transform(head_dev_trans)
info['dev_head_t'] = dev_head_trans

# if Sn is "almost" aligned with negative z-axis, we get numerical issues
almost_negz = np.where(1 - Sn.dot([0, 0, -1]) < 1e-2)[0]
Sn[almost_negz, :] = np.array([0, 0, -1])


# %% check array geometry
fig = mlab.figure()
_mlab_points3d(Sc, figure=fig, scale_factor=0.001)
_mlab_quiver3d(Sc, Sn, figure=fig)


# %% create mne sensor data structures
info_ = info.copy()  # make sure to get an unaltered copy, in case we run multiple times
# get coil locations and orientations
coil_types = NSENSORS * [COIL_TYPE]
# apply no rotations to integration points
Iprot = np.zeros(NSENSORS)
sensors_ = list(_sensordata_to_ch_dicts(Sc, Sn, Iprot, coil_types))
info_['chs'] = sensors_
info_['nchan'] = len(sensors_)
info_['ch_names'] = [ch['ch_name'] for ch in info_['chs']]
info_['description'] = f'pointlike array with {NSENSORS} radial sensors'


# %% check SSS basis properties
MAG_SCALING = 1
LIN, LOUT = 16, 3  # use higher basis dim for OPM (sensors closer to head)
# for helmetlike OPM array, ensure that origin is above the helmet rim;
# moving origin below z = 0 causes basis condition number to blow up
sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords

print('using SSS origin: %s' % sss_origin)
exp = {'origin': sss_origin, 'int_order': LIN, 'ext_order': LOUT}
nin = _sss_basis_nvecs(LIN)
nout = _sss_basis_nvecs(LOUT)

coils = _create_meg_coils(info_['chs'], 'accurate')
Su = _sss_basis_basic(exp, coils, mag_scale=MAG_SCALING)
S = _normalize_columns(Su)

Sin = S[:, :nin]
Sout = S[:, nin:]

print('basis dim: Lin=%d Lout=%d' % (LIN, LOUT))
print('condition for S: %g' % np.linalg.cond(S))
print('condition for Sin: %g' % np.linalg.cond(Sin))


# %% save the array data to disk
coilname = _coil_name(COIL_TYPE)
fn = f'RADIAL_N{NSENSORS}_R{ARRAY_RADIUS*1e3:.0f}mm_coverage{COVERAGE_ANGLE / np.pi}pi_{FLIP_SENSORS}flipped_{coilname}.dat'
print('saving %s' % fn)
with open(
    fn,
    'wb',
) as f:
    pickle.dump(info_, f)
