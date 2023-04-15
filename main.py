#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XXX: fill in comments
Simulation code for XXX

@author: jussi
"""


# %% INITIALIZE

from pathlib import Path
import pickle
import scipy
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.transforms import invert_transform, apply_trans
from mne.preprocessing.maxwell import _sss_basis_basic
from mne.forward import _create_meg_coils

from miscutils import _ipython_setup

from mathutils import (
    _prettyprint_xyz,
    _normalize_columns,
    subspace_angles_deg,
)
from forward_comp import (
    _split_leadfield,
    _split_normals,
    _scale_magmeters,
    _scale_array,
    _sss_basis_nvecs,
    _idx_deg_ord,
    _min_norm_pinv,
    _scalarize_src_data,
    _resolution_kernel,
    _spatial_dispersion,
    _node_to_source_index,
    _hemi_slice,
)
from viz import (
    _montage_pysurfer_brain_plots,
    _montage_mlab_trimesh,
    _make_array_tri,
)
from paths import DATA_PATH


_ipython_setup()
plt.rcParams['figure.dpi'] = 150
assert DATA_PATH.is_dir()
FIG_DIR = DATA_PATH  # where to put the generated figures

# head position shift to set headpos (independent of source shift)
HEAD_SHIFT = np.array([0, -0e-2, -0e-2])
# to use the default sensor data (Vectorview 306-ch), supply None
# otherwise give a filename for a new sensor file
LOAD_ARRAY_DATA = Path('RADIAL_N1000_R120mm_coverage4.0pi_POINT_MAGNETOMETER.dat')

# restrict to gradiometers or magnetometers
SENSOR_TYPE = 'all'
# magnetometer signal scaling relative to gradiometers
MAG_SCALING = 50
# scale array (sensor locations)
ARRAY_SCALING = False
# use regularized SSS basis for L-decompositions
REGULARIZE = False

# whether to fix source orientations according to cortex or use free orientations
FIX_ORI = True
# source spacing; normally 'oct6', 'oct4' for sparse source space
SRC_SPACING = 'oct6'
# whether to use BEM for forward computations; False for sphere model
USE_BEM = True
# BEM: minimum distance of sources from inner skull surface (in mm)
BEM_MINDIST = 5
# origin for sphere model (head coords); also used as nominal "head origin"
# for source distance calculations
HEAD_ORIGIN = np.array((0.0, 0.0, 0.04))
# which hemisphere and surface to use for visualization
HEMI = 'lh'
SURF = 'white'
# index of a representative source (tangential on left hemi); this depends on source space
REPR_SOURCE = 1251

# use 'sample' subject for coreg + mri
data_path = Path(mne.datasets.sample.data_path())  # path to fiff data
subjects_dir = data_path / 'subjects'
raw_file = data_path / 'MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_file)

info['bads'] = list()  # reset bads
# restrict info to MEG channels
if SENSOR_TYPE == 'mag':
    meg_sensors = 'mag'
elif SENSOR_TYPE == 'grad':
    meg_sensors = 'grad'
elif SENSOR_TYPE == 'all':
    meg_sensors = True
else:
    raise RuntimeError('Invalid MEG sensors paramter')

meg_indices = mne.pick_types(info, meg=meg_sensors, eeg=False)
mne.pick_info(info, meg_indices, copy=False)

raw = mne.io.read_raw_fif(raw_file)
head_mri_trans_file = data_path / 'MEG/sample/sample_audvis_raw-trans.fif'
head_mri_trans = mne.read_trans(head_mri_trans_file)
bem_file = subjects_dir / 'sample/bem/sample-5120-5120-5120-bem-sol.fif'
subject = 'sample'

print('\n')

if LOAD_ARRAY_DATA is not None:
    print(f'using saved array: {LOAD_ARRAY_DATA}')
    array_name = LOAD_ARRAY_DATA.stem  # name without extension
    with open(DATA_PATH / LOAD_ARRAY_DATA, 'rb') as f:
        info = pickle.load(f)
else:
    array_name = 'VV-306'
    print('using default VV-306 array')

if SENSOR_TYPE != 'all':
    print(f'*** restricting to MEG sensor type: {SENSOR_TYPE}')

print(f'using mag scaling: {MAG_SCALING:g}')

# do the head translation
if np.any(HEAD_SHIFT):
    print('*** translating the head...')
    head_dev_trans = invert_transform(info['dev_head_t'])
    head_dev_trans['trans'][:3, 3] += HEAD_SHIFT
    headpos = head_dev_trans['trans'][:3, 3]
    dev_head_trans = invert_transform(head_dev_trans)
    info['dev_head_t'] = dev_head_trans
    headpos_str = _prettyprint_xyz(headpos)

# array scaling
if ARRAY_SCALING and ARRAY_SCALING != 1.0:
    print('*** scaling array')
    _scale_array(info, ARRAY_SCALING)

print(f'using {SRC_SPACING} source spacing')

if USE_BEM:
    print('using a BEM model:')
    print(f'{BEM_MINDIST=} mm')
else:
    print('using a sphere model')

if FIX_ORI:
    print('using cortically constrained source orientations')
    FIX_ORI_DESCRIPTION = 'fixed_ori'
else:
    print('using free source orientations')
    FIX_ORI_DESCRIPTION = 'free_ori'

print(f'using hemi: {HEMI}')
# source related data will be computed by-hemisphere and stored into dicts indexed by hemi
# this is mostly because our visualization requires hemi-specific data
hemi_to_ind = {'lh': 0, 'rh': 1}
HEMI_IND = hemi_to_ind[HEMI]


# %% DEFINE source spaces and forward computations.
#
# If the solution is unconstrained (FIX_ORI=False), the leadfield and the
# src_normal matrix will have 3*Nsrc elements due to the 3 orthogonal dipoles at
# each source location. The number of source locations (src_coords) remains the same.

# first create the volume source space
src_cort = mne.setup_source_space(
    subject, spacing=SRC_SPACING, subjects_dir=subjects_dir, add_dist=False
)

# compute the forwards
if USE_BEM:
    model = mne.make_bem_model(
        subject=subject, ico=4, conductivity=[0.3], subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)
    fwd_cort = mne.make_forward_solution(
        info,
        head_mri_trans_file,
        src_cort,
        bem,
        eeg=False,
        mindist=BEM_MINDIST,
    )
else:
    # default sphere model
    sphere = mne.make_sphere_model(r0=HEAD_ORIGIN, head_radius=None)
    fwd_cort = mne.make_forward_solution(
        info,
        head_mri_trans_file,
        src_cort,
        sphere,
        eeg=False,
    )

# fixed orientation fwd
if FIX_ORI:
    fwd_cort = mne.convert_forward_solution(
        fwd_cort, surf_ori=True, force_fixed=True, copy=True
    )

# make hemi-indexed dicts
node_coords = dict()  # (N_nodes, 3) source coordinates
node_dists = dict()  # (N_nodes,) dist from origin
src_normals = dict()  # (N_nodes, 3) for fixed orientations, or (3*N_nodes, 3) for free
src_node_inds = dict()  # (N_nodes, 1) indices into full discretized cortex volume
nsrc_valid = dict()
nverts = dict()
# for these, shapes are (N_nodes, 3)/(3*N_nodes, 3) for fixed/free
leads = _split_leadfield(fwd_cort)
src_normals = _split_normals(fwd_cort)

for hemi_ind in [0, 1]:
    inuse = fwd_cort['src'][hemi_ind]['inuse'].astype(bool)
    src_node_inds[hemi_ind] = np.where(inuse)[0]
    node_coords[hemi_ind] = fwd_cort['src'][hemi_ind]['rr'][inuse, :]
    nverts[hemi_ind] = node_coords[hemi_ind].shape[0]
    node_dists[hemi_ind] = np.linalg.norm(node_coords[hemi_ind] - HEAD_ORIGIN, axis=1)
    nsrc_valid[hemi_ind] = leads[hemi_ind].shape[1]

HEMI_SLICE = _hemi_slice(HEMI_IND, nsrc_valid)  # a slice object for the chosen hemi

# define some array vars for the full cortical volume, to facilitate computations
leads_sc = {hemi: _scale_magmeters(leads[hemi], info, MAG_SCALING) for hemi in [0, 1]}
leads_all_sc = np.hstack(list(leads_sc.values()))  # full leadfield
node_coords_all = np.concatenate((node_coords[0], node_coords[1]))
nverts_all = sum(nverts.values())

# define some hemi-specific vars for convenience
nsrc_valid_thishemi = nsrc_valid[HEMI_IND]
node_coords_thishemi = node_coords[HEMI_IND]
node_dists_thishemi = node_dists[HEMI_IND]
src_normals_thishemi = src_normals[HEMI_IND]
nverts_thishemi = nverts[HEMI_IND]
leads_thishemi = leads[HEMI_IND]
leads_thishemi_sc = leads_sc[HEMI_IND]  # leadfield for this hemi
lead_norms_thishemi = np.linalg.norm(leads_thishemi, axis=0)

#  distance matrix
if FIX_ORI:
    src_dij_thishemi = scipy.spatial.distance_matrix(
        node_coords_thishemi, node_coords_thishemi
    )
    src_dij_all = scipy.spatial.distance_matrix(node_coords_all, node_coords_all)
else:
    node_coords_thishemi_redundant = np.empty((nsrc_valid_thishemi, 3))
    node_coords_thishemi_redundant[0::3, :] = node_coords_thishemi
    node_coords_thishemi_redundant[1::3, :] = node_coords_thishemi
    node_coords_thishemi_redundant[2::3, :] = node_coords_thishemi
    src_dij_thishemi = scipy.spatial.distance_matrix(
        node_coords_thishemi_redundant, node_coords_thishemi_redundant
    )
    # XXX: src_dij_all is not yet computed for free ori


# %% COMPUTE multipole basis, multipole-based leadfields etc.
#
# NB: magnetometers are scaled by factor of MAG_SCALING in the basis
# some heuristics for the basis dimension and origin
_array_name = str(array_name).lower()
head_dev_trans = invert_transform(info['dev_head_t'])
print(f'array: {array_name}')
if array_name == 'VV-306':
    LIN, LOUT = 9, 3
    # use head model origin
    sss_origin = apply_trans(head_dev_trans, HEAD_ORIGIN)
elif 'radial' in _array_name:  # radial-spherical
    LIN, LOUT = 16, 3
    sss_origin = np.array([0.0, -0.0, 0.0])  # origin of device coords
else:
    raise RuntimeError('Unknown array')

print(f'using SSS origin: {sss_origin}')
exp = {'origin': sss_origin, 'int_order': LIN, 'ext_order': LOUT}
nin = _sss_basis_nvecs(LIN)
nout = _sss_basis_nvecs(LOUT)
print(f'{nin} internal basis vectors')

# basic (more comprehensible) SSS basis computation; takes a list of coil defs
# from _create_meg_coils
# this seems to be fast enough
coils = _create_meg_coils(info['chs'], 'accurate')
Su = _sss_basis_basic(exp, coils, mag_scale=MAG_SCALING)
S = _normalize_columns(Su)
Sin = S[:, :nin]
Sout = S[:, nin:]

# this can be set to optimize basis by dropping certain basis vectors
Sin_drop_inds = list()

print(f'basis dim: Lin={int(LIN)} Lout={int(LOUT)}')
with np.printoptions(precision=3):
    print(f'subspace angles Sin vs Sout: {subspace_angles_deg(Sin, Sout)}')
print(f'condition of S: {np.linalg.cond(S):g}')
print(f'condition of Sin: {np.linalg.cond(Sin):g}')

# compute L for each matrix element
basisvec_L = np.array([_idx_deg_ord(k)[0] for k in range(nin)])

# corresponding multipole-based leadfield
xin_leads_all = np.squeeze(np.linalg.pinv(Sin) @ leads_all_sc)
print(f'condition of leadfield: {np.linalg.cond(leads_all_sc):g}')
print(f'condition of multipole-based leadfield: {np.linalg.cond(xin_leads_all):g}')

# source node distances to SSS origin
head_dev_trans = invert_transform(info['dev_head_t'])
node_coords_thishemi_dev = apply_trans(head_dev_trans, node_coords_thishemi)
node_origin_dists = np.linalg.norm(node_coords_thishemi_dev - sss_origin, axis=1)


# %% COMPUTE resolution kernel and spatial dispersion.


def _compute_res_kernels(tikhonov_lambda):
    sds_multipole = dict()
    xin_res_kernels = dict()
    #
    # FIT_REDUCED_BASES=False: decompose the sensor-based leadfield into the full
    # range of available L components first, then pick the components for each L in
    # turn. In principle, this has the advantage of avoiding aliasing, since the
    # initial fit includes all spatial freqs. If L is very high, the initial
    # decomposition may require regularization.
    #
    # FIT_REDUCED_BASES=True: fit leadfield onto the reduced-order basis at each L.
    #
    FIT_REDUCED_BASES = True
    RES_METHOD = 'tikhonov'  # how to regularize when computing the resolution kernels
    xin_lead_conds = list()
    xin_leads = list()
    print('computing multipole-based resolution kernels...')
    for L in range(1, LIN + 1):
        theLs = list(range(1, L + 1))
        # the component indices to include
        include = np.where(np.isin(basisvec_L, theLs))[0]
        if FIT_REDUCED_BASES:
            Sin_red = Sin[:, include]
            xin_leads_filt = np.squeeze(np.linalg.pinv(Sin_red) @ leads_all_sc)
        else:
            xin_leads_filt = xin_leads_all[include]
        xin_leads.append(xin_leads_filt)
        xin_lead_conds.append(np.linalg.cond(xin_leads_filt))
        print(f'{L=}, leadfield condition: {np.linalg.cond(xin_leads_filt):g}')
        res_kernel = _resolution_kernel(
            xin_leads_filt,
            method=RES_METHOD,
            tikhonov_lambda=tikhonov_lambda,
        )
        xin_res_kernels[L] = res_kernel
        sds_multipole[L] = _spatial_dispersion(res_kernel, src_dij_all)

    # compute resolution kernel and spatial dispersion for sensor-based leadfield
    print('computing sensor-based resolution kernel...')
    sensor_res_kernel = _resolution_kernel(
        leads_all_sc,
        method=RES_METHOD,
        tikhonov_lambda=tikhonov_lambda,
    )
    sds_sensor = _spatial_dispersion(sensor_res_kernel, src_dij_all)
    print('done')
    return {'sds_multipole': sds_multipole, 'sds_sensor': sds_sensor,
            'xin_res_kernels': xin_res_kernels, 'sensor_res_kernel': sensor_res_kernel}

def _compute_res_kernels_vs_lambda(lambdas):
    # compute resolution kernel and spatial dispersion for sensor-based leadfield as
    # a function of Tikhonov lambda
    print('computing sensor-based resolution kernels with varying regularization...')
    sds_lambda = dict()
    for _lambda in lambdas:
        res_kernel = _resolution_kernel(
            leads_all_sc, method='tikhonov', tikhonov_lambda=_lambda
        )
        sds_lambda[_lambda] = _spatial_dispersion(res_kernel, src_dij_all)
    print('done')
    return sds_lambda



# %% FIGURE 4: Plot spatial dispersion vs lambda and L.
sds_multipole = _compute_res_kernels(lambda=0)['sds_multipole']
lambdas = 10.0 ** np.arange(-5, -12, -1)
sds_lambda = _compute_res_kernels_vs_lambda(lambdas)

outfn = FIG_DIR / 'mean_PSF_SD_vs_L_and_lambda.png'
Lvals = list(range(1, LIN + 1))
REDUCER_FUN = np.mean
YLABEL = 'Mean PSF spatial dispersion (mm)'
YTICKS = list(range(20, 90, 10))
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.semilogx(
    lambdas,
    [1e3 * REDUCER_FUN(sds_lambda[l][:]) for l in lambdas],
    label='sensor-based',
)
ax1.set_xlabel(r'$\lambda$ (sensor-based inverse)')
fig.supylabel(YLABEL)
ax1.set_yticks(YTICKS)
ax1.invert_xaxis()
ax2.set_xticks(Lvals)
ax2.set_xlabel('$L$ (multipole-based inverse)')
ax2.plot(
    Lvals, [1e3 * REDUCER_FUN(sds_multipole[L][:]) for L in Lvals], 'r', label='multipole-based'
)
ax2.set_yticks(YTICKS)
plt.tight_layout()
plt.savefig(outfn)


# %% FIGURE 2: plot PSF spatial dispersion as function of Lin.
#
kernels_data = _compute_res_kernels(lambda=1e-11)
sds_sensor = kernels_data['sds_sensor']
sds_multipole = kernels_data['sds_multipole']

N_SKIP = 2  # reduce n of plots by stepping the index
MIN_LIN = 1
MAX_LIN = 13
SURF = 'inflated'
outfn = FIG_DIR / f'dispersion_cortexplot_{FIX_ORI_DESCRIPTION}_{array_name}.png'
# restrict dispersion to current hemi
sds_thishemi = [sd[HEMI_SLICE] for sd in sds_multipole.values()]
titles = list()
src_datas = list()

# multipole-based data
for L in range(MIN_LIN, MAX_LIN + 1, N_SKIP):
    src_data = sds_multipole[L][HEMI_SLICE]
    # scalarize and convert m->mm
    src_data = 1e3 * _scalarize_src_data(src_data, nverts_thishemi, reducer_fun=np.mean)
    src_datas.append(src_data)
    title = f'L=1..{L}'
    titles.append(title)

# sensor-based data
src_data = sds_sensor[HEMI_SLICE]
# scalarize and convert m->mm
src_data = 1e3 * _scalarize_src_data(src_data, nverts_thishemi, reducer_fun=np.mean)
title = 'sensor'
src_datas.append(src_data)
titles.append(title)

fmin, fmax = 0, 60
_montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    src_datas,
    titles,
    src_node_inds[HEMI_IND],
    HEMI,
    outfn,
    surf=SURF,
    frange=(fmin, fmax),
    ncols_max=4,
    colormap='plasma_r',
    colorbar_nlabels=4,
    title_width=0.3,
)


# %% FIGURE 1: plot single source PSF as function of Lin, no regularization.
#
kernels_data = _compute_res_kernels(lambda=1e-11)
xin_res_kernels = kernels_data['xin_res_kernels']
sensor_res_kernel = kernels_data['sensor_res_kernel']
sds_sensor = kernels_data['sds_sensor']
sds_multipole = kernels_data['sds_multipole']

N_SKIP = 2  # reduce n of plots by stepping the index
MIN_LIN = 1
MAX_LIN = 13  # max LIN value to use
COLOR_THRES = None  # don't show colors below given value
SURF = 'inflated'  # which surface; usually either 'white' or 'inflated'
# NOTE: source indices are global (index the complete leadfield, not a hemi)
SRC_IND = REPR_SOURCE
SRC_IND = _node_to_source_index(SRC_IND, FIX_ORI)
# frange = 0, .05  # global fixed
frange = 'separate'  # individual auto
if not FIX_ORI:
    SRC_IND = SRC_IND[1]  # pick a single orientation
NCOLS_MAX = 4
outfn = (
    FIG_DIR
    / f'psf_cortexplot_{FIX_ORI_DESCRIPTION}_{array_name}_LIN{MAX_LIN}_surf_{SURF}.png'
)

titles = list()
src_datas = list()

# multipole-based data
for L in range(MIN_LIN, MAX_LIN + 1, N_SKIP):
    # for each L, slice correct row from resolution matrix, restrict to current hemi
    src_data = np.abs(xin_res_kernels[L][SRC_IND, HEMI_SLICE])
    src_data = _scalarize_src_data(src_data, nverts_thishemi)
    src_datas.append(src_data)
    sd = sds_multipole[L][SRC_IND] * 1e3
    title = f'L=1..{L}, SD={sd:.0f} mm'
    titles.append(title)

# sensor-based data
src_data = np.abs(sensor_res_kernel[SRC_IND, HEMI_SLICE])
sd = sds_sensor[SRC_IND] * 1e3
title = f'sensor, SD={sd:.0f} mm'
src_datas.append(src_data)
titles.append(title)

_montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    src_datas,
    titles,
    src_node_inds[HEMI_IND],
    HEMI,
    outfn,
    thresh=COLOR_THRES,
    smoothing_steps=None,
    surf=SURF,
    frange=frange,
    ncols_max=NCOLS_MAX,
    colormap='plasma',
    colorbar_nlabels=4,
    title_width=0.7,
    do_colorbar=False,
)


# %% FIGURE 3: plot single source PSF as function of Lin, regularization with lambda = 1e-8
kernels_data = _compute_res_kernels(lambda=1e-8)   # XXX: rename?
xin_res_kernels = kernels_data['xin_res_kernels']
sensor_res_kernel = kernels_data['sensor_res_kernel']
sds_sensor = kernels_data['sds_sensor']
sds_multipole = kernels_data['sds_multipole']

#
N_SKIP = 2  # reduce n of plots by stepping the index
MIN_LIN = 1
MAX_LIN = 13  # max LIN value to use
COLOR_THRES = None  # don't show colors below given value
SURF = 'inflated'  # which surface; usually either 'white' or 'inflated'
# NOTE: source indices are global (index the complete leadfield, not a hemi)
SRC_IND = REPR_SOURCE
SRC_IND = _node_to_source_index(SRC_IND, FIX_ORI)
frange = 'separate'  # individual auto
if not FIX_ORI:
    SRC_IND = SRC_IND[1]  # pick a single orientation
NCOLS_MAX = 4
outfn = (
    FIG_DIR
    / f'psf_cortexplot_regu_{FIX_ORI_DESCRIPTION}_{array_name}_LIN{MAX_LIN}_surf_{SURF}.png'
)

titles = list()
src_datas = list()

# multipole-based data
for L in range(MIN_LIN, MAX_LIN + 1, N_SKIP):
    # for each L, slice correct row from resolution matrix, restrict to current hemi
    src_data = np.abs(xin_res_kernels[L][SRC_IND, HEMI_SLICE])
    src_data = _scalarize_src_data(src_data, nverts_thishemi)
    src_datas.append(src_data)
    sd = sds_multipole[L][SRC_IND] * 1e3
    title = f'L=1..{L}, SD={sd:.0f} mm'
    titles.append(title)

# sensor-based data
src_data = np.abs(sensor_res_kernel[SRC_IND, HEMI_SLICE])
sd = sds_sensor[SRC_IND] * 1e3
title = f'sensor, SD={sd:.0f} mm'
src_datas.append(src_data)
titles.append(title)

_montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    src_datas,
    titles,
    src_node_inds[HEMI_IND],
    HEMI,
    outfn,
    thresh=COLOR_THRES,
    smoothing_steps=None,
    surf=SURF,
    frange=frange,
    ncols_max=NCOLS_MAX,
    colormap='plasma',
    colorbar_nlabels=4,
    title_width=0.7,
    do_colorbar=False,
)


# %% FIGURE 5: plot lead field SVD vectors on array trimesh.
#
outfn = FIG_DIR / 'svd_basis_trimesh.png'

U, Sigma, V = np.linalg.svd(leads_all_sc)

inds, locs, tri = _make_array_tri(info)

src_datas = list()
titles = list()
for k in range(20):
    src_datas.append(U[:, k])
    title = f'k={k+1}'.ljust(6)
    titles.append(title)
_montage_mlab_trimesh(locs, tri, src_datas, titles, outfn, ncols_max=5, distance=0.5)


# %% FIGURE 6:  plot some VSHs on array trimesh.
#
outfn = FIG_DIR / 'vsh_basis_trimesh.png'

inds, locs, tri = _make_array_tri(info)

src_datas = list()
titles = list()
for ind in range(20):
    src_datas.append(Sin[:, ind])
    L, m = _idx_deg_ord(ind)
    # title = f'{L=}, {m=}'
    title = f'({L}, {m})'
    titles.append(title)
_montage_mlab_trimesh(locs, tri, src_datas, titles, outfn, ncols_max=5, distance=0.5)


# %% FIGURE 7: L-dependent MNP solution with noise.
#
# Pick a single source from the leadfield matrix, add noise and do MNP in the
# multipole domain.

outfn = FIG_DIR / 'inverse_vs_SNR.png'

REGU_METHOD = 'tikhonov'
LAMBDA = 0
PINV_RCOND = 1e-20
FIG_BG_COLOR = (0.3, 0.3, 0.3)
FIGSIZE = (400, 300)

SNR = 2  # desired SNR (defined as ratio of signal vector norms)
SNR_VALS = [1, 2, 5, 10]
LIN_VALS = [6, 7, 8, 9]
NCOLS_MAX = len(LIN_VALS)
frange = 'separate'  # individual auto
SRC_IND = REPR_SOURCE
DO_COLORBAR = False

inverses = list()
titles = list()

for SNR in SNR_VALS:
    np.random.seed(10)
    src_lead_sensors = leads_thishemi_sc[:, SRC_IND]
    noisevec = np.random.randn(*src_lead_sensors.shape)
    noisevec /= np.linalg.norm(noisevec) * SNR
    noisevec *= np.linalg.norm(src_lead_sensors)
    src_lead_sensors_noisy = src_lead_sensors + noisevec

    for L in LIN_VALS:
        # select multipole components to include according to L
        theLs = list(range(1, L + 1))
        include = np.where(np.isin(basisvec_L, theLs))[0]
        Sin_red = Sin[:, include]
        # L-limited multipole transformation of leadfield
        xin_leads_red = np.squeeze(np.linalg.pinv(Sin_red) @ leads_all_sc)
        lead_cond = np.linalg.cond(xin_leads_red)
        # multipole components for noisy forward field
        lead_cond = np.linalg.cond(xin_leads_red) / np.linalg.cond(xin_leads_red, -2)
        scond = np.linalg.cond(Sin_red) / np.linalg.cond(Sin_red, -2)
        print(f'{L=} condition leadfield: {lead_cond:.2e} Sin: {scond} {SNR=}')
        xin_source_red = np.squeeze(np.linalg.pinv(Sin_red) @ src_lead_sensors_noisy)
        inv_sol_full = _min_norm_pinv(
            xin_leads_red,
            xin_source_red,
            method=REGU_METHOD,
            tikhonov_lambda=LAMBDA,
            rcond=PINV_RCOND,
        )
        inv_sol = inv_sol_full[HEMI_SLICE]
        inv_sol = _scalarize_src_data(inv_sol, nverts_thishemi)
        inverses.append(inv_sol)
        # Lstr = str(L).ljust(2)
        titles.append(f'{L=}, {SNR=}')

_montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    inverses,
    titles,
    src_node_inds[HEMI_IND],
    HEMI,
    outfn,
    frange=frange,
    ncols_max=NCOLS_MAX,
    colormap='plasma',
    colorbar_nlabels=4,
    title_width=0.4,
    do_colorbar=DO_COLORBAR,
)
