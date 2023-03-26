#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study mne inverse.

@author: jussi
"""

# %% INITIALIZE
import mne
from mne.io.constants import FIFF
import numpy as np
from mayavi import mlab
from surfer import Brain
from mne.transforms import invert_transform, apply_trans, _deg_ord_idx
from mne.preprocessing.maxwell import _sss_basis, _sss_basis_basic, _prep_mf_coils
from mne.forward import _create_meg_coils
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import scipy
import trimesh

from megsimutils.viz import (
    _mlab_quiver3d,
    _mlab_points3d,
    _mlab_trimesh,
    _mlab_colorblobs,
)
from megsimutils.envutils import _ipython_setup

from misc import (
    _vector_angles,
    _prettyprint_xyz,
    _normalize_columns,
    subspace_angles_deg,
    _find_points_in_range,
    _unit_impulse,
    _moore_penrose_pseudoinverse,
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
    _decompose_sigvec,
    _limit_L,
    _resolution_kernel,
    _spatial_dispersion,
    _focality,
    _node_to_source_index,
)
from viz import _montage_pysurfer_brain_plots, _montage_mlab_brain_plots


_ipython_setup(enable_reload=True)

homedir = Path.home()
projectpath = homedir / 'projects/spatialfreq_ms_code'  # where the code resides
assert projectpath.is_dir()
figuredir = projectpath  # where to put the figures
assert figuredir.is_dir()

# adjustable parameters
# head position shift to set headpos (independent of source shift)
HEAD_SHIFT = np.array([0, -0e-2, -0e-2])
# whether to replace default sensor data (306-ch) with a new geometry
LOAD_ALTERNATE_ARRAY = True
if LOAD_ALTERNATE_ARRAY:
    # choose array
    alt_array_name = Path('OPM_TRIPLESENSORS_306.dat')
    alt_array_name = Path('QUSPIN_ZFOPM_MAG2_400chan.dat')
    alt_array_name = Path('Compumedics_177.dat')
    alt_array_name = Path('OPM_HELMET400_NEW.dat')
    # alt_array_name = Path('OPM_CORTEX1400_FULL_SCALED.dat')
    # alt_array_name = Path('RADIAL_N1000_R20cm.dat')
    # alt_array_name = Path('RADIAL_N1000_R170mm_POINT_MAGNETOMETER.dat')
    # alt_array_name = Path('RADIAL_N1000_R150mm_VV_MAG_T4.dat')
    #alt_array_name = Path('RADIAL_N1000_R120mm_POINT_MAGNETOMETER.dat')
    # alt_array_name = Path('RADIAL_N1000_R100mm_coverage4.0pi_POINT_MAGNETOMETER.dat')
    # alt_array_name = Path('RADIAL_N1000_R120mm_coverage3.0pi_POINT_MAGNETOMETER.dat')
    # alt_array_name = Path('OPM_CORTEX1400_POINTLIKE.dat')
    # alt_array_name = Path('RADIAL_N2000_R10cm_0flipped.dat')
    # alt_array_name = Path('RADIAL_N10000_R75cm_0flipped.dat')
    # alt_array_name = Path('BARBUTE_500_480_12cm_1.75pi.dat')
    # alt_array_name = Path('OPM_HELMET1400.dat')
    # alt_array_name = Path('RADIAL_N1000_R12cm_POINT_MAGNETOMETER.dat')
    #alt_array_name = Path('RADIAL_N10000_R120mm_coverage4.0pi_POINT_MAGNETOMETER.dat')
    #alt_array_name = Path(
    #    'RADIAL_N10000_R100mm_coverage4.0pi_6666flipped_POINT_MAGNETOMETER.dat'
    #)
    alt_array_name = Path('RADIAL_N1000_R120mm_coverage4.0pi_POINT_MAGNETOMETER.dat')
    # alt_array_name = Path('RADIAL_N5000_R150mm_coverage4.0pi_0flipped_POINT_MAGNETOMETER.dat')

else:
    alt_array_name = None

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
surf = 'white'

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

if LOAD_ALTERNATE_ARRAY:
    print(f'using saved array: {alt_array_name}')
    array_name = alt_array_name.stem  # name without extension
    with open(projectpath / alt_array_name, 'rb') as f:
        info = pickle.load(f)
else:
    array_name = 'VV-306'
    print('using default VV-306 array')

if SENSOR_TYPE != 'all':
    print(f'*** restricting to MEG sensor type: {SENSOR_TYPE}')

print('using mag scaling: %g' % MAG_SCALING)

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


# %% SOURCE spaces and forward computations
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
# for these shapes are (N_nodes, 3)/(3*N_nodes, 3) for fixed/free
leads = _split_leadfield(fwd_cort)
src_normals = _split_normals(fwd_cort)

for hemi_ind in [0, 1]:
    inuse = fwd_cort['src'][hemi_ind]['inuse'].astype(bool)
    src_node_inds[hemi_ind] = np.where(inuse)[0]
    node_coords[hemi_ind] = fwd_cort['src'][hemi_ind]['rr'][inuse, :]
    nverts[hemi_ind] = node_coords[hemi_ind].shape[0]
    node_dists[hemi_ind] = np.linalg.norm(node_coords[hemi_ind] - HEAD_ORIGIN, axis=1)
    nsrc_valid[hemi_ind] = leads[hemi_ind].shape[1]

# define some hemi-specific vars to reduce need for typing
nsrc_valid_thishemi = nsrc_valid[HEMI_IND]
node_coords_thishemi = node_coords[HEMI_IND]
node_dists_thishemi = node_dists[HEMI_IND]
src_normals_thishemi = src_normals[HEMI_IND]
nverts_thishemi = nverts[HEMI_IND]
leads_thishemi = leads[HEMI_IND]
leads_thishemi_sc = _scale_magmeters(leads_thishemi, info, MAG_SCALING)
lead_norms_thishemi = np.linalg.norm(leads_thishemi, axis=0)

# define some source nodes for testing
try:
    test_nodes = dict()
    # these sources are near the midline (small x)
    test_nodes['superficial_z'] = _find_points_in_range(
        node_coords_thishemi, ((-0.02, 0.02), (0.015, 0.025), (0.12, 0.14))
    )[0]
    test_nodes['deep_z'] = _find_points_in_range(
        node_coords_thishemi, ((-0.02, 0.02), (0.015, 0.025), (0.06, 0.065))
    )[0]
    test_nodes['midrange_z'] = _find_points_in_range(
        node_coords_thishemi, ((-0.02, 0.02), (0.015, 0.025), (0.08, 0.09))
    )[0]
    # lateral source (NB: specific to left hemi!)
    test_nodes['lateral'] = _find_points_in_range(
        node_coords_thishemi, ((-0.06, -0.05), (0.015, 0.025), (0.06, 0.065))
    )[0]
except IndexError:
    print('warning: some test nodes were not found')

if FIX_ORI:
    test_sources = {
        key: _unit_impulse(nsrc_valid_thishemi, node)
        for key, node in test_nodes.items()
    }
else:
    test_sources = dict()
    # generic (1,1,1) vector for all sources
    for key in test_nodes:
        ind = 3 * test_nodes[key]  # since we have 3 orientation elements for each node
        srcvec = np.zeros(nsrc_valid_thishemi)
        # set some specific orientations
        if key == 'superficial_z':
            srcvec[ind : ind + 3] = np.array([-0.14288205, -0.8898686, 0.43326493])
        else:
            srcvec[ind : ind + 3] = 1
        test_sources[key] = srcvec

# distance matrix and resolution kernel
if FIX_ORI:
    src_dij_thishemi = scipy.spatial.distance_matrix(
        node_coords_thishemi, node_coords_thishemi
    )
else:
    node_coords_thishemi_redundant = np.empty((nsrc_valid_thishemi, 3))
    node_coords_thishemi_redundant[0::3, :] = node_coords_thishemi
    node_coords_thishemi_redundant[1::3, :] = node_coords_thishemi
    node_coords_thishemi_redundant[2::3, :] = node_coords_thishemi
    src_dij_thishemi = scipy.spatial.distance_matrix(
        node_coords_thishemi_redundant, node_coords_thishemi_redundant
    )


# %% PLOT alignment of sources vs. sensors
# highlight sources closest to sensor integration points & plot distance
# histogram
pt_scale = 0.3e-2
src_color = (0.8, 0, 0)
n_closest = 100
# transform src coords into device frame
node_coords_all = np.concatenate((node_coords[0], node_coords[1]))
head_dev_trans = invert_transform(info['dev_head_t'])
node_coords_all_dev = apply_trans(head_dev_trans, node_coords_all)
ips = _prep_mf_coils(info)[0]

node_sensor_dists = list()
for node_ind in range(node_coords_all_dev.shape[0]):
    node_loc = node_coords_all_dev[node_ind, :]
    dr = ips - node_loc
    _mindist = np.linalg.norm(dr, axis=1).min()
    node_sensor_dists.append(_mindist)
node_sensor_dists = np.array(node_sensor_dists)
# index of src that is closest to the sensor array
closest_inds = np.argsort(node_sensor_dists)[:n_closest]
print(node_sensor_dists[closest_inds])
plt.hist(node_sensor_dists, bins=100)
print(f'smallest source-sensor distance is {1e3 * min(node_sensor_dists):.1f} mm')
fig = mlab.figure()
pts = _mlab_points3d(node_coords_all_dev, figure=fig, scale_factor=pt_scale)
pts_ips = _mlab_points3d(ips, figure=fig, scale_factor=pt_scale / 2)

# highlight closest sources
colors = np.zeros(node_coords_all_dev.shape[0])
colors[closest_inds] = 1
pts.glyph.scale_mode = 'scale_by_vector'
pts.mlab_source.dataset.point_data.scalars = colors

# highlight some other sources
# colors = np.zeros(node_coords_all_dev.shape[0])
# colors[sd_improves_inds] = 1
# pts.glyph.scale_mode = 'scale_by_vector'
# pts.mlab_source.dataset.point_data.scalars = colors

# color some ips
# colors = np.zeros(ips.shape[0])
# inds = np.arange(850, 1000)
# colors[inds] = 1
# pts_ips.glyph.scale_mode = 'scale_by_vector'
# pts_ips.mlab_source.dataset.point_data.scalars = colors


# %% FIND superficial nodes by using the cortical mesh
# this takes a while

INITIAL_VERTEX_SPACING = 1e-3
ZLIM = (
    0.05  # pick sources higher than this (m) to exclude sources at bottom of the brain
)
DIST_LIMIT = 0.02  # (m) pick sources that are closer than this to cortical mesh

# create a cortical surface mesh
bs = mne.read_bem_surfaces(bem_file)
brainsurf = [s for s in bs if s['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN][0]
assert brainsurf['coord_frame'] == FIFF.FIFFV_COORD_MRI
brain_pts, brain_tris = brainsurf['rr'], brainsurf['tris']
tm_brain = trimesh.Trimesh(vertices=brain_pts, tris=brain_tris)
pts_, tris_ = trimesh.remesh.subdivide_to_size(
    brain_pts, brain_tris, INITIAL_VERTEX_SPACING
)
# filter & clean up mesh
tm = trimesh.Trimesh(vertices=pts_, faces=tris_)
trimesh.smoothing.filter_humphrey(tm)
# not sure if necessary
trimesh.repair.fix_winding(tm)
trimesh.repair.fix_inversion(tm)
trimesh.repair.fill_holes(tm)
pts_, tris_ = tm.vertices, tm.faces
print('after smoothing/cleanup: %d vertices %d faces' % (pts_.shape[0], tris_.shape[0]))
# MRI -> head coords
mri_head_t = invert_transform(head_mri_trans)
pts_ = apply_trans(mri_head_t, pts_)

# plot surface trimesh and all source nodes
# fig = mlab.figure()
# _mlab_trimesh(1.005 * pts_, tris_, figure=fig, transparent=True)
# _mlab_points3d(node_coords_thishemi, figure=fig, scale_factor=2e-3)

# find source distances to cortical mesh
node_surf_dists = list()
for node_ind in range(node_coords_thishemi.shape[0]):
    node_loc = node_coords_thishemi[node_ind, :]
    dr = pts_ - node_loc
    _mindist = np.linalg.norm(dr, axis=1).min()
    node_surf_dists.append(_mindist)
node_surf_dists = np.array(node_surf_dists)

# find superficial sources
superf_node_inds = np.where(
    np.logical_and(node_surf_dists < DIST_LIMIT, node_coords_thishemi[:, 2] > ZLIM)
)[0]

deeper_node_inds = np.where(
    np.logical_and(node_surf_dists > DIST_LIMIT, node_coords_thishemi[:, 2] > ZLIM)
)[0]

print(
    f'dist limit of {DIST_LIMIT:.2f} m: found {len(superf_node_inds)} superficial sources'
)
# plot
fig = mlab.figure()
_mlab_trimesh(pts_, tris_, figure=fig, transparent=True)
_mlab_points3d(node_coords_thishemi[superf_node_inds], figure=fig, scale_factor=2e-3)
fig = mlab.figure()
_mlab_trimesh(pts_, tris_, figure=fig, transparent=True)
_mlab_points3d(node_coords_thishemi[deeper_node_inds], figure=fig, scale_factor=2e-3)


# %% SSS basis
# NB: magnetometers are scaled by factor of MAG_SCALING in the basis
# some heuristics for the basis dimension and origin
_array_name = str(array_name).lower()
head_dev_trans = invert_transform(info['dev_head_t'])
print(f'array: {array_name}')
if array_name == 'VV-306':
    LIN, LOUT = 8, 3
    # use head model origin
    sss_origin = apply_trans(head_dev_trans, HEAD_ORIGIN)
    # OBS: experiment
    sss_origin = np.array([0.0, 0.025, 0.0])  # origin of device coords
elif 'compumedics' in _array_name:
    LIN, LOUT = 8, 3
    # use head model origin
    sss_origin = apply_trans(head_dev_trans, HEAD_ORIGIN)
elif 'cortex' in _array_name:  # the 'cortical' OPM array
    LIN, LOUT = 20, 3
    sss_origin = np.array([0.0, 0.0, 0.015])  # origin of device coords
elif 'opm' in _array_name:  # some other OPM array
    LIN, LOUT = 20, 3
    # for helmetlike OPM array, ensure that origin is above the helmet rim;
    # moving origin below z = 0 causes basis condition number to blow up
    # this is (apparently) due to the reduced solid angle coverage
    sss_origin = np.array([0.005, 0.02, 0.0])  # origin of device coords
elif 'radial' in _array_name:  # radial-spherical
    LIN, LOUT = 13, 3
    sss_origin = np.array([0.0, -0.0, 0.0])  # origin of device coords
elif 'barbute' in _array_name:  # barbute helmet
    LIN, LOUT = 20, 3
    sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
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

print('basis dim: Lin=%d Lout=%d' % (LIN, LOUT))
with np.printoptions(precision=3):
    print('subspace angles Sin vs Sout: %s' % subspace_angles_deg(Sin, Sout))
print('condition of S: %g' % np.linalg.cond(S))
print('condition of Sin: %g' % np.linalg.cond(Sin))

# compute L for each matrix element
basisvec_L = np.array([_idx_deg_ord(k)[0] for k in range(nin)])

# corresponding multipole-based leadfield
xin_leads_thishemi = np.squeeze(np.linalg.pinv(Sin) @ leads_thishemi_sc)
print('condition of leadfield: %g' % np.linalg.cond(leads_thishemi_sc))
print('condition of multipole-based leadfield: %g' % np.linalg.cond(xin_leads_thishemi))

# multipole-based resolution kernel
xin_res_kernel_thishemi = np.linalg.pinv(xin_leads_thishemi) @ xin_leads_thishemi

# source node distances to SSS origin
head_dev_trans = invert_transform(info['dev_head_t'])
node_coords_thishemi_dev = apply_trans(head_dev_trans, node_coords_thishemi)
node_origin_dists = np.linalg.norm(node_coords_thishemi_dev - sss_origin, axis=1)


# %% CHECK SSS geometry
pt_scale = 2e-3
# the integration points in device coords
ips = _prep_mf_coils(info)[0]
ip_dists = np.linalg.norm(ips - sss_origin, axis=1)
rmax_sss = ip_dists.min()
# highlight sources outside SSS sphere
inds_bad = np.where(node_origin_dists > rmax_sss)[0]
n_bad = len(inds_bad)
print('%d bad sources' % n_bad)
fig = mlab.figure()
pts = _mlab_points3d(node_coords_thishemi_dev, figure=fig, scale_factor=pt_scale)
_mlab_points3d(ips, figure=fig, scale_factor=pt_scale / 2)

# _mlab_quiver3d(
#    src_coords_thishemi_dev[inds_bad, :], src_normals_thishemi[inds_bad, :], figure=fig, scale_factor=5e-3, color=(1.0, 1.0, 1.0)
# )

colors = np.zeros(node_coords_thishemi_dev.shape[0])
colors[inds_bad] = 1
pts.glyph.scale_mode = 'scale_by_vector'
pts.mlab_source.dataset.point_data.scalars = colors
# plot the SSS 'inner sphere'
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0 : 2 * pi : 101j]
x = rmax_sss * sin(phi) * cos(theta) + sss_origin[0]
y = rmax_sss * sin(phi) * sin(theta) + sss_origin[1]
z = rmax_sss * cos(phi) + sss_origin[2]
mlab.mesh(x, y, z, opacity=0.35, figure=fig)
# ip-source distances (indices to this hemi)
node_surf_dists = list()
for node_ind in range(node_coords_thishemi.shape[0]):
    node_loc = node_coords_thishemi_dev[node_ind, :]
    dr = ips - node_loc
    _mindist = np.linalg.norm(dr, axis=1).min()
    node_surf_dists.append(_mindist)
node_surf_dists = np.array(node_surf_dists)


# %% check leadfield-basis match
ANG_QUANTILE = 0.995
STEP = 5  # step for looping over source indices (undersample for speedup)
basis_angles = np.zeros(nsrc_valid_thishemi) * np.nan
for k, ind in enumerate(np.arange(0, nsrc_valid_thishemi, STEP)):
    lead_unshifted = leads_thishemi[:, ind].copy()
    lead_unshifted_sc = _scale_magmeters(lead_unshifted, info, MAG_SCALING)
    basis_angles[ind] = subspace_angles_deg(Sin, lead_unshifted_sc)
print('Mean leadfield-basis angle: %.2f deg' % np.nanmean(basis_angles))
print('Worst leadfield-basis angle: %.2f deg' % np.nanmax(basis_angles))
print('%g quantile: %.2f' % (ANG_QUANTILE, np.nanquantile(basis_angles, ANG_QUANTILE)))


# %% plot some leads
fig, axs = plt.subplots(4, 2)
for k, ax in enumerate(axs.flat):
    ax.plot(leads_thishemi[:, k])
fig.suptitle(f'some forwards for {array_name}')


# %% study representation of signal energy (single source)
node_ind = np.where(test_sources['superficial_z'])[0]  # pick a source
node_ind = [1014]  # something else
leads_thishemi_sc = _scale_magmeters(leads_thishemi, info, MAG_SCALING)
sigvec = leads_thishemi_sc[:, node_ind]
sigvec_L = _limit_L(sigvec, Sin, basisvec_L)
cum_energy = np.linalg.norm(sigvec_L, axis=0) / np.linalg.norm(sigvec) * 100
plt.plot(np.arange(1, LIN + 1), cum_energy)
plt.xticks(np.arange(1, LIN + 1))
# plt.ylim([0, 100])
plt.grid()


# %% study representation of signal energy (multipole sources) vs Lin
# this calculates the ratio of 2-norms of the L-limited and full signal
# space vecs
leads_thishemi_sc = _scale_magmeters(leads_thishemi, info, MAG_SCALING)
node_inds = np.where(node_dists_thishemi > 0.06)[0]  # pick superficial sources
cergs = list()
nsrcs = len(node_inds)
for k, node_ind in enumerate(node_inds):
    sigvec = leads_thishemi_sc[:, node_ind]
    sigvec_L = _limit_L(sigvec, Sin, basisvec_L)
    cum_energy = np.linalg.norm(sigvec_L, axis=0) / np.linalg.norm(sigvec) * 100
    cergs.append(cum_energy)
    if not k % 10:
        print(f'computing source representation: {100 * k / nsrcs:.1f}%')
cergs = np.array(cergs)


# %% plot the above
# plt.plot(np.arange(1, LIN + 1), cergs.mean(axis=0), label='mean')
# plt.plot(np.arange(1, LIN + 1), cergs.min(axis=0), label='worst')
# plt.plot(np.arange(1, LIN + 1), cergs.max(axis=0), label='best')
plt.plot(np.arange(1, LIN + 1), cergs.T)
# plt.plot(np.arange(1, LIN + 1), np.mean(cergs, axis=0), 'k--')
plt.xticks(np.arange(1, LIN + 1))
# plt.ylim([0, 100])
plt.grid()
plt.ylabel('Relative signal vector norm (%)')
plt.xlabel('Lin')
plt.title(f'Relative norm vs. maximum Lin, {array_name}')


# ***
# %% compute resolution kernel, spatial dispersion and focality for multipole leadfields
sds = dict()
xin_res_kernels = dict()
focs = dict()
#
# Method for getting the multipole-based leadfield for each L.
#
# OLD METHOD: decompose the sensor-based leadfield into the full range of
# available L components first, then pick the components for each L in turn.
# In principle, this has the advantage of avoiding aliasing, since the initial
# fit includes all spatial freqs.
# If L is very high, the initial decomposition may require regularization.
#
# ALT METHOD (FIT_REDUCED_BASES=True:: fit leadfield onto the reduced-order
# basis at each L. Since this always 
#  
# 
FIT_REDUCED_BASES = True
RCOND = 1e-14  # for pinv; 1e-7 gives reasonable regularization
xin_lead_conds = list()
for L in range(1, LIN + 1):
    theLs = list(range(1, L + 1))
    # the component indices to include
    include = np.where(np.isin(basisvec_L, theLs))[0]
    if FIT_REDUCED_BASES:
        Sin_red = Sin[:, include]
        xin_leads_filt = np.squeeze(np.linalg.pinv(Sin_red) @ leads_thishemi_sc)
    else:
        xin_leads_filt = xin_leads_thishemi[include]
    xin_lead_conds.append(np.linalg.cond(xin_leads_filt))
    print(f'{L=}, leadfield condition: {np.linalg.cond(xin_leads_filt):g}')
    res_kernel = _resolution_kernel(xin_leads_filt, rcond=RCOND)
    xin_res_kernels[L] = res_kernel
    sds[L] = _spatial_dispersion(res_kernel, src_dij_thishemi)
    focs[L] = _focality(res_kernel)


# ***
# %% plot leadfield conditioning
LIN_list = list(range(1, LIN + 1))
plt.axhline(1e9, color='r', ls='--', label='field-based')
plt.semilogy(LIN_list, xin_lead_conds, label='multipole-based')
# plt.title('Multipole vs sensor-based leadfield conditioning')
plt.xticks(LIN_list[::1])
plt.xlabel('L')
plt.ylabel('Condition number')
plt.legend()

plt.savefig(
    f'multipole_leadfield_cond_{array_name}.png',
    facecolor='w',
    dpi=200,
)


# %% SVD of leadfield


def tsvd(A, b, n_svd_comps):
    """Solves Ax=b by TSVD"""
    U, D, Vh = np.linalg.svd(A, full_matrices=False)
    # solution based on truncated SVD and pseudoinverse formula
    # see numpy.linalg.pinv docs
    return Vh[:n_svd_comps, :].T @ np.diag(1/D[:n_svd_comps]) @ U[:, :n_svd_comps].T @ b


def tsvd2(A, b, n_svd_comps):
    """Solves Ax=b by TSVD"""
    U, D, Vh = np.linalg.svd(A, full_matrices=False)
    # it can be shown that the reduced rank pseudoinverse solution equals this sum
    # (see Jussi's M.Sc.)
    # interestingly, solution looks "better" than above
    return sum(U[:, k] @ b * Vh[k, :] / D[k] for k in range(n_svd_comps))

# "smart" least squares
x, _, rank, _ = scipy.linalg.lstsq(A, b)




# %% SD from sensor-based leadfield
RCOND_SENSOR = 1e-7  # for pinv; 1e-7 gives reasonable regularization
res_kernel_thishemi = _resolution_kernel(leads_thishemi_sc, rcond=RCOND_SENSOR)
sds_sensor = _spatial_dispersion(res_kernel_thishemi, src_dij_thishemi)
sds_sensor[1480]

# %% mpl plot dispersion, color coded curves

from matplotlib import cm
from matplotlib.colors import Normalize

cmap = cm.plasma
cdatas = node_origin_dists  # use dist from SSS origin
norm = Normalize(vmin=cdatas.min(), vmax=cdatas.max())

LIN_limit = 18  # highest LIN to plot
sds_matrix = np.array(list(v for k, v in sds.items() if k <= LIN_limit))
LIN_list = list(range(1, LIN_limit + 1))

for k, sd in enumerate(sds_matrix.T):
    val = cdatas[k]
    # if val < .03:  # set a condition for curves to plot
    plt.plot(LIN_list, sd, c=cmap(norm(val)))

plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='dist from SSS origin (m)')

plt.xlabel('Lin')
plt.ylabel('Dispersion (m)')
# plt.legend()
plt.title(f'Dispersion vs. LIN (R>7cm)\n{array_name}', fontdict={'size': 10})
plt.ylim([0, 0.1])

plt.xticks(list(range(1, LIN_limit + 1))[::2])
plt.savefig(
    f'dispersion_curveplot_{FIX_ORI_DESCRIPTION}_{array_name}.png',
    facecolor='w',
    dpi=200,
)


# %% mpl plot dispersion as function of source and Lin, median curves

LIN_limit = 8  # highest LIN to plot
# convert dispersion data to mm
sds_matrix = 1e3 * np.array(list(v for k, v in sds.items() if k <= LIN_limit))
LIN_list = list(range(1, LIN_limit + 1))

DEEP_LIMIT = 0.06
SUPERF_LIMIT = 0.06
deep_node_inds = np.where(node_origin_dists < DEEP_LIMIT)[0]
shallow_node_inds = np.where(node_origin_dists >= SUPERF_LIMIT)[0]

plt.plot(
    LIN_list,
    np.median(sds_matrix[:, deep_node_inds], axis=1),
    'k',
    label=f'deep (d < {DEEP_LIMIT} m)',
)

plt.plot(
    LIN_list,
    np.median(sds_matrix[:, shallow_node_inds], axis=1),
    'k--',
    label=f'superficial (d > {SUPERF_LIMIT} m)',
)

plt.xlabel('Lin')
plt.ylabel('Dispersion (mm)')
plt.legend()
# plt.title(f'Dispersion vs. LIN\n{array_name}')
# plt.ylim([0, .1])
plt.xticks(list(range(1, LIN_limit + 1))[::1])
plt.savefig(
    f'dispersion_median_curveplot_{FIX_ORI_DESCRIPTION}_{array_name}.png',
    facecolor='w',
    dpi=200,
)


# %% plot focality as function of source and Lin
focs_matrix = np.array(list(focs.values()))
plt.plot(list(range(1, LIN + 1)), focs_matrix[:, superf_node_inds])
plt.plot(
    list(range(1, LIN + 1)),
    np.median(focs_matrix[:, superf_node_inds], axis=1),
    'k--',
)
plt.xlabel('Lin')
plt.ylabel('Focality')
plt.title(
    f'Focality vs. LIN for superficial sources (N={len(superf_node_inds)})\n{array_name}'
)
plt.xticks(list(range(1, LIN + 1)))
plt.ylim([0, 120])
plt.savefig(f'focality_curveplot_{FIX_ORI_DESCRIPTION}_{array_name}.png', facecolor='w')


# %% PySurfer plot of spatial dispersion as function of Lin
START_IND = 0
N_SKIP = 2  # reduce n of plots by stepping the index
# scalarize the dispersion and convert to mm
src_datas = [
    1e3 * _scalarize_src_data(val, nverts_thishemi, reducer_fun=np.mean)
    for val in sds.values()
]
titles = [f'L=1..{k}' for k in range(1, LIN + 1)]
outfn = projectpath / f'dispersion_cortexplot_{FIX_ORI_DESCRIPTION}_{array_name}.png'
fmin, fmax = 0, 60
_montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    src_datas[START_IND::N_SKIP],
    titles[START_IND::N_SKIP],
    src_node_inds[HEMI_IND],
    HEMI,
    outfn,
    frange=(fmin, fmax),
    ncols_max=3,
    colormap='plasma_r',
    colorbar_nlabels=4,
    title_width=0.3,
)


# %% PySurfer plot of focality as function of Lin
titles = [f'L=1..{k}' for k in range(1, LIN + 1)]
outfn = projectpath / f'focality_cortexplot_{FIX_ORI_DESCRIPTION}_{array_name}.png'
fmin, fmax = 0, 30
_montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    focs.values(),
    titles,
    src_node_inds[HEMI_IND],
    HEMI,
    outfn,
    frange=(fmin, fmax),
    ncols_max=4,
    colormap='plasma_r',
    colorbar_nlabels=5,
    title_width=0.3,
)


# ***
# %% PySurfer plot of a single source PSF as function of Lin
N_SKIP = 1  # reduce n of plots by stepping the index
MAX_LIN = 16  # max LIN value to use
ind = 931
ind = superf_node_inds[ind]  # pick one of superficial nodes
ind = _node_to_source_index(ind, FIX_ORI)
#ind = 1480
# frange = 0, .05  # global fixed
# frange = None  # global auto
frange = 'separate'  # individual auto
if not FIX_ORI:
    ind = ind[1]  # pick a single orientation
src_datas = [np.abs(res_kernel[ind, :]) for res_kernel in xin_res_kernels.values()][
    :MAX_LIN:N_SKIP
]
src_datas_scalar = [_scalarize_src_data(d, nverts_thishemi) for d in src_datas]
titles = [
    f'L=1..{k}, SD={1e3*disp[ind]:.2f} mm'
    for k, disp in zip(range(1, MAX_LIN + 1), sds.values())
][:MAX_LIN:N_SKIP]
outfn = projectpath / f'psf_cortexplot_{FIX_ORI_DESCRIPTION}_{array_name}_LIN{MAX_LIN}.png'
_montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    src_datas_scalar,
    titles,
    src_node_inds[HEMI_IND],
    HEMI,
    outfn,
    frange=frange,
    ncols_max=4,
    colormap='plasma',
    colorbar_nlabels=4,
    title_width=0.7,
    do_colorbar=False,
)


# %% WIP: mlab based plot
outfn = projectpath / f'dispersion_mlab.png'
titles = [f'L=1..{k}' for k in range(1, LIN + 1)]
foo = _montage_mlab_brain_plots(
    sds.values(),
    titles,
    node_coords_thishemi,
    outfn,
)


# %% single PySurfer plot

src_ind = 1480
src_data = res_kernel[src_ind, :]
# src_data = xin_res_kernels[18][src_ind, :]
src_data = np.abs(src_data)

fig = mlab.figure()
brain = Brain(
    subject,
    HEMI,
    surf,
    subjects_dir=subjects_dir,
    background='white',
    figure=fig,
)

brain.add_data(
    src_data,
    vertices=src_node_inds[HEMI_IND],
    colormap='plasma',
    hemi=HEMI,
    colorbar=True,
    smoothing_steps=None,
)

# cb = brain._data_dicts[HEMI][0]['colorbars'][0]
# cb.label_text_property.bold = 0
# cb.scalar_bar.unconstrained_font_size = True
# cb.scalar_bar.number_of_labels = 6
# cb.label_text_property.font_size = 16

frange = (0, 6e-3)
frange = None
if frange is not None:
    fmin, fmax = frange
    fmid = (fmin + fmax) / 2
    brain.scale_data_colormap(
        fmin=fmin, fmid=fmid, fmax=fmax, transparent=False, verbose=False
    )
title = 'test' * 10
title_size = 1
title_height = 0.9
# mlab.title(title, size=title_size, height=title_height)
sd = 1000 * sds_sensor[src_ind]
mlab.title(f'Sensor-based, SD={sd:.2f} mm')


# %% compare multipole and sensor-based inverse
plt.figure()
# plt.plot(res_kernel_thishemi[src_ind, :], label='sensor')
plt.plot(sds_sensor)
plt.ylim([0, 0.05])
plt.figure()
# plt.plot(xin_res_kernels[18][src_ind, :], label='multipole')
plt.plot(sds[26])
plt.ylim([0, 0.05])


# %% sensor-space leadfield inversion, noisy version
# NB: conditioning of sensor-based leadfield is very bad, so regularization is needed
# (even without explicit additive noise)

SNR = 100
REGU_METHOD = 'tikhonov_svd'
# in literature, values of ~1e-5 to ~1e-11 have been considered
LAMBDA = 1e-8
PINV_RCOND = 1e-2
FIG_BG_COLOR = (0.3, 0.3, 0.3)
FIGSIZE = (400, 300)
NCOLS_MAX = 5
src_ind = test_nodes['lateral']  # pick a source
src_ind = 1480

# pick source forward from leadfield
src_lead_sensors = leads_thishemi_sc[:, src_ind]
# NB: this may disproportionely affect the other type of sensor (mags/grads)

noisevec = np.random.randn(*src_lead_sensors.shape)
snr_scaler = np.linalg.norm(noisevec) / np.linalg.norm(src_lead_sensors) * SNR

noisevec /= snr_scaler

snr_true = np.linalg.norm(src_lead_sensors) / np.linalg.norm(noisevec)
print(f'{snr_true=}')

src_lead_sensors_noisy = src_lead_sensors + noisevec
inv_sol = _min_norm_pinv(
    leads_thishemi_sc,
    src_lead_sensors_noisy,
    method=REGU_METHOD,
    tikhonov_lambda=LAMBDA,
    rcond=PINV_RCOND,
)
#inv_sol, _, irank, _ = np.linalg.lstsq(leads_thishemi_sc, src_lead_sensors_noisy)
#inv_sol = tsvd(leads_thishemi_sc, src_lead_sensors_noisy, 200)

plt.plot(inv_sol)

fig = mlab.figure()
brain = Brain(
    subject,
    HEMI,
    surf,
    subjects_dir=subjects_dir,
    background='white',
    figure=fig,
)

brain.add_data(
    inv_sol,
    vertices=src_node_inds[HEMI_IND],
    colormap='plasma',
    hemi=HEMI,
    colorbar=True,
    smoothing_steps=None,
)

frange = (0, max(inv_sol))
if frange is not None:
    fmin, fmax = frange
    fmid = (fmin + fmax) / 2
    brain.scale_data_colormap(
        fmin=fmin, fmid=fmid, fmax=fmax, transparent=False, verbose=False
    )

# plot the source as arrow
# XXX: arrow goes into wrong place - coord transform issue?

src_loc = node_coords_thishemi[src_ind, :]
#src_loc_mri = apply_trans(mri_head_t, src_loc)
src_normal = src_normals[HEMI_IND][src_ind, :]
_mlab_quiver3d(
    src_loc,
    src_normal,
    figure=fig,
    scale_factor=1,
    color=(1.0, 1.0, 1.0),
)

mlab.title(f'{array_name} ({SNR=}, {LAMBDA=})')
mlab.savefig(f'tikhonov_inversion_{array_name}_SNR{SNR}_LAMBDA{LAMBDA}.png')


# %%
# visualize the inverse
fig = mlab.figure(bgcolor=FIG_BG_COLOR)
pt_scale = 0.003
src_coords_all = np.concatenate((src_coords[0], src_coords[1]))
nodes = _mlab_points3d(src_coords_thishemi, figure=fig, scale_factor=pt_scale)
nodes.glyph.scale_mode = 'scale_by_vector'
# MNE solution is scalar (fixed-orientation leadfield); plot the absolute value
current_abs = np.abs(inv_sol)
focality = current_abs.max() / current_abs.sum()
focality = len(np.where(current_abs > current_abs.max() / 3)[0])
# L-specific colormap scaling (each L scaled by its maximum)
colors = current_abs / current_abs.max()
# uniform scaling across L values
# colors = current_abs / .05
nodes.mlab_source.dataset.point_data.scalars = colors
# plot the source as arrow
src_loc = src_coords_thishemi[None, src_ind, :]
src_normal = src_normals[HEMI_IND][None, src_ind, :]
_mlab_quiver3d(
    src_loc, src_normal, figure=fig, scale_factor=0.05, color=(1.0, 1.0, 1.0)
)
mlab.view(170, 50, roll=60)  # lateral view
# mlab.title(str(TIKH_L))



# %% L-dependent MNP solution with noise
# added 2.3.2022 PSF

# XXX: invert the full leadfield instead of 1 hemi; this should also be fixed
# elsewhere
leads_sc_0 = _scale_magmeters(leads[0], info, MAG_SCALING)
leads_sc_1 = _scale_magmeters(leads[1], info, MAG_SCALING)
leads_sc_all = np.hstack((leads_sc_0, leads_sc_1))

REGU_METHOD = 'tikhonov_naive'
# in literature, values of ~1e-5 to ~1e-11 have been considered; 1e-8 is reasonable
LAMBDA = 0
PINV_RCOND = 1e-20
FIG_BG_COLOR = (0.3, 0.3, 0.3)
FIGSIZE = (400, 300)
NCOLS_MAX = 5
SNR = 1e10  # desired SNR (defined as ratio of signal vector norms)
LIN_MAX = 14
frange = 'separate'  # individual auto

ind = superf_node_inds[931]  # pick one of superficial nodes
src_ind = _node_to_source_index(ind, FIX_ORI)
src_lead_sensors = leads_thishemi_sc[:, src_ind]
# XXX: noisevec mag scaling?
noisevec = np.random.randn(*src_lead_sensors.shape)
noisevec /= (np.linalg.norm(noisevec) * SNR)
noisevec *= np.linalg.norm(src_lead_sensors)
src_lead_sensors_noisy = src_lead_sensors + noisevec

inverses = list()
titles = list()
for L in range(1, LIN_MAX + 1):
    # select multipole components to include according to L
    theLs = list(range(1, L + 1))
    include = np.where(np.isin(basisvec_L, theLs))[0]
    Sin_red = Sin[:, include]
    # L-limited multipole transformation of leadfield
    xin_leads_red = np.squeeze(np.linalg.pinv(Sin_red) @ leads_sc_all)
    lead_cond = np.linalg.cond(xin_leads_red)
    # multipole components for noisy forward field
    lead_cond = np.linalg.cond(xin_leads_red) / np.linalg.cond(xin_leads_red, -2)
    scond = np.linalg.cond(Sin_red) / np.linalg.cond(Sin_red, -2)
    print(f'{L=} condition leadfield: {lead_cond:.2e} Sin: {scond}')
    xin_source_red = np.squeeze(np.linalg.pinv(Sin_red) @ src_lead_sensors_noisy)
    # do the minimum norm inverse
    # NOTE: need for regularization depends on the xin leadfield matrix
    # regularization of well-conditioned matrices may screw things up
    inv_sol = _min_norm_pinv(
        xin_leads_red,
        xin_source_red,
        method=REGU_METHOD,
        tikhonov_lambda=LAMBDA,
        rcond=PINV_RCOND,
    )[:3732]                        # XXX: hack!!
    inverses.append(inv_sol)
    Lstr = str(L).ljust(2)
    titles.append(f'L={Lstr}')

outfn = projectpath / 'foo.png'

_montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    inverses,
    titles,
    src_node_inds[HEMI_IND],
    HEMI,
    outfn,
    frange=frange,
    ncols_max=5,
    colormap='plasma',
    colorbar_nlabels=4,
    title_width=0.07,
    do_colorbar=False,
)


# %%
    # visualize the inverse
    fig = mlab.figure(bgcolor=FIG_BG_COLOR)
    pt_scale = 0.003
    src_coords_all = np.concatenate((node_coords[0], node_coords[1]))
    nodes = _mlab_points3d(node_coords_thishemi, figure=fig, scale_factor=pt_scale)
    nodes.glyph.scale_mode = 'scale_by_vector'
    # MNE solution is scalar (fixed-orientation leadfield); plot the absolute value
    node_strength = np.abs(inv_sol)
    focality = node_strength.max() / node_strength.sum()
    focality = len(np.where(node_strength > node_strength.max() / 3)[0])
    # L-specific colormap scaling (each L scaled by its maximum)
    colors = node_strength / node_strength.max()
    # uniform scaling across L values
    # colors = current_abs / .05
    nodes.mlab_source.dataset.point_data.scalars = colors
    # plot the source as arrow
    src_loc = node_coords_thishemi[None, src_ind, :]
    src_normal = src_normals[HEMI_IND][None, src_ind, :]
    _mlab_quiver3d(
        src_loc, src_normal, figure=fig, scale_factor=0.05, color=(1.0, 1.0, 1.0)
    )
    mlab.view(170, 50, roll=60)  # lateral view
    mlab.title('L=%d, f=%d' % (L, focality))











# %% L-filtered min. norm solution, noisy version
# this first computes multipole coefficients for the forward and leadfield,
# and then filters according to L
# +save figs into png files
CUMUL = True  # include single L at a time, or all components upto L
SAVE_FIGS = True
FIG_BG_COLOR = (0.3, 0.3, 0.3)
FIG_BG_COLOR = (1., 1., 1.)
FIGSIZE = (400, 300)
NCOLS_MAX = 5
REL_NOISE = 0.1  # noise relative to std. dev of signal
REGU_METHOD = 'pinv'
TIKH_L = 1e-27
PINV_RCOND = 1e-10
src_ind = test_sources['superficial_z']  # pick a source
# src_ind = 1330  # alternative source
mlab.options.offscreen = SAVE_FIGS
fignames = list()
inverses = list()
# get multipole coefficients for all leadfields
leads_thishemi_sc = _scale_magmeters(leads_thishemi, info, MAG_SCALING)
src_lead_sensors = leads_thishemi_sc[:, src_ind]
noisevec = (
    REL_NOISE * np.std(src_lead_sensors) * np.random.randn(*src_lead_sensors.shape)
)
src_lead_sensors_noisy = src_lead_sensors + noisevec
src_lead_multipole = np.linalg.pinv(Sin) @ src_lead_sensors_noisy
xin_leads = np.squeeze(np.linalg.pinv(Sin) @ leads_thishemi_sc)

for L in range(1, LIN + 1):
    theLs = list(range(1, L + 1)) if CUMUL else [L]
    print(theLs)
    # select multipole components to include according to given L values
    include = np.where(np.isin(basisvec_L, theLs))[0]
    xin_leads_filt = xin_leads[include]
    src_lead_red = src_lead_multipole[include]
    # NOTE: need for regularization depends on the xin leadfield matrix
    # regularization of well-conditioned matrices may screw up things
    # Tikhonov
    inv_sol = _min_norm_pinv(
        xin_leads_filt,
        src_lead_red,
        method=REGU_METHOD,
        tikhonov_lambda=TIKH_L,
        rcond=PINV_RCOND,
    )
    inverses.append(inv_sol)
    # visualize the inverse
    fig = mlab.figure(bgcolor=FIG_BG_COLOR)
    pt_scale = 0.003
    src_coords_all = np.concatenate((src_coords[0], src_coords[1]))
    nodes = _mlab_points3d(src_coords_thishemi, figure=fig, scale_factor=pt_scale)
    nodes.glyph.scale_mode = 'scale_by_vector'
    # MNE solution is scalar (fixed-orientation leadfield); plot the absolute value
    node_strength = np.abs(inv_sol)
    focality = node_strength.max() / node_strength.sum()
    focality = len(np.where(node_strength > node_strength.max() / 3)[0])
    # L-specific colormap scaling (each L scaled by its maximum)
    colors = node_strength / node_strength.max()
    # uniform scaling across L values
    # colors = current_abs / .05
    nodes.mlab_source.dataset.point_data.scalars = colors
    # plot the source as arrow
    src_loc = src_coords_thishemi[None, src_ind, :]
    src_normal = src_normals[HEMI_IND][None, src_ind, :]
    _mlab_quiver3d(
        src_loc, src_normal, figure=fig, scale_factor=0.05, color=(1.0, 1.0, 1.0)
    )
    mlab.view(170, 50, roll=60)  # lateral view
    mlab.title('L=%d, f=%d' % (L, focality))
    cumul_desc = 'cumul' if CUMUL else 'single'
    fname = 'inverse_%s_%d.png' % (cumul_desc, L)
    if SAVE_FIGS:
        # save fig for the montage
        fname = _named_tempfile(suffix='.png')
        print('saving %s' % fname)
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)
        mlab.close(fig)
if SAVE_FIGS:
    # complete the montage using empty figures, so that the background is
    # consistent
    n_last = NCOLS_MAX - len(fignames) % NCOLS_MAX
    if n_last == NCOLS_MAX:
        n_last = 0
    for k in range(n_last):
        fig = mlab.figure(bgcolor=FIG_BG_COLOR)
        fname = _named_tempfile(suffix='.png')
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)
    mlab.options.offscreen = False  # restore
    # set filename for montage image
    montage_fn = 'inverse_%s.png' % cumul_desc
    montage_fn = f'mnp_{cumul_desc}_{array_name}.png'
    _montage_figs(fignames, montage_fn, ncols_max=NCOLS_MAX)
del fignames
plt.figure()
plt.plot(src_lead_sensors)
plt.plot(noisevec)
plt.legend(('source field', 'noise'))




# %% regu exp



def tikhonov_svd(A, b, lbda):
    """Solve Ax=b for x using Tikhonov regularization. Implementation via SVD"""
    U, D, Vh = np.linalg.svd(A, full_matrices=False)
    # compute SVD filter factors for Tikhonov
    # see https://en.wikipedia.org/wiki/Tikhonov_regularization
    f_tikh = D**2 / (lbda + D**2)
    return sum(f_tikh[k] * U[:, k] @ b * Vh[k, :] / D[k] for k in range(len(D)))


def pinv_svd(A, b):
    U, D, Vh = np.linalg.svd(A, full_matrices=False)
    return Vh @ (np.diag(D) @ U.T) @ b


fld = leads_thishemi_sc[:, 2000]
x1 = np.linalg.lstsq(leads_thishemi_sc, fld)[0]
x2 = np.linalg.pinv(leads_thishemi_sc) @ fld
x3 = tikhonov_svd(leads_thishemi_sc, fld, 1e-8)
x4 = pinv_svd(leads_thishemi_sc, fld)
minn = 1900
maxn = 2100
plt.figure()
plt.plot(x1[minn:maxn], label='least squares')
plt.plot(x2[minn:maxn], label='pinv')
plt.plot(x3[minn:maxn], label='tikh')
plt.plot(x4[minn:maxn], label='pinv_svd')

plt.legend()

