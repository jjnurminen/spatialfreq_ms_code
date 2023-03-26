#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots for manuscripts. Data is computed by mne_inverse.py

@author: jussi
"""

# %% INITIALIZE
import mne
import numpy as np
from mayavi import mlab
from surfer import Brain
import matplotlib.pyplot as plt

from forward_comp import (
    _scale_magmeters,
    _min_norm_pinv,
    _scalarize_src_data,
)
from viz import _rescale_brain_colormap
from megsimutils.viz import (
    _mlab_quiver3d,
    _mlab_points3d,
    _mlab_colorblobs,
)
from megsimutils.fileutils import _named_tempfile, _montage_figs
from megsimutils.envutils import _ipython_setup


_ipython_setup(enable_reload=True)


# %% single-source focality vs. Lin
# first computes multipole coefficients for the forward and leadfield,
# and then filters according to L
# 'dotlike' visualization % montage
if not FIX_ORI:
    raise RuntimeError('free ori not yet implemented')
CUMUL = 'cumul'  # include single L at a time, or all components upto L
SAVE_FIGS = True
FIG_BG_COLOR = (0.3, 0.3, 0.3)
# FIG_BG_COLOR = (0., 0., 0.)
FIGSIZE = (400, 300)
NCOLS_MAX = 5
src_ind = np.where(test_sources['lateral'])[0]  # pick a source
# src_ind = 1738  # good!
# src_ind = 1580  # good (f=5)
# src_ind = 1744  # good (f=5)
# src_ind = 2417  # good (f=4, bit lowish)
# src_ind = 2494
mlab.options.offscreen = SAVE_FIGS
fignames = list()
inverses = list()
conds = list()
# get multipole coefficients for all leadfields
leads_thishemi_sc = _scale_magmeters(leads_thishemi, info, MAG_SCALING)
xin_leads = np.squeeze(np.linalg.pinv(Sin) @ leads_thishemi_sc)
for L in range(1, LIN + 1):
    if CUMUL == 'single':
        theLs = [L]
        plot_title = str(L)
    elif CUMUL == 'cumul':
        theLs = list(range(1, L + 1))
        plot_title = f'L=1..{L}'
    elif CUMUL == 'reverse':
        theLs = list(range(L, LIN + 1))
        plot_title = f'L={L}..{LIN}'
    elif CUMUL[:5] == 'range':  # 'bandpass'
        bp_width = int(CUMUL[5:])
        theLs = list(range(L - bp_width, L + bp_width + 1))
        theLs = [l for l in theLs if l >= 1 and l <= LIN]
        plot_title = f'{min(theLs)}..{max(theLs)}'
    else:
        raise RuntimeError('invalid CUMUL parameter')
    print(f'{theLs=}')
    # select multipole components to include according to given L values
    include = np.where(np.backisin(basisvec_L, theLs))[0]
    xin_leads_filt = xin_leads[include]
    # do the inverse (source lead is picked directly out of leadfield matrix)
    src_lead_filt = xin_leads_filt[:, src_ind]
    # NOTE: need for regularization depends on the xin leadfield matrix
    # regularization of well-conditioned matrices may screw up things
    xin_leads_filt_cond = np.linalg.cond(xin_leads_filt)
    conds.append(xin_leads_filt_cond)
    print(f'{xin_leads_filt_cond=:.1e}')
    inv_sol = _min_norm_pinv(xin_leads_filt, src_lead_filt, method='pinv')
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
    # mlab.view(180, 0, roll=90)
    # mlab.title(f'{plot_title}, f={focality}, cond={xin_leads_filt_cond:.0e}', size=.6)
    mlab.title(f'{plot_title}, f={focality}')
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


# %% single-source focality vs. Lin
# first computes multipole coefficients for the forward and leadfield,
# and then filters according to L
# PySurfer visualization & montage
if not FIX_ORI:
    raise RuntimeError('free ori not yet implemented')
CUMUL = 'cumul'  # include single L at a time, or all components upto L
SAVE_FIGS = True
FIG_BG_COLOR = (0.3, 0.3, 0.3)
# FIG_BG_COLOR = (0., 0., 0.)
FIGSIZE = (400, 300)
NCOLS_MAX = 4
src_ind = np.where(test_sources['superficial_z'])[0]  # pick a source
# src_ind = 1738  # good!
# src_ind = 1580  # good (f=5)
# src_ind = 1744  # good (f=5)
# src_ind = 2417  # good (f=4, bit lowish)
# lateral cands: [1810, 1844, 1908, 1909, 1911, 1949, 1973, 2009, 2010, 2012, 2013,
# 2014, 2075, 2077, 2108, 2131, 2132, 2157, 2211]
src_ind = 1909  # 13 min focality
src_ind = 1911  # 12
src_ind = 1949
mlab.options.offscreen = SAVE_FIGS
fignames = list()
inverses = list()
conds = list()
# get multipole coefficients for all leadfields
leads_thishemi_sc = _scale_magmeters(leads_thishemi, info, MAG_SCALING)
xin_leads = np.squeeze(np.linalg.pinv(Sin) @ leads_thishemi_sc)
brains = list()
for L in range(1, LIN + 1):
    if CUMUL == 'single':
        theLs = [L]
        plot_title = str(L)
    elif CUMUL == 'cumul':
        theLs = list(range(1, L + 1))
        plot_title = f'L=1-{L}'
    elif CUMUL == 'reverse':
        theLs = list(range(L, LIN + 1))
        plot_title = f'L={L}-{LIN}'
    elif CUMUL[:5] == 'range':  # 'bandpass'
        bp_width = int(CUMUL[5:])
        theLs = list(range(L - bp_width, L + bp_width + 1))
        theLs = [l for l in theLs if l >= 1 and l <= LIN]
        plot_title = f'{min(theLs)}..{max(theLs)}'
    else:
        raise RuntimeError('invalid CUMUL parameter')
    print(f'{theLs=}')
    # select multipole components to include according to given L values
    include = np.where(np.isin(basisvec_L, theLs))[0]
    xin_leads_filt = xin_leads[include]
    # do the inverse (source lead is picked directly out of leadfield matrix)
    src_lead_filt = xin_leads_filt[:, src_ind]
    # NOTE: need for regularization depends on the xin leadfield matrix
    # regularization of well-conditioned matrices may screw up things
    xin_leads_filt_cond = np.linalg.cond(xin_leads_filt)
    conds.append(xin_leads_filt_cond)
    print(f'{xin_leads_filt_cond=:.1e}')
    inv_sol = _min_norm_pinv(xin_leads_filt, src_lead_filt, method='pinv')
    inverses.append(inv_sol)

    # MNE solution is scalar (fixed-orientation leadfield); plot the absolute value
    node_strength = np.abs(inv_sol)
    focality = node_strength.max() / node_strength.sum()
    focality = len(np.where(node_strength > node_strength.max() / 3)[0])
    # L-specific colormap scaling (each L scaled by its maximum)
    colors = node_strength / node_strength.max()

    # visualize the inverse
    fig = mlab.figure(bgcolor=FIG_BG_COLOR)
    brain = Brain(
        subject, HEMI, surf, subjects_dir=subjects_dir, background='white', figure=fig
    )
    brains.append(brain)
    colorbar = False
    brain.add_data(
        colors,
        vertices=src_vertices[HEMI_IND],
        colormap='plasma',
        hemi=HEMI,
        colorbar=colorbar,
    )

    # plot the source as arrow
    src_loc = src_coords_thishemi[None, src_ind, :]
    src_normal = src_normals[HEMI_IND][None, src_ind, :]
    _mlab_quiver3d(
        src_loc, src_normal, figure=fig, scale_factor=0.05, color=(1.0, 1.0, 1.0)
    )
    # mlab.view(170, 50, roll=60)  # lateral view

    # mlab.view(180, 0, roll=90)
    # mlab.title(f'{plot_title}, f={focality}, cond={xin_leads_filt_cond:.0e}', size=.6)
    mlab.title(f'{plot_title}, f={focality}')
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


# %% spatial distribution of the focality index:
# indicates how well we can pinpoint sources at different cortical
# locations using the MNP solution
# (inversely related to the "point spread width" at each cortical location)
# focality: N of sources exceeding 33% of the maximum source amplitude
LMAX = 20  # restrict source and leadfield to a spatial frequency maximum
DIST_THRE = 0.095
print(f'using {LMAX=}')
theLs = list(range(1, LMAX + 1))
include = np.where(np.isin(basisvec_L, theLs))[0]
xin_leads_filt = xin_leads[include]
focs = list()
inv_sols = list()
focs_vs_L = dict()

src_inds = np.where(src_dists_thishemi > DIST_THRE)[0]  # pick superficial sources
# src_inds = np.arange(nsrc_valid_thishemi)  # all sources for this hemi

nsrcs = len(src_inds)
print(f'{nsrcs=}')

# %% do the hard work
for k, src_ind in enumerate(src_inds):
    # get the multipole forward field for this source
    src_lead = xin_leads_filt[:, src_ind]
    # do the mne inverse in multipole space
    inv_sol = _min_norm_pinv(xin_leads_filt, src_lead, method='pinv')
    inv_sols.append(inv_sol)
    node_strength = _scalarize_src_data(inv_sol, nverts_thishemi)
    focality = len(np.where(node_strength > node_strength.max() / 3)[0])
    focs.append(focality)
    if k % 100 == 0:
        print(f'computing inverses: {100 * k / nsrcs:.1f}%')
focs = np.array(focs).astype(np.float)
focs_vs_L[LMAX] = focs


# %% compute distribution of focality vs. L
#
# limit complete multipole based leadfield to given L value (the "old" way)
#
LMAX = 20
L_RANGE = np.arange(
    1, LMAX + 1
)  # restrict source and leadfield to a spatial frequency maximum
focs_vs_L = dict()

# src_inds = superf_source_inds  # select source indices to use
src_inds = np.arange(nsrc_valid_thishemi)
nsrcs = len(src_inds)
print(f'{nsrcs=}')

for this_L in L_RANGE:
    print(f'{this_L=}')
    theLs = list(range(1, this_L + 1))
    include = np.where(np.isin(basisvec_L, theLs))[0]
    xin_leads_filt = xin_leads[include]
    focs = list()
    inv_sols = list()

    for k, src_ind in enumerate(src_inds):
        # get the multipole forward field for this source
        src_lead = xin_leads_filt[:, src_ind]
        # do the mne inverse in multipole space
        inv_sol = _min_norm_pinv(xin_leads_filt, src_lead, method='pinv')
        inv_sols.append(inv_sol)
        node_strength = _scalarize_src_data(inv_sol, nverts_thishemi)
        focality = len(np.where(node_strength > node_strength.max() / 3)[0])
        focs.append(focality)
        if k % 100 == 0:
            print(f'computing inverses: {100 * k / nsrcs:.1f}%')
    focs = np.array(focs).astype(np.float)
    focs_vs_L[this_L] = focs

focs_vs_L_array = np.array(list(focs_vs_L.values()))
plt.plot(L_RANGE, focs_vs_L_array[:, superf_source_inds])
plt.ylim([0, 200])


# %% compute distribution of focality vs. L
#
# method: estimate multipoles using limited L values ("proper" way?)
#
LMAX = 20
L_RANGE = np.arange(
    1, LMAX + 1
)  # restrict source and leadfield to a spatial frequency maximum
focs_vs_L = dict()

src_inds = superf_source_inds  # select source indices to use
nsrcs = len(src_inds)
print(f'{nsrcs=}')

for this_L in L_RANGE:
    print(f'{this_L=}')
    theLs = list(range(1, this_L + 1))
    # create limited-L basis and fit sensor-space leadfield to it
    include = np.where(np.isin(basisvec_L, theLs))[0]
    Sin_this = Sin[:, include]
    xin_leads_this = np.squeeze(np.linalg.pinv(Sin_this) @ leads_thishemi_sc)

    # compute focality values for the given sources and current L
    focs = list()
    inv_sols = list()
    for k, src_ind in enumerate(src_inds):
        # the multipole forward for this source
        src_lead = xin_leads_this[:, src_ind]
        # do the mne inverse in multipole space
        inv_sol = _min_norm_pinv(xin_leads_this, src_lead, method='pinv')
        inv_sols.append(inv_sol)
        node_strength = _scalarize_src_data(inv_sol, nverts_thishemi)
        focality = len(np.where(node_strength > node_strength.max() / 3)[0])
        focs.append(focality)
        if k % 100 == 0:
            print(f'computing inverses: {100 * k / nsrcs:.1f}%')
    focs = np.array(focs).astype(np.float)
    focs_vs_L[this_L] = focs

focs_vs_L_array = np.array(list(focs_vs_L.values()))


# %% curve plot of focality vs. L
plt.plot(L_RANGE, focs_vs_L_array)
plt.plot(L_RANGE, np.median(focs_vs_L_array, axis=1), 'k--')
plt.ylim([0, 50])
plt.xticks(L_RANGE)
plt.xlabel('Lin')
plt.ylabel('Source focality')


# %% montaged cortical plots of focality vs. L
#
# # for unconstrained sources we have 3 focality measures for each node
# and will apply reducer_fun to get one value per node
#
SAVE_FIGS = True
FIG_BG_COLOR = (1.0, 1.0, 1.0)
FIGSIZE = (400, 300)
NCOLS_MAX = 4
REDUCER_FUN = np.min
FMIN, FMAX = 0, 30
if not FIX_ORI:
    src_color_data = REDUCER_FUN(focs.reshape(-1, 3), axis=1)
else:
    src_color_data = focs

mlab.options.offscreen = SAVE_FIGS
fignames = list()
brains = list()

for k, L in enumerate(range(1, LIN + 1)):
    src_color_data = focs_vs_L_array[k]
    fig = mlab.figure()
    brain = Brain(
        subject, HEMI, surf, subjects_dir=subjects_dir, background='white', figure=fig
    )
    colorbar = L == LIN  # only on the last plot
    brain.add_data(
        src_color_data,
        vertices=src_vertices[HEMI_IND],
        colormap='plasma_r',
        hemi=HEMI,
        colorbar=colorbar,
    )
    _rescale_brain_colormap(brain, src_color_data, fmin=FMIN, fmax=FMAX)
    mlab.title(f'{L=}', size=1.2, height=0.9)
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
    montage_fn = 'foctest.png'
    _montage_figs(fignames, montage_fn, ncols_max=NCOLS_MAX)
del fignames


# %% WIP: alternative viz of focality (tries to show depth also)
fig = mlab.figure()
_mlab_colorblobs(src_coords_thishemi, src_color_data, normalize_data=True, figure=fig)
mlab.colorbar(fig)


# %% plot the mne inverse for a given source
src_label = 'lateral'
src_node = test_nodes[src_label]  # index into cortical locations
src_vec = test_sources[src_label]
src_lead = xin_leads @ src_vec
inv_sol = _min_norm_pinv(xin_leads, src_lead, method='pinv')
src_color_data = _scalarize_src_data(inv_sol, nverts_thishemi)

FMIN = 0
FMAX = np.quantile(src_color_data, 0.98, axis=None)
# FMAX = 6e-3

RESCALE_COLORMAP = True

fig = mlab.figure()
brain = Brain(
    subject, HEMI, surf, subjects_dir=subjects_dir, background='white', figure=fig
)
# add color coded data
colorbar = True
brain.add_data(
    src_color_data,
    vertices=src_vertices[HEMI_IND],
    colormap='plasma',
    hemi=HEMI,
    colorbar=colorbar,
)
if RESCALE_COLORMAP:
    fmin, fmax = (FMIN, FMAX)
    fmid = (fmin + fmax) / 2
    brain.scale_data_colormap(
        fmin=fmin, fmid=fmid, fmax=fmax, transparent=False, verbose=False
    )

con_str = 'constrained' if FIX_ORI else 'unconstrained'
mlab.title(f'{con_str} inverse for source node {src_node}', size=0.45, height=0.9)
