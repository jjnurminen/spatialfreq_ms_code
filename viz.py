#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utils.

@author: jussi
"""

import numpy as np
from mayavi import mlab
from surfer import Brain
import mne
from mne.transforms import _cart_to_sph, _pol_to_cart, apply_trans
from mne.io.constants import FIFF
from scipy.spatial import Delaunay

from miscutils import _named_tempfile, _montage_figs


def _info_meg_locs(info):
    """Return sensor locations for MEG sensors"""
    return np.array(
        [
            info['chs'][k]['loc'][:3]
            for k in range(info['nchan'])
            if info['chs'][k]['kind'] == FIFF.FIFFV_MEG_CH
        ]
    )


def _delaunay_tri(rr):
    """Surface triangularization based on 2D proj and Delaunay"""
    # this is a straightforward projection to xy plane
    com = rr.mean(axis=0)
    rr = rr - com
    xy = _pol_to_cart(_cart_to_sph(rr)[:, 1:][:, ::-1])
    # do Delaunay for the projection and hope for the best
    return Delaunay(xy).simplices


def _get_plotting_inds(info):
    """Get magnetometer inds only, if they exist. Otherwise, get gradiometer inds."""
    mag_inds = mne.pick_types(info, meg='mag')
    grad_inds = mne.pick_types(info, meg='grad')
    # if mags exist, use only them
    if mag_inds.size:
        inds = mag_inds
    elif grad_inds.size:
        inds = grad_inds
    else:
        raise RuntimeError('no MEG sensors found')
    return inds


def _make_array_tri(info, to_headcoords=True):
    """Make triangulation of array for topography views.
    If array has both mags and grads, triangulation will be based on mags
    only. Corresponding sensor indices are returned as inds.
    If to_headcoords, returns the sensor locations in head coordinates."""
    inds = _get_plotting_inds(info)
    locs = _info_meg_locs(info)[inds, :]
    if to_headcoords:
        locs = apply_trans(info['dev_head_t'], locs)
    locs_tri = _delaunay_tri(locs)
    return inds, locs, locs_tri


def _mlab_quiver3d(rr, nn, **kwargs):
    """Plots vector field as arrows.
    rr : (N x 3) array-like
        The locations of the vectors.
    nn : (N x 3) array-like
        The vectors.
    """
    vx, vy, vz = rr.T
    u, v, w = nn.T
    return mlab.quiver3d(vx, vy, vz, u, v, w, **kwargs)


def _mlab_points3d(rr, *args, **kwargs):
    """Plots points.
    rr : (N x 3) array-like
        The locations of the vectors.
    Note that the api to mayavi points3d is weird, there is no way to specify colors and sizes
    individually. See:
    https://stackoverflow.com/questions/22253298/mayavi-points3d-with-different-size-and-colors
    """
    vx, vy, vz = rr.T
    return mlab.points3d(vx, vy, vz, *args, **kwargs)


def _mlab_trimesh(pts, tris, **kwargs):
    """Plots trimesh specified by pts and tris into given figure.
    pts : (N x 3) array-like
    """
    x, y, z = pts.T
    return mlab.triangular_mesh(x, y, z, tris, **kwargs)


def _montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    src_datas,
    titles,
    src_vertices,
    hemi,
    fn_out,
    surf=None,
    thresh=None,
    smoothing_steps=None,
    frange=None,
    ncols_max=None,
    colormap=None,
    colorbar_nlabels=None,
    title_width=None,
    do_colorbar=True,
):
    """Create and montage several PySurfer -based plots into a .png file.

    Parameters:
    -----------
    subject : str
    subjects_dir : str
    src_datas : list
        Source-based scalar data arrays, one for each figure.
    titles : list
        Corresponding figure titles.
    frange : tuple | str | None
        Use tuple of (fmin, fmax) to set min and max values for scaling the colormap.
        If None, use global min/max of source data.
        If 'separate', each plot will be individually auto-scaled.
    src_vertices : array
        Indices of source vertices.
    hemi : str
        Hemisphere to plot ('lh' or 'rh').
    fn_out : str
        Name of .png file to write
    ncols_max : int
        Max. n of columns in montaged figure.
    """
    FIG_BG_COLOR = (1.0, 1.0, 1.0)
    brains = list()  # needed to retain refs to the PySurfer figs
    if surf is None:
        surf = 'inflated'
    FIGSIZE = (400, 300)  # size of a single figure (pixels)
    if ncols_max is None:
        ncols_max = 4
    if colormap is None:
        colormap = 'plasma'
    if colorbar_nlabels is None:
        colorbar_nlabels = 6  # default is too many
    if title_width is None:
        title_width = 0.5

    colorbar_fontsize = int(FIGSIZE[0] / 24)  # heuristic

    nfigs = len(src_datas)

    if frange is None:
        fmax, fmin = np.max(src_datas), np.min(src_datas)
    elif isinstance(frange, str) and frange == 'separate':
        pass
    else:
        fmin, fmax = frange

    mlab.options.offscreen = True
    fignames = list()
    assert len(titles) == nfigs

    for src_data, title, idx in zip(src_datas, titles, range(nfigs)):
        fig = mlab.figure()
        brain = Brain(
            subject,
            hemi,
            surf,
            subjects_dir=subjects_dir,
            background='white',
            figure=fig,
        )
        brains.append(brain)
        plot_colorbar = (
            do_colorbar and idx == nfigs - 1
        )  # only add colorbar to last figure
        brain.add_data(
            src_data,
            vertices=src_vertices,
            colormap=colormap,
            hemi=hemi,
            thresh=thresh,
            colorbar=plot_colorbar,
            smoothing_steps=smoothing_steps,
        )
        if plot_colorbar:
            # we need to dive deep into the brain to get a handle on the colorbar
            cb = brain._data_dicts[hemi][0]['colorbars'][0]
            cb.label_text_property.bold = 0
            cb.scalar_bar.unconstrained_font_size = True
            cb.scalar_bar.number_of_labels = colorbar_nlabels
            cb.label_text_property.font_size = colorbar_fontsize

        if frange != 'separate':
            fmid = (fmin + fmax) / 2
            brain.scale_data_colormap(
                fmin=fmin, fmid=fmid, fmax=fmax, transparent=False, verbose=False
            )
        mlab.text(0.1, 0.8, title, width=title_width)
        # view for an occipital source
        #mlab.view(-119.5036519701047, 90.65370784316794, 527.0917968750098, [0., 0., 0.])      

        # temporarily save fig for the montage
        fname = _named_tempfile(suffix='.png')
        print(f'creating figure {idx+1}/{nfigs}')
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)
        mlab.close(fig)

    # complete the montage using empty figures, so that the background is
    # consistent
    n_last = ncols_max - nfigs % ncols_max  # n of figs on last row
    # do not fill first row, if it's the only one
    if n_last == ncols_max or nfigs < ncols_max:
        n_last = 0
    for _ in range(n_last):
        fig = mlab.figure(bgcolor=FIG_BG_COLOR)
        fname = _named_tempfile(suffix='.png')
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)

    mlab.options.offscreen = False  # restore
    _montage_figs(fignames, fn_out, ncols_max=ncols_max)


def _montage_mlab_trimesh(
    locs, tri, src_datas, titles, fn_out, ncols_max=None, distance=None
):
    """Montage trimesh plots"""

    # FIG_BG_COLOR = (0.3, 0.3, 0.3)
    FIG_BG_COLOR = (1.0, 1.0, 1.0)
    FIGSIZE = (400, 300)

    if ncols_max is None:
        ncols_max = 4

    if distance is None:
        distance = 0.6  # view distance

    nfigs = len(src_datas)

    mlab.options.offscreen = True

    fignames = list()
    for src_data, title, idx in zip(src_datas, titles, range(nfigs)):
        fig = mlab.figure(bgcolor=FIG_BG_COLOR)
        _mlab_trimesh(locs, tri, scalars=src_data, figure=fig)
        mlab.view(distance=distance)
        # mlab.title(title, color=(0., 0., 0.))
        # HACK: adapt text width to title length
        text_width = 0.04 * len(title)
        mlab.text(0, 0.8, title, width=text_width, color=(0.0, 0.0, 0.0))
        # save fig for the montage
        fname = _named_tempfile(suffix='.png')
        print(f'creating figure {idx}/{nfigs}')
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)
        mlab.close(fig)

    # complete the montage using empty figures, so that the background is
    # consistent
    n_last = ncols_max - nfigs % ncols_max
    if n_last == ncols_max:
        n_last = 0
    for _ in range(n_last):
        fig = mlab.figure(bgcolor=FIG_BG_COLOR)
        fname = _named_tempfile(suffix='.png')
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)

    mlab.options.offscreen = False  # restore
    _montage_figs(fignames, fn_out, ncols_max=ncols_max)
