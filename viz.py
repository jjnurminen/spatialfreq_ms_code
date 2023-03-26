#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misc visualization.

@author: jussi
"""

import numpy as np
from mayavi import mlab
from surfer import Brain
from megsimutils.fileutils import _named_tempfile, _montage_figs
from megsimutils.viz import _mlab_points3d, _mlab_quiver3d
from trimesh.visual import color


def _montage_pysurfer_brain_plots(
    subject,
    subjects_dir,
    src_datas,
    titles,
    src_vertices,
    hemi,
    fn_out,
    frange=None,
    ncols_max=None,
    colormap=None,
    colorbar_nlabels=None,
    title_width=None,
    do_colorbar=True,
):
    """Montage several PySurfer -based plots into a .png file.

    src_datas : list of source-based scalar data arrays
    titles : corresponding figure titles
    frange : tuple of (fmin, fmax), min and max values for scaling the colormap
    src_vertices : indices of source vertices
    hemi : hemisphere to plot ('lh' or 'rh')
    fn_out : path of .png file to write
    ncols_max : max. n of columns in montaged figure
    """
    FIG_BG_COLOR = (1.0, 1.0, 1.0)
    brains = list()  # needed to retain refs to the PySurfer figs
    surf = 'white'
    FIGSIZE = (400, 300)  # size of a single figure (pixels)
    if ncols_max is None:
        ncols_max = 4
    if colormap is None:
        colormap = 'plasma'
    if colorbar_nlabels is None:
        colorbar_nlabels = 6  # default is too many
    if title_width is None:
        title_width = .5

    colorbar_fontsize = int(FIGSIZE[0] / 16)  # heuristic

    nfigs = len(src_datas)
    y_height = int(nfigs / ncols_max) * FIGSIZE[1]  # total y height
    title_size = y_height / 120  # what's the unit?

    if frange is None:
        fmax, fmin = np.max(src_datas), np.min(src_datas)
    elif isinstance(frange, str) and frange == 'separate':
        pass
    else:
        fmin, fmax = frange

    mlab.options.offscreen = True
    fignames = list()
    assert len(titles) == nfigs

    # convert possible path objects
    fn_out = str(fn_out)

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
        plot_colorbar = do_colorbar and idx == nfigs - 1  # only add colorbar to last figure
        brain.add_data(
            src_data,
            vertices=src_vertices,
            colormap=colormap,
            hemi=hemi,
            colorbar=plot_colorbar,
            smoothing_steps='nearest',
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
        mlab.text(.1, .8, title, width=title_width)

        # temporarily save fig for the montage
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
    for k in range(n_last):
        fig = mlab.figure(bgcolor=FIG_BG_COLOR)
        fname = _named_tempfile(suffix='.png')
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)

    mlab.options.offscreen = False  # restore
    _montage_figs(fignames, fn_out, ncols_max=ncols_max)


def _montage_mlab_brain_plots(
    src_datas,
    titles,
    src_coords,
    fn_out,
    frange=None,
    ncols_max=None,
):
    """Montage several mlab-based cortical plots into a .png file.

    XXX: COLORMAP SCALING IS STILL BROKEN!

    src_datas : list of source-based scalar data arrays
    titles : corresponding figure titles
    frange : tuple of (fmin, fmax), min and max values for scaling the colormap
    src_coords : indices of source vertices
    hemi : hemisphere to plot ('lh' or 'rh')
    ncols_max : max. n of columns in montaged figure
    """
    FIG_BG_COLOR = (0.3, 0.3, 0.3)
    FIGSIZE = (400, 300)
    pt_scale = 0.003
    if ncols_max is None:
        ncols_max = 4
    mlab.options.offscreen = True
    fignames = list()

    nfigs = len(src_datas)

    # scale the data
    # mlab (apparently) allocates the colormap for values between -1 and 1
    # if autoscaling, the data will be scaled to that range,
    # otherwise according to fmin, fmax (so data may saturate)
    if frange is not None:
        fmin, fmax = frange
    else:  # autoscale
        fmin, fmax = None, None
        for src_data in src_datas:
            if fmax is None or src_data.max() > fmax:
                fmax = src_data.max()
            if fmin is None or src_data.min() < fmin:
                fmin = src_data.min()
    print(f'{fmin=}')
    print(f'{fmax=}')

    src_datas_scaled = list()
    for _src_data in src_datas:
        src_data = _src_data.copy()
        src_data -= src_data.min()  # 0 -> max
        src_data /= 0.5 * (fmax - fmin)  # 0 -> 2
        src_data -= 1  # -1 -> 1
        src_datas_scaled.append(src_data)

    for src_data, title, idx in zip(src_datas_scaled, titles, range(nfigs)):
        fig = mlab.figure(bgcolor=FIG_BG_COLOR)
        nodes = _mlab_points3d(src_coords, figure=fig, scale_factor=pt_scale)
        nodes.glyph.scale_mode = 'scale_by_vector'
        nodes.mlab_source.dataset.point_data.scalars = src_data
        mlab.title(title)
        mlab.view(170, 50, roll=60)  # lateral view
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
    for k in range(n_last):
        fig = mlab.figure(bgcolor=FIG_BG_COLOR)
        fname = _named_tempfile(suffix='.png')
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)

    mlab.options.offscreen = False  # restore
    _montage_figs(fignames, fn_out, ncols_max=ncols_max)
    return src_datas_scaled


def _rescale_brain_colormap(
    brain, src_color_data, fmin=None, fmax=None, fmin_quantile=None, fmax_quantile=None
):
    """Scale color data of FreeSurfer brain object according to min/max values.

    For fmin and fmax, either absolute values or quantiles can be given.
    """
    if fmin is None:
        if fmin_quantile is None:
            fmin_quantile = 0.05
        fmin = np.quantile(src_color_data, fmin_quantile, axis=None)
    if fmax is None:
        if fmax_quantile is None:
            fmax_quantile = 0.95
        fmax = np.quantile(src_color_data, fmax_quantile, axis=None)
    fmid = (fmin + fmax) / 2
    brain.scale_data_colormap(
        fmin=fmin, fmid=fmid, fmax=fmax, transparent=False, verbose=False
    )
