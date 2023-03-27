#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field computation & sensor related stuff

@author: jussi
"""

import numpy as np
import mne
from mne.io.constants import FIFF
from mne.transforms import rotation3d_align_z_axis, _deg_ord_idx
from mne.preprocessing.maxwell import _sss_basis
import scipy
import copy
from functools import reduce

from misc import _rot_around_axis_mat


def _info_meg_locs(info):
    """Return sensor locations for MEG sensors"""
    return np.array(
        [
            info['chs'][k]['loc'][:3]
            for k in range(info['nchan'])
            if info['chs'][k]['kind'] == FIFF.FIFFV_MEG_CH
        ]
    )


def _info_meg_normals(info):
    """Return sensor normal vectors for MEG sensors"""
    Sn = np.zeros((info['nchan'], 3))
    for k in range(info['nchan']):
        rotm = info['chs'][k]['loc'][3:].reshape((3, 3)).T
        Sn[k, :] = rotm @ np.array([0, 0, 1])
        Sn[k, :] /= np.linalg.norm(Sn[k, :])
    return Sn


def _sensordata_to_ch_dicts(Sc, Sn, Iprot, coiltypes):
    """Convert sensor data from Sc (Mx3 locations) and Sn (Mx3 normals) into
    mne channel dicts (e.g. info['chs'][k]"""
    locs = _sensordata_to_loc(Sc, Sn, Iprot)
    for k, (loc, coiltype) in enumerate(zip(locs, coiltypes)):
        ch = dict()
        number = k + 1
        ch['loc'] = loc
        ch['ch_name'] = 'MYMEG %d' % number
        ch['coil_type'] = coiltype
        ch['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
        ch['kind'] = FIFF.FIFFV_MEG_CH
        ch['logno'] = number
        # these two apply only to data read from the disk, so shouldn't matter
        ch['cal'] = 1
        ch['range'] = 1
        ch['scanno'] = number
        ch['unit'] = FIFF.FIFF_UNIT_T
        ch['unit_mul'] = FIFF.FIFF_UNITM_NONE
        yield ch


def _scale_array(info, scale_factor):
    """Scales the array (sensor locations) inplace by given factor. Does not affect coil data"""
    for k in range(info['nchan']):
        info['chs'][k]['loc'][:3] *= scale_factor


def _sensordata_to_loc(Sc, Sn, Iprot):
    """Convert sensor data from Sc (Mx3 locations) and Sn (Mx3 normals) into
    mne loc matrices, used in e.g. info['chs'][k]['loc']. Integration data is
    handled separately via the coil definitions.

    Sn is the desired sensor normal, used to align the xy-plane integration
    points. Iprot (Mx1, degrees) can optionally be applied to first rotate the
    integration point in the xy plane. Rotation is CCW around z-axis.
    """
    assert Sn.shape[0] == Sc.shape[0]
    assert Sn.shape[1] == Sc.shape[1] == 3
    for k in range(Sc.shape[0]):
        # get rotation matrix corresponding to desired sensor orientation
        R2 = rotation3d_align_z_axis(Sn[k, :])
        # orient integration points in their xy plane
        R1 = _rot_around_axis_mat([0, 0, 1], Iprot[k])
        rot = R2 @ R1
        loc = np.zeros(12)
        loc[:3] = Sc[k, :]  #  first 3 elements are the loc
        loc[3:] = rot.T.flat  # next 9 elements are the flattened rot matrix
        yield loc


def _split_leadfield(forward):
    """Splits leadfield of a forward model into left and right hemispheres.
    Return dict with keys 0 = left hemi, 1 = right hemi"""
    leadfld = forward['sol']['data']
    dim_left = forward['src'][0]['nuse']
    dim_right = forward['src'][1]['nuse']
    # for unconstrained solution, leadfield will have 3*N_SOURCES entries
    assert leadfld.shape[1] == dim_left + dim_right or leadfld.shape[1] == 3 * (
        dim_left + dim_right
    )
    split_ind = dim_left if leadfld.shape[1] == dim_left + dim_right else 3 * dim_left
    leadfld_left, leadfld_right = np.split(leadfld, [split_ind], axis=1)
    return {0: leadfld_left.astype(np.float64), 1: leadfld_right.astype(np.float64)}


def _split_normals(forward):
    """Splits normal vectors of a forward model into left and right hemispheres.
    Return dict with keys 0 = left hemi, 1 = right hemi."""
    normals = forward['source_nn']
    dim_left = forward['src'][0]['nuse']
    dim_right = forward['src'][1]['nuse']
    # for unconstrained solution, normals will have 3*N_SOURCES entries
    assert normals.shape[0] == dim_left + dim_right or normals.shape[0] == 3 * (
        dim_left + dim_right
    )
    split_ind = dim_left if normals.shape[0] == dim_left + dim_right else 3 * dim_left
    normals_left, normals_right = np.split(normals, [split_ind], axis=0)
    return {0: normals_left, 1: normals_right}


def _hemi_slice(hemi, nsrc_valid):
    """Return a slice for picking hemi-specific data from a data array.
    Number of sources for each hemi are N0 and N1.

    Parameters:
    -----------
    hemi : int
        Index for desired hemisphere (0 or 1).
    nsrc_valid : dict
        Dict containing the number of sources for each hemi {0: N0, 1: N1}
    """
    ind0 = 0 if hemi == 0 else nsrc_valid[0]
    ind1 = nsrc_valid[0] if hemi == 0 else nsrc_valid[0] + nsrc_valid[1]
    return slice(ind0, ind1)


def _get_shifted_forwards(
    subject,
    info,
    trans_file,
    subjects_dir,
    use_bem=None,
    head_origin=None,
    spacing='oct6',
    source_shift=None,
    fix_ori=True,
):
    """Create a cortical source space and compute forwards for unshifted
    and (x,y,z) shifted source spaces. Fixed-orientation forwards are returned.
    Returns forwards with compatible dimensions (i.e. sources that end up outside
    the inner skull surface on one particular shift are dropped from all forwards)

    Parameters:
    -------
    source_shift : float
        How much to shift sources (m), positive

    Returns:
    --------
    fwds: dict
        The fixed orientation forwards: 'unshifted' and 'x', 'y', 'z'
    valid_vertex_indices : dict
        Hemi-specific dicts of vertex numbers (indices into source vertex
        matrix) that are valid for ALL shifted forwards. (For BEM, some sources
        may end up outside the inner skull when shifted. Thus, resulting forward
        matrices may be of different dimensions and cannot be directly
        compared.)
    """
    if source_shift is None:
        source_shift = 5e-3
    elif source_shift < 0:
        raise ValueError('shift must be positive')

    # first create the volume source space
    src_cort = mne.setup_source_space(
        subject, spacing=spacing, subjects_dir=subjects_dir, add_dist=False
    )

    # shift sources in each dim in turn and compute the corresponding forwards
    fwds_cort_surfori_shifted = dict()
    for shift_dim in ['unshifted', 0, 1, 2]:
        src_cort_ = copy.deepcopy(src_cort)  # get an unaltered copy
        if shift_dim != 'unshifted':
            # shift sources of each hemi separately
            for hemi_ind in [0, 1]:
                shiftvec = np.zeros(3)  # the source shift vector
                shiftvec[shift_dim] = source_shift
                _coords_thishemi = src_cort_[hemi_ind]['rr']
                _coords_midpoint = _coords_thishemi.mean(axis=0)[shift_dim]
                # x dim is special due to hemispheric asymmetry,
                # so there we shift sources towards the midline (x=0)
                if shift_dim == 0:
                    if _coords_midpoint < 0:
                        _coords_thishemi += shiftvec
                    else:
                        _coords_thishemi -= shiftvec
                else:
                    # y and z dims are shifted towards the geometric mean of the vertex set
                    posi = np.where(_coords_thishemi[:, shift_dim] > _coords_midpoint)[
                        0
                    ]
                    negi = np.where(_coords_thishemi[:, shift_dim] < _coords_midpoint)[
                        0
                    ]
                    _coords_thishemi[negi, :] += shiftvec
                    _coords_thishemi[posi, :] -= shiftvec

        # compute fwd for (possibly) shifted sources
        if use_bem:
            fwd_cort = mne.make_forward_solution(
                info, trans_file, src_cort_, use_bem, eeg=False
            )
        else:
            # default sphere model
            if head_origin is None:
                head_origin = (0.0, 0.0, 0.04)
            sphere = mne.make_sphere_model(r0=head_origin, head_radius=None)
            fwd_cort = mne.make_forward_solution(
                info, trans_file, src_cort_, sphere, eeg=False
            )

        if fix_ori:
            # convert to fixed orientation fwd
            fwd_cort = mne.convert_forward_solution(
                fwd_cort, surf_ori=True, force_fixed=True, copy=True
            )
        fwds_cort_surfori_shifted[shift_dim] = fwd_cort

    # get numbers of vertices that are valid (fwd sol exists) for all the shifted forwards
    # these are indices into the full vertex matrix
    valid_vertices = dict()
    for hemi_ind in [0, 1]:
        nverts_ = [
            fwd['src'][hemi_ind]['vertno'] for fwd in fwds_cort_surfori_shifted.values()
        ]
        valid_vertices[hemi_ind] = reduce(np.intersect1d, nverts_)

    # get indices of valid vertices (as defined above) for each fwd
    # these are indices into the small vertex set (=source vertices) of each fwd,
    # and can be used to index the forward solution
    valid_vertex_indices = dict()
    for key, fwd in fwds_cort_surfori_shifted.items():
        valid_vertex_indices[key] = dict()
        for hemi_ind in [0, 1]:
            _mask_valid = np.isin(
                fwd['src'][hemi_ind]['vertno'], valid_vertices[hemi_ind]
            )
            valid_vertex_indices[key][hemi_ind] = np.where(_mask_valid)[0]

    return fwds_cort_surfori_shifted, valid_vertex_indices, valid_vertices


def _scale_magmeters(data, info, scale):
    """Scales magnetometers in data (nchan x M) if necessary"""
    data = data.copy()
    mag_inds = mne.pick_types(info, meg='mag')
    grad_inds = mne.pick_types(info, meg='grad')
    if data.ndim == 1:
        data = data[:, None]
    # scale only if both types of sensors exist
    if not (grad_inds.size and mag_inds.size):
        return data
    if data.shape[0] != info['nchan']:
        raise ValueError('data has unexpected dims')
    data[mag_inds, :] *= scale
    return data


def _sss_basis_nvecs(L):
    """Return number of vecs for SSS int/ext basis of order L"""
    return L * (L + 2)


def basis_dim(n):
    """Inverse of _sss_basis_nvecs(), i.e. L for given number of vectors"""
    L = np.sqrt(1 + n) - 1
    if L >= 0 and int(L) == L:
        return int(L)
    else:
        raise ValueError('invalid number of basis vectors')


def _idx_deg_ord(idx):
    """Returns (degree, order) tuple for a given multipole index."""
    # this is just an ugly inverse of _deg_ord_idx, do not expect speed
    for deg in range(1, 60):
        for ord in range(-deg, deg + 1):
            if _deg_ord_idx(deg, ord) == idx:
                return deg, ord
    return None


def _decompose_sigvec(sigvec, Sin, basisvec_L):
    """Decompose signal vector of internal subspace into its L components.

    First solves all multipole coeffs (using the total basis), then reconstructs
    the signal parts corresponding to each L.

    Parameters:
    -----------
    sigvec: array of (Nchan,)
        The signal vector, mags scaled as in Sin
    Sin: array of (Nchan x nvecs)
        (Possibly optimized) SSS basis
    basisvec_L: list or array
        Value of L for each basis vector in Sin

    Returns:
    --------
    sigvec_decomp: Nchan x Lmax
        The decomposition for each L
    """
    nchan = len(sigvec)
    basisvec_L = np.array(basisvec_L)
    Lmax = max(basisvec_L)
    # first solve for all coefficients
    xin = np.squeeze(np.linalg.pinv(Sin) @ sigvec)
    sigvec_decomp = np.zeros((nchan, Lmax))
    # then get the part corresponding to each L
    for L in range(1, Lmax + 1):
        inds = np.where(basisvec_L == L)[0]
        if inds.size == 0:
            sigvec_this = np.zeros((nchan,))
        else:
            Sin_this = Sin[:, inds]
            xin_this = xin[inds]
            sigvec_this = Sin_this @ xin_this
        sigvec_decomp[:, L - 1] = sigvec_this
    return sigvec_decomp


def _limit_L(sigvec, Sin, basisvec_L):
    """Reconstruct signal vector with limited multipole components.

    Different from _decompose_sigvec() which fits all L components, then
    reconstructs the signal using a limited set only. This one picks a limited
    set corresponding to each L and fits it.

    Parameters:
    -----------
    sigvec: array of (Nchan,)
        The signal vector, mags scaled as in Sin
    Sin: array of (Nchan x nvecs)
        (Possibly optimized) SSS basis
    basisvec_L: list or array
        Value of L for each basis vector in Sin

    Returns:
    --------
    sigvec_limited: Nchan x Lmax
        L-limited reconstruction for each L.
    """
    nchan = len(sigvec)
    basisvec_L = np.array(basisvec_L)
    Lmax = max(basisvec_L)
    sigvec_lim = np.zeros((nchan, Lmax))
    for L in range(1, Lmax + 1):
        inds = np.where(basisvec_L <= L)[0]
        Sin_this = Sin[:, inds]
        xin_this = np.squeeze(np.linalg.pinv(Sin_this) @ sigvec)
        sigvec_this = Sin_this @ xin_this
        sigvec_lim[:, L - 1] = sigvec_this
    return sigvec_lim


def _min_norm_pinv(A, b, method='tikhonov', tikhonov_lambda=0, rcond=1e-15):
    """Minimum norm pseudoinverse (MNP) inverse solution"""
    nsensors = A.shape[0]
    if method == 'tikhonov':
        # naive Tikhonov reg
        mnp = A.T @ np.linalg.inv(A @ A.T + tikhonov_lambda * np.eye(nsensors)) @ b
    elif method == 'tikhonov_svd':
        if b.ndim > 1:
            raise ValueError('Tikhonov SVD needs a vector input for b')
        mnp = tikhonov_svd(A, b, tikhonov_lambda)
    elif method == 'pinv':
        mnp = np.linalg.pinv(A, rcond=rcond) @ b
    elif method == 'unreg':
        mnp = np.linalg.pinv(A, rcond=1e-25) @ b
    elif method == 'lstsq':
        mnp = np.linalg.lstsq(A, b)[0]
    elif method == 'scipy_lstsq':
        mnp = scipy.linalg.lstsq(A, b, lapack_driver='gelsy')[0]
    else:
        raise RuntimeError('Invalid method')
    return mnp


def tikhonov_svd(A, b, _lambda):
    """Solve Ax=b for x using Tikhonov regularization. Implementation via SVD"""
    U, D, Vh = scipy.linalg.svd(A, full_matrices=False)  # XXX: scipy
    # compute SVD filter factors for Tikhonov
    # see https://en.wikipedia.org/wiki/Tikhonov_regularization
    f_tikh = D**2 / (_lambda + D**2)
    return sum(f_tikh[k] * U[:, k] @ b * Vh[k, :] / D[k] for k in range(len(D)))


def _scale_ips(ips, coilslices, inds, scaling_factor):
    """Scale up sensors given by indices inds.

    This modifies the integration points (ips) in place.
    ips and the coil indexing matrix coilslices are given by
    mne function _prep_mf_coils().
    """
    for ind in inds:
        coilslice = coilslices[ind]
        ips_this = ips[coilslice]
        ctr_this = ips_this.mean(axis=0)
        ips_this -= ctr_this
        ips_this *= scaling_factor
        ips_this += ctr_this


def _prep_mf_coils_pointlike(rmags, nmags):
    """Prepare the coil data for pointlike magnetometers.

    rmags, nmags are sensor locations and normals respectively, with shape (N,3).
    """
    n_coils = rmags.shape[0]
    mag_mask = np.ones(n_coils).astype(bool)
    slice_map = {k: slice(k, k + 1, None) for k in range(n_coils)}
    bins = np.arange(n_coils)
    return rmags, nmags, bins, n_coils, mag_mask, slice_map


def _normalized_basis(rmags, nmags, sss_params):
    """Compute normalized SSS basis matrices for a pointlike array."""
    allcoils = _prep_mf_coils_pointlike(rmags, nmags)
    S = _sss_basis(sss_params, allcoils)
    S /= np.linalg.norm(S, axis=0)  # normalize basis
    nvecs_in = sss_params['int_order'] ** 2 + 2 * sss_params['int_order']
    Sin, Sout = S[:, :nvecs_in], S[:, nvecs_in:]
    return S, Sin, Sout


def _sssbasis_cond_pointlike(rmags, nmags, sss_params, cond_type='int'):
    """Calculate basis matrix condition for a pointlike array.

    cond : str
        Which condition number to return. 'total' for whole basis, 'int' for
        internal basis, 'l_split' for each L order separately, 'l_cumul' for
        cumulative L orders, 'single' for individual basis vectors.
    """
    Lin = sss_params['int_order']
    S, Sin, _ = _normalized_basis(rmags, nmags, sss_params)
    if cond_type == 'total':
        cond = np.linalg.cond(S)
    elif cond_type == 'int':
        cond = np.linalg.cond(Sin)
    elif cond_type == 'l_split' or cond_type == 'l_cumul':
        cond = list()
        for L in np.arange(1, Lin + 1):
            ind0 = _deg_ord_idx(L, -L) if cond_type == 'l_split' else 0
            ind1 = _deg_ord_idx(L, L)
            cond.append(np.linalg.cond(Sin[:, ind0 : ind1 + 1]))
    elif cond_type == 'single':
        cond = list()
        for v in np.arange(Sin.shape[0]):
            cond.append(np.linalg.cond(Sin[:, 0 : v + 1]))
    else:
        raise ValueError('invalid cond argument')
    return cond


def _scalarize_src_data(fwd, nverts, posdef=True, reducer_fun=None):
    """Convert a forward (or other source-based measure) to scalar for visualization.

    Shape of fwd can be either (nverts,) for constrained solutions or
    (3*nverts,) for unconstrained. In the former case, the absolute value is
    returned if posdef=True (default). In the latter case, regular 2-norms of
    3-component vectors are returned by default. That is, giving fwd as
    [vx, vy, vz, wx, wy, wz,...] will return [||v||, ||w||,...]
    This corresponds to reducer_fun=np.linalg.norm. Other reducers can be supplied
    as args. They should take a 2-D matrix and an axis argument.
    """
    if reducer_fun is None:
        reducer_fun = np.linalg.norm
    fwd = np.array(fwd)
    if fwd.shape[0] == nverts:  # scalar valued
        return np.abs(fwd) if posdef else fwd
    elif fwd.shape[0] == 3 * nverts:
        fwd_3 = fwd.reshape(-1, 3)  # convert to Nx3
        return reducer_fun(fwd_3, axis=1)
    else:
        raise ValueError('shape of forward does not match nverts')


def _node_to_source_index(index, fixed_ori):
    """For unconstrained sources, there are N source locations (nodes) and N*3 sources.
    This converts node indices to corresponding source indices.

    E.g. for constrained (fixed_ori=True):
    node index 101 -> source index 101
    for unconstrained:
    node index 1 -> source indices 3,4,5
    """
    if fixed_ori:
        return index
    else:
        return np.arange(3 * index, 3 * index + 3)


def _resolution_kernel(leadfld, method='tikhonov', tikhonov_lambda=0, rcond=1e-15):
    """MNP resolution kernel from leadfield matrix.

    The resolution kernel contains inverse solutions for each elementary source
    (leadfield element) in the noiseless case. It depends on the exact inverse
    method (also regularization etc.)
    """
    return _min_norm_pinv(
        leadfld, leadfld, method=method, tikhonov_lambda=tikhonov_lambda, rcond=rcond
    )


def _spatial_dispersion(res_kernel, src_dij):
    """Compute spatial dispersion for point spread functions of a resolution
    kernel.

    src_dij: Euclidian distance matrix (nsrc x nsrc) of the sources
    """
    sd = list()
    for i in np.arange(src_dij.shape[0]):
        sdi = np.sqrt(
            np.sum(src_dij[i, :] ** 2 * res_kernel[i, :] ** 2)
            / np.sum(res_kernel[i, :] ** 2)
        )
        sd.append(sdi)
    return np.array(sd)


def _focality(res_kernel):
    """Compute focality for the point spread functions of a resolution
    kernel."""
    focs = list()
    for i in np.arange(res_kernel.shape[0]):
        thre = res_kernel[i, :].max() / 3
        foc = len(np.where(res_kernel[i, :] > thre)[0])
        focs.append(foc)
    return focs
