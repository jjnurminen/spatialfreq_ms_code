#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common funcs

@author: jussi
"""

import numpy as np
from scipy.linalg import subspace_angles


def _prettyprint_xyz(v, multiplier=1e3, res_unit='mm'):
    """Pretty-print an xyz array"""
    return '(%.1f %.1f %.1f) %s' % (tuple(multiplier * v) + (res_unit,))


def _vector_angles(V, W):
    """Subspace angles in degrees between column vectors of V and W. V must be dxM
    and W either dxM (for pairwise angles) or dx1 (for all-to-one angles)"""
    assert V.shape[0] == W.shape[0]
    if V.ndim == 1:
        V = V[:, None]
    if W.ndim == 1:
        W = W[:, None]
    assert V.shape[1] == W.shape[1] or W.shape[1] == 1
    Vn = V / np.linalg.norm(V, axis=0)
    Wn = W / np.linalg.norm(W, axis=0)
    dots = np.sum(Vn * Wn, axis=0)
    dots = np.clip(dots, -1, 1)
    return np.arccos(dots) / np.pi * 180


def _mpl_plot3d(rr):
    from mpl_toolkits import mplot3d

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(rr[:, 0], rr[:, 1], rr[:, 2], 'o')
    # ax.set_aspect('equal')


def subspace_angles_deg(A, B):
    """Return subspace angles in degrees"""
    return subspace_angles(A, B) / np.pi * 180


def _normalize_columns(M):
    """Normalize column vectors of matrix M"""
    return M / np.linalg.norm(M, axis=0)


def _random_unit(N):
    """Return random unit vector in N-dimensional space"""
    v = np.random.randn(N)
    return v / np.linalg.norm(v)


def _find_points_in_range(pts, range):
    """Finds points in pts (Npts x 3) within given x, y, and z ranges
    Range should be a 3-tuple of (min, max), each in (x, y, z) dims
    or (None, None) if that dim does not matter
    """
    total_mask = np.ones(pts.shape[0], dtype=bool)
    for dim, (dim_min, dim_max) in enumerate(range):
        if dim_min is not None and dim_max is not None:
            dim_mask = np.logical_and(pts[:, dim] <= dim_max, pts[:, dim] >= dim_min)
            total_mask = np.logical_and(dim_mask, total_mask)
    return np.where(total_mask)[0]


def _rotate_to(v1, v2):
    """A matrix for rotating unit vector v1 to v2
    See https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    NB: may do weird things for some rotations (e.g. identity)
    """
    assert v1.shape == v2.shape == (3,)
    assert np.linalg.norm(v1) == np.linalg.norm(v2) == 1.0
    vs = v1 + v2
    return 2 * np.outer(vs, vs) / np.dot(vs, vs) - np.eye(3)


def _rot_around_axis_mat(a, theta):
    """Matrix for rotating right-handedly around vector a by theta degrees"""
    theta = theta / 180 * np.pi
    x, y, z = a[0], a[1], a[2]
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    return np.array(
        [
            [
                ctheta + (1 - ctheta) * x**2,
                (1 - ctheta) * x * y - stheta * z,
                (1 - ctheta) * x * z + stheta * y,
            ],
            [
                (1 - ctheta) * y * x + stheta * z,
                ctheta + (1 - ctheta) * y**2,
                (1 - ctheta) * y * z - stheta * x,
            ],
            [
                (1 - ctheta) * z * x - stheta * y,
                (1 - ctheta) * z * y + stheta * x,
                ctheta + (1 - ctheta) * z**2,
            ],
        ]
    )


def _rot_around_axis(v, a, theta):
    """Rotate v right-handedly around a by theta degrees"""
    return _rot_around_axis_mat(a, theta) @ v.T


def _spherepts_golden(N, angle=4 * np.pi):
    """Approximate uniformly distributed points on a unit sphere.

    This is the "golden ratio algorithm".
    See: http://blog.marmakoide.org/?p=1

    Parameters
    ----------
    n : int
        Number of points.

    angle : float
        Solid angle (symmetrical around z axis) covered by the points. By
        default, the whole sphere. Must be between 0 and 4*pi

    Returns
    -------
    ndarray
        (N, 3) array of Cartesian point coordinates.
    """
    # create linearly spaced azimuthal coordinate
    dlong = np.pi * (3 - np.sqrt(5))
    longs = np.linspace(0, (N - 1) * dlong, N)
    # create linearly spaced z coordinate
    z_top = 1
    z_bottom = 1 - 2 * (angle / (4 * np.pi))
    dz = (z_top - z_bottom) / N

    z = np.linspace(z_top - dz / 2, z_bottom + dz / 2, N)
    r = np.sqrt(1 - z**2)
    # this looks like the usual cylindrical -> Cartesian transform?
    return np.column_stack((r * np.cos(longs), r * np.sin(longs), z))


def _unit_impulse(N, n):
    """Return N-sample unit impulse that is 1 at index n"""
    z = np.zeros(N)
    z[n] = 1
    return z


def _moore_penrose_pseudoinverse(L):
    """Naive Moore-Penrose pseudoinverse (without regularization)"""
    return L.T @ np.linalg.inv(L @ L.T)
