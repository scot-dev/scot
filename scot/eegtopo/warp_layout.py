# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2015 Martin Billinger

"""
Summary
-------
Provides functions to warp electrode layouts.
"""

import numpy as np
import scipy as sp


def warp_locations(locations, y_center=None, return_ellipsoid=False, verbose=False):
    """ Warp EEG electrode locations to spherical layout.

    EEG Electrodes are warped to a spherical layout in three steps:
        1. An ellipsoid is least-squares-fitted to the electrode locations.
        2. Electrodes are displaced to the nearest point on the ellipsoid's surface.
        3. The ellipsoid is transformed to a sphere, causing the new locations to lie exactly on a spherical surface
           with unit radius.

    This procedure intends to minimize electrode displacement in the original coordinate space. Simply projecting
    electrodes on a sphere (e.g. by normalizing the x/y/z coordinates) typically gives much larger displacements.

    Parameters
    ----------
    locations : array-like, shape = [n_electrodes, 3]
        Eeach row of `locations` corresponds to the location of an EEG electrode in cartesian x/y/z coordinates.
    y_center : float, optional
        Fix the y-coordinate of the ellipsoid's center to this value (optional). This is useful to align the ellipsoid
        with the central electrodes.
    return_ellipsoid : bool, optional
        If `true` center and radii of the ellipsoid are returned.

    Returns
    -------
    newlocs : array-like, shape = [n_electrodes, 3]
        Electrode locations on unit sphere.
    c : array-like, shape = [3], (only returned if `return_ellipsoid` evaluates to `True`)
        Center of the ellipsoid in the original location's coordinate space.
    r : array-like, shape = [3], (only returned if `return_ellipsoid` evaluates to `True`)
        Radii (x, y, z) of the ellipsoid in the original location's coordinate space.
    """
    locations = np.asarray(locations)

    if y_center is None:
        c, r = _fit_ellipsoid_full(locations)
    else:
        c, r = _fit_ellipsoid_partial(locations, y_center)

    elliptic_locations = _project_on_ellipsoid(c, r, locations)

    if verbose:
        print('Head ellipsoid center:', c)
        print('Head ellipsoid radii:', r)
        distance = np.sqrt(np.sum((locations - elliptic_locations)**2, axis=1))
        print('Minimum electrode displacement:', np.min(distance))
        print('Average electrode displacement:', np.mean(distance))
        print('Maximum electrode displacement:', np.max(distance))

    spherical_locations = (elliptic_locations - c) / r

    if return_ellipsoid:
        return spherical_locations, c, r

    return spherical_locations


def _fit_ellipsoid_full(locations):
    """identify all 6 ellipsoid parametes (center, radii)"""
    a = np.hstack([locations*2, locations**2])
    lsq = sp.linalg.lstsq(a, np.ones(locations.shape[0]))
    x = lsq[0]
    c = -x[:3] / x[3:]
    gam = 1 + np.sum(x[:3]**2 / x[3:])
    r = np.sqrt(gam / x[3:])
    return c, r


def _fit_ellipsoid_partial(locations, cy):
    """identify only 5 ellipsoid parameters (y-center determined by e.g. Cz)"""
    a = np.vstack([locations[:, 0]**2,
                   locations[:, 1]**2 - 2 * locations[:, 1] * cy,
                   locations[:, 2]**2,
                   locations[:, 0]*2,
                   locations[:, 2]*2]).T
    x = sp.linalg.lstsq(a, np.ones(locations.shape[0]))[0]
    c = [-x[3] / x[0], cy, -x[4] / x[2]]
    gam = 1 + x[3]**2 / x[0] + x[4]**2 / x[2]
    r = np.sqrt([gam / x[0], gam / x[1], gam / x[2]])
    return c, r


def _project_on_ellipsoid(c, r, locations):
    """displace locations to the nearest point on ellipsoid surface"""
    p0 = locations - c  # original locations

    l2 = 1 / np.sum(p0**2 / r**2, axis=1, keepdims=True)
    p = p0 * np.sqrt(l2)  # initial approximation (projection of points towards center of ellipsoid)

    fun = lambda x: np.sum((x.reshape(p0.shape) - p0)**2)              # minimize distance between new and old points
    con = lambda x: np.sum(x.reshape(p0.shape)**2 / r**2, axis=1) - 1  # new points constrained to surface of ellipsoid
    res = sp.optimize.minimize(fun, p, constraints={'type': 'eq', 'fun': con}, method='SLSQP')

    return res['x'].reshape(p0.shape) + c
