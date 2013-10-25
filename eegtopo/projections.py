# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

from numpy import arcsin, sin, cos, pi


def project_radial_to2d(point_3d):
    point_2d = point_3d.copy()
    point_2d.z = 0
    beta = point_2d.norm()
    if beta == 0:
        alpha = 0
    else:
        alpha = arcsin(beta) / beta

    if point_3d.z < 0:
        alpha = pi / beta - alpha

    point_2d *= alpha

    return point_2d


def project_radial_to3d(point_2d):
    alpha = point_2d.norm()
    if alpha == 0:
        beta = 1
    else:
        beta = sin(alpha) / alpha
    point_3d = point_2d * beta
    point_3d.z = cos(alpha)
    return point_3d
