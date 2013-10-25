# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

import numpy as np
from scipy.interpolate import interp1d
#noinspection PyPep8Naming
import matplotlib.pyplot as plot
#noinspection PyPep8Naming
import matplotlib.path as path
#noinspection PyPep8Naming
import matplotlib.patches as patches
from .projections import project_radial_to3d, project_radial_to2d
from .geometry.euclidean import Vector

class Topoplot:
    ''' Creates 2D scalp maps. '''
    def __init__(self, m=4, num_lterms=20):
        self.interprange = np.pi * 3/4        
        head_radius = self.interprange
        nose_angle = 15
        nose_length = 0.12
        
        verts = np.array([
            (1, 0),
            (1, 0.5535714285714286), (0.5535714285714286, 1), (0, 1),
            (-0.5535714285714286, 1), (-1, 0.5535714285714286), (-1, 0),
            (-1, -0.5535714285714286), (-0.5535714285714286, -1), (0, -1),
            (0.5535714285714286, -1), (1, -0.5535714285714286), (1, 0),
            ]) * self.interprange        
        codes = [path.Path.MOVETO,
                path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4,
                path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4,
                path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4,
                path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4,
                ]
        self.path_head = path.Path(verts, codes)
           
        x = head_radius * np.cos((90.0-nose_angle/2)*np.pi/180.0);
        y = head_radius * np.sin((90.0-nose_angle/2)*np.pi/180.0);        
        verts = np.array([(x,y), (0,head_radius * (1+nose_length)), (-x,y)])
        codes = [path.Path.MOVETO, path.Path.LINETO, path.Path.LINETO]
        self.path_nose = path.Path(verts, codes)
        
        self.legendre_factors = self.calc_legendre_factors( m, num_lterms )

        self.locations = None
        self.g = None
        self.z = None
        self.c = None
        self.image = None
        
    @staticmethod
    def calc_legendre_factors(m, num_lterms):
        return [(2*n+1) / (n**m * (n+1)**m * 4*np.pi) for n in range(1, num_lterms+1)]
        
    def calc_g(self, x):
        return np.polynomial.legendre.legval( x, self.legendre_factors )
        
    def set_locations(self, locations):        
        n = len(locations)
        
        g = np.zeros( (1+n, 1+n) )
        g[:,0] = np.ones(1+n)
        g[-1,:] = np.ones(1+n)
        g[-1,0] = 0
        for i in range(n):
            for j in range(n):
                g[i,j+1] = self.calc_g( np.dot(locations[i], locations[j])  )
        
        self.locations = locations
        self.g = g
        
    def set_values(self, z):
        self.z = z
        self.c = np.linalg.solve(self.g, np.concatenate((z,[0])))
        
    def get_map(self):
        return self.image
        
    def set_map(self, img):
        self.image = img
        
    def create_map(self, pixels=32):
        self.image = np.zeros((pixels,pixels)) * np.nan        
        
        gridlocs = np.linspace(-self.interprange, self.interprange, pixels)
        dx2 = (gridlocs[2] - gridlocs[0]) # distance of two pixels
        
        for j in range(pixels):
            x = gridlocs[j]
            for i in range(pixels):
                y = -gridlocs[i]
                
                if x**2 + y**2 <= (self.interprange+dx2)**2: # skip some unnecessary calculations
                    e = project_radial_to3d( Vector(x,y,0) )        
                    self.image[i,j] = self.c[0] + self.c[1:].dot( self.calc_g( np.dot( self.locations, [k for k in e] ) ) )
                    
    def plot_map(self, axes=None, crange=None):
        if axes is None: axes = plot.gca()
        cliptransform = axes.transData
        if crange is None:
            vru = np.nanmax(np.abs(self.image))
            vrl = -vru;
        else:
            vrl, vru = crange            
        return axes.imshow(self.image, vmin=vrl, vmax=vru, clip_path=(self.path_head,cliptransform), extent=(-self.interprange, self.interprange, -self.interprange, self.interprange) )
        
    def plot_locations(self, axes=None):
        if axes is None: axes = plot.gca()
        for p in self.locations:
            p2 = project_radial_to2d( Vector.fromiterable(p) )
            axes.plot(p2.x, p2.y, 'k.')
            
    def plot_head(self, axes=None):
        if axes is None: axes = plot.gca()
        axes.add_patch(patches.PathPatch(self.path_head, facecolor='none', lw=2))
        axes.add_patch(patches.PathPatch(self.path_nose, facecolor='none', lw=2))
        
    def plot_circles(self, radius, axes=None):
        if axes is None: axes = plot.gca()
        col = interp1d([-1, 0, 1], [[0, 1, 1], [0, 1, 0], [1, 1, 0]])
        for i in range(len(self.locations)):
            p3 = self.locations[i]
            p2 = project_radial_to2d( Vector.fromiterable(p3) )
            circ = plot.Circle((p2.x, p2.y), radius=radius, color=col(self.z[i]))
            axes.add_patch(circ)
        
def topoplot( values, locations, axes=None ):
    topo = Topoplot( )
    topo.set_locations(locations)
    topo.set_values(values)
    topo.create_map()
    h = topo.plot_map(axes)
    topo.plot_locations(axes)
    topo.plot_head(axes)    
    #plot.colorbar(h)
    return topo
