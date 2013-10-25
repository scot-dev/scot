# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

'''Spherical geometry support module'''

import math
from .euclidean import Vector

################################################################################

eps = 1e-15

################################################################################

class Point:
    '''Point on the surface of a sphere'''
    
    def __init__(self, x=None, y=None, z=None):
        if x is None and y is None and z is None:
            self._pos3d = Vector(0, 0, 1)
        elif x is not None and y is not None and z is None:
            self._pos3d = Vector(x, y, math.sqrt(1-x**2-y**2))
        elif x is not None and y is not None and z is not None:
            self._pos3d = Vector(x, y, z).normalized()
        else:
            raise RuntimeError('invalid parameters')
            
    @classmethod
    def fromvector(cls, v):
        '''Initialize from euclidean vector'''
        w = v.normalized()
        return cls(w.x, w.y, w.z)
    
    @property
    def vector(self):
        '''position in 3d space'''
        return self._pos3d
        
    @vector.setter
    def vector(self, v):
        self._pos3d.x = v.x
        self._pos3d.y = v.y
        self._pos3d.z = v.z
            
    def __repr__(self):
        return ''.join((__class__.__name__, '(',str(self._pos3d.x),', ',str(self._pos3d.y),', ',str(self._pos3d.z),')'))
            
    def distance(self, other):
        """Distance to another point on the sphere"""
        return math.acos( self._pos3d.dot( other.vector ) )
            
    def distances(self, points):
        """Distance to other points on the sphere"""
        return [math.acos(self._pos3d.dot(p.vector)) for p in points]
        
################################################################################
        
class Line:
    '''Line on the spherical surface (also known as grand circle)'''
    
    def __init__(self, A, B ):
        self.A = Point.fromvector(A.vector)
        self.B = Point.fromvector(B.vector)
        
    def getPoint(self, l):
        d = self.A.distance(self.B)
        N = self.A.vector.cross(self.B.vector)
        P = Point.fromvector(self.A.vector)
        P.vector.rotate(l*d, N)
        return P
        
    def distance(self, P):
        N = Point.fromvector(self.A.vector.cross(self.B.vector))
        return abs( math.pi/2 - N.distance(P) )
        
################################################################################
        
class Circle:
    '''Arbitrary circle on the spherical surface'''
    
    def __init__(self, A, B, C=None ):
        if C is None:
            self.C = Point.fromvector(A.vector)     # Center
            self.X = Point.fromvector(B.vector)     # A point on the circle
        else:
            self.C = Point.fromvector((B.vector-A.vector).cross(C.vector-B.vector).normalized())    # Center
            self.X = Point.fromvector(B.vector)     # A point on the circle
        
    def getPoint(self, l):
        return Point.fromvector(self.X.vector.rotated(l, self.C.vector))
        
    def getRadius(self):
        return self.C.distance(self.X)
        
    def angle(self, P):

        c = self.C.vector * self.X.vector.dot(self.C.vector) # center in circle plane
        
        a = (self.X.vector - c).normalized( );
        b = (P.vector - c).normalized( );
        return math.acos(a.dot(b))
        
    def distance(self, P):
        return abs( self.C.distance(P) - self.C.distance(self.X) )
        
    
################################################################################
        
class construct:
    '''Collection of methods for geometric construction on a sphere'''
    
    @staticmethod
    def midpoint( A, B ):
        '''Point exactly between A and B'''
        return Point.fromvector( (A.vector + B.vector) / 2 )
        
    @staticmethod
    def line_intersect_line( K, L ):
        C1 = K.A.vector.cross(K.B.vector)
        C2 = L.A.vector.cross(L.B.vector)
        P = C1.cross(C2)
        return (Point.fromvector(P), Point.fromvector(P*-1))
        
    @staticmethod
    def line_intersect_circle(L, C):
        cross_line = L.A.vector.cross(L.B.vector)
        cross_lc = cross_line.cross(C.C.vector)
        dot_circle = C.C.vector.dot( C.X.vector )
        if abs(cross_lc.z) > eps:
            a = cross_lc.dot(cross_lc)
            b = 2*dot_circle*cross_line.cross(cross_lc).z
            c = dot_circle*dot_circle*(cross_line.x**2+cross_line.y**2) - cross_lc.z**2            
            s = math.sqrt(b**2-4*a*c)
            z1 = (s - b)/(2*a)
            x1 = (cross_lc.x*z1 - cross_line.y*dot_circle)/cross_lc.z
            y1 = (cross_lc.y*z1 + cross_line.x*dot_circle)/cross_lc.z
            z2 = -(s + b)/(2*a)
            x2 = (cross_lc.x*z2 - cross_line.y*dot_circle)/cross_lc.z
            y2 = (cross_lc.y*z2 + cross_line.x*dot_circle)/cross_lc.z            
            return Point(x1, y1, z1), Point(x2, y2, z2)
        else:
            return None
        
    @staticmethod
    def circle_intersect_circle(A, B):
        AC = A.C.vector
        BC = B.C.vector
        cross = AC.cross(BC)
        dot_a = AC.dot(A.X.vector)
        dot_b = BC.dot(B.X.vector)
        if abs(cross.z) > eps:
            a = cross.dot(cross)
            b = 2*(dot_b*AC.cross(cross).z - dot_a*BC.cross(cross).z)
            c = dot_b**2*(AC.x**2 + AC.y**2) - 2*dot_a*dot_b*(AC.x*BC.x + AC.y*BC.y) + dot_a**2*(BC.x**2 + BC.y**2) - cross.z**2
            s = math.sqrt(b**2-4*a*c)
            z1 = (s - b)/(2*a)
            x1 = (BC.y*dot_a - AC.y*dot_b + cross.x*z1)/cross.z
            y1 = (AC.x*dot_b - BC.x*dot_a + cross.y*z1)/cross.z
            z2 = -(s + b)/(2*a)
            x2 = (BC.y*dot_a - AC.y*dot_b + cross.x*z2)/cross.z
            y2 = (AC.x*dot_b - BC.x*dot_a + cross.y*z2)/cross.z      
            return Point(x1, y1, z1), Point(x2, y2, z2)
        else:
            return None
        
################################################################################
