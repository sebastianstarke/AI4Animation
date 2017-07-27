import numpy as np

from Quaternions import Quaternions

class Pivots:    
    """
    Pivots is an ndarray of angular rotations

    This wrapper provides some functions for
    working with pivots.

    These are particularly useful as a number 
    of atomic operations (such as adding or 
    subtracting) cannot be achieved using
    the standard arithmatic and need to be
    defined differently to work correctly
    """
    
    def __init__(self, ps): self.ps = np.array(ps)
    def __str__(self): return "Pivots("+ str(self.ps) + ")"
    def __repr__(self): return "Pivots("+ repr(self.ps) + ")"
    
    def __add__(self, other): return Pivots(np.arctan2(np.sin(self.ps + other.ps), np.cos(self.ps + other.ps)))
    def __sub__(self, other): return Pivots(np.arctan2(np.sin(self.ps - other.ps), np.cos(self.ps - other.ps)))
    def __mul__(self, other): return Pivots(self.ps  * other.ps)
    def __div__(self, other): return Pivots(self.ps  / other.ps)
    def __mod__(self, other): return Pivots(self.ps  % other.ps)
    def __pow__(self, other): return Pivots(self.ps ** other.ps)
    
    def __lt__(self, other): return self.ps <  other.ps
    def __le__(self, other): return self.ps <= other.ps
    def __eq__(self, other): return self.ps == other.ps
    def __ne__(self, other): return self.ps != other.ps
    def __ge__(self, other): return self.ps >= other.ps
    def __gt__(self, other): return self.ps >  other.ps
    
    def __abs__(self): return Pivots(abs(self.ps))
    def __neg__(self): return Pivots(-self.ps)
    
    def __iter__(self): return iter(self.ps)
    def __len__(self): return len(self.ps)
    
    def __getitem__(self, k):    return Pivots(self.ps[k]) 
    def __setitem__(self, k, v): self.ps[k] = v.ps
    
    def _ellipsis(self): return tuple(map(lambda x: slice(None), self.shape))
    
    def quaternions(self, plane='xz'):
        fa = self._ellipsis()
        axises = np.ones(self.ps.shape + (3,))
        axises[fa + ("xyz".index(plane[0]),)] = 0.0
        axises[fa + ("xyz".index(plane[1]),)] = 0.0
        return Quaternions.from_angle_axis(self.ps, axises)
    
    def directions(self, plane='xz'):
        dirs = np.zeros((len(self.ps), 3))
        dirs["xyz".index(plane[0])] = np.sin(self.ps)
        dirs["xyz".index(plane[1])] = np.cos(self.ps)
        return dirs
    
    def normalized(self):
        xs = np.copy(self.ps)
        while np.any(xs >  np.pi): xs[xs >  np.pi] = xs[xs >  np.pi] - 2 * np.pi
        while np.any(xs < -np.pi): xs[xs < -np.pi] = xs[xs < -np.pi] + 2 * np.pi
        return Pivots(xs)
    
    def interpolate(self, ws):
        dir = np.average(self.directions, weights=ws, axis=0)
        return np.arctan2(dir[2], dir[0])
    
    def copy(self):
        return Pivots(np.copy(self.ps))
    
    @property
    def shape(self):
        return self.ps.shape
    
    @classmethod
    def from_quaternions(cls, qs, forward='z', plane='xz'):
        ds = np.zeros(qs.shape + (3,))
        ds[...,'xyz'.index(forward)] = 1.0
        return Pivots.from_directions(qs * ds, plane=plane)
        
    @classmethod
    def from_directions(cls, ds, plane='xz'):
        ys = ds[...,'xyz'.index(plane[0])]
        xs = ds[...,'xyz'.index(plane[1])]
        return Pivots(np.arctan2(ys, xs))
    
