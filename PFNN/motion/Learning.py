import numpy as np
import scipy.linalg as linalg
import scipy.spatial as spatial

kernels = {
    'multiquadric': lambda x: np.sqrt(x**2 + 1),
    'inverse':      lambda x: 1.0 / np.sqrt(x**2 + 1),
    'gaussian':     lambda x: np.exp(-x**2),
    'linear':       lambda x: x,
    'quadric':      lambda x: x**2,
    'cubic':        lambda x: x**3,
    'quartic':      lambda x: x**4,
    'quintic':      lambda x: x**5,
    'thin_plate':   lambda x: x**2 * np.log(x + 1e-10),
    'logistic':     lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -5, 5))),
    'smoothstep':   lambda x: ((np.clip(1.0 - x, 0.0, 1.0))**2.0) * (3 - 2*(np.clip(1.0 - x, 0.0, 1.0)))
}

class Solve:
    
    def __init__(self, l=1e-5):
        self.l = l
        
    def fit(self, X, Y):
        self.M = linalg.lu_solve(linalg.lu_factor(X.T + np.eye(len(X)) * self.l), Y).T
        
    def __call__(self, Xp):
        return self.M.dot(Xp.T).T
    
    
class RBF:
    
    def __init__(self, L=None, epsilon=None, function='multiquadric', smooth=1e-10):
        self.L = Solve(l=-smooth) if L is None else L
        self.f = kernels.get(function, function)
        self.E = epsilon
        
    def fit(self, X, Y):
        self.X = X
        D = spatial.distance.cdist(self.X, self.X)
        self.E = np.ones(len(D)) / D.mean() if self.E is None else self.E
        self.L.fit(self.f(self.E * D), Y)
        
    def __call__(self, Xp):
        D = spatial.distance.cdist(Xp, self.X)
        return self.L(self.f(self.E * D))
    

    
