# TODOC
import numpy as np

## Create a path following controller class
class PathFollowing:
    def __init__(self, **kwargs):
        self.funpd = kwargs['pd']
        self.funpdD = kwargs['pdD']
        self.K = kwargs['gain']
        self.eps = kwargs['eps']
        self.vd = kwargs['vd']
        self.invDelta = np.linalg.pinv(np.array([[1.0, self.eps[1]], [0.0, self.eps[0]]]))
        
    def computeInput(self, t, x):
        p = x[0:1]
        theta = x[2]
        gamma = x[3]
        
        pd = self.funpd(gamma)
        pdDot = self.funpdD(gamma)*self.vd
        
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        
        e =  np.matmul(R.T,(p - pd)) + self.eps
    
        u_ff = np.matmul(R.T, pdDot)
    
        u = np.matmul(-self.K,e) + u_ff
    
        g_err = 0 # TODO: make the modifications
    
        gamma_dot = self.vd + g_err
        
        return np.append(u, gamma_dot)