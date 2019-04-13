# TODOC
import numpy as np
from ..core import controller

## Create a path following controller class
class TrajectoryTracking(controller.StaticController):

    def __init__(self, **kwargs):

        super().__init__()

        self.funpd = kwargs['pd']

        self.funpdDot = kwargs['pdDot']

        self.K = kwargs['gain']

        self.eps = kwargs['eps']

        self.invDelta = np.linalg.pinv(np.array([[1.0, -self.eps[1]], [0.0, self.eps[0]]]))

    def computeInput(self, t, x):

        p = x[0:2]

        theta = x[2]

        pd = self.funpd(t)

        pdDot = self.funpdDot(t)

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        e =  (R.T)@(p - pd) + self.eps

        u_ff = (R.T)@pdDot

        u = self.invDelta@(-self.K@e + u_ff)

        return u
