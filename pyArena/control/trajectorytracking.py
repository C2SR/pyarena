# TODOC
from ..core import controller
from ..algorithms import gaussian_process as GP

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

## Create a path following controller class
class TrajectoryTracking(controller.StaticController):

    def __init__(self, **kwargs):

        super().__init__()

        self.funpd = kwargs['pd']

        self.funpdDot = kwargs['pdDot']

        self.K = kwargs['gain']

        self.eps = kwargs['eps']

        self.invDelta = np.linalg.pinv(np.array([[1.0, -self.eps[1]], [0.0, self.eps[0]]]))

    def computeInput(self, t, x, *args):

        p = x[0:2]

        theta = x[2]

        pd = self.funpd(t)

        pdDot = self.funpdDot(t)

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        e =  (R.T)@(p - pd) + self.eps

        u_ff = (R.T)@pdDot

        u = self.invDelta@(-self.K@e + u_ff)

        return u


class TrajectoryTrackingWithGP(TrajectoryTracking):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        kwargsGP = kwargs['GaussianProcess']

        self.mGP = GP.GPRegression(**kwargsGP)

        self.counter = 0

    def computeInput(self, t, x, *args):
        if (self.counter%5 != 0):
            self.mGP.trainGPIterative( x[0:2], args[0], False)                
        else:
            self.mGP.trainGPIterative( x[0:2], args[0], True)            
            # Plotting sensorEnv ground truth
            num = 20
            
            xmin = [-5,-5]
            xmax = [5,5]
            x0 = np.linspace(xmin[0], xmax[0], num)
            x1 = np.linspace(xmin[1], xmax[1], num)
            X0, X1 = np.meshgrid(x0,x1)
            xTest = np.stack((X0.reshape(X0.shape[0]*X0.shape[1]), \
                        X1.reshape(X1.shape[0]*X1.shape[1]) ))
           

            ypred = np.zeros(num**2)
            var_pred = np.zeros(num**2)
            for index in range(0,num**2):
                ypred[index], var_pred[index] = self.mGP.predict_value(xTest[:,index])
            
            h1 = plt.figure(1)
            ax3 = plt.subplot(2,2,3)  
            p = ax3.pcolor(X0, X1, ypred.reshape([num,num]), cmap=cm.jet, vmin=-20, vmax=20)
            ax4 = plt.subplot(2,2,4)    
            p = ax4.pcolor(X0, X1, var_pred.reshape([num,num]), cmap=cm.jet)
            plt.pause(.1)

        u = super(TrajectoryTrackingWithGP, self).computeInput(t,x,*args)

        self.counter += 1

        return u









   