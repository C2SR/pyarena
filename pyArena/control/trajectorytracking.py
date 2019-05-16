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
        self.inpTrain=[]
        self.outTrain=[]

    def computeInput(self, t, x, *args):
        # Store input training data
        if (self.inpTrain!=[]):
            self.inpTrain=self.inpTrain.T
        self.inpTrain = np.append(self.inpTrain, x[0:2]).reshape(self.counter+1,2).T

        # Store output training data
        self.outTrain =  np.append(self.outTrain, args[0]) 

        # Train GP
        if (self.counter%10 == 0):
            self.mGP.trainGP(self.inpTrain, self.outTrain) 

            # Plot
            num=10
            X0, X1, ypred, var_pred =  self.mGP.predict_grid_value(xmin= [-5,-5], xmax= [5,5], gridSize=num)
            h = plt.figure(2)
            plt.clf()
            ax1 = plt.subplot(1,2,1)  
            p = ax1.pcolor(X0, X1, ypred.reshape([num,num]), cmap=cm.jet, vmin=-20, vmax=20)
            cb = h.colorbar(p)
            plt.axis('equal')

            ax2 = plt.subplot(1,2,2)    
            p = ax2.pcolor(X0, X1, var_pred.reshape([num,num]), cmap=cm.jet, vmin=0, vmax=1)
            cb = h.colorbar(p)
            plt.axis('equal')
            plt.pause(.1)

        u = super(TrajectoryTrackingWithGP, self).computeInput(t,x,*args)

        self.counter += 1

        return u









   