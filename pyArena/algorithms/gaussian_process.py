import numpy as np
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib import cm

class GaussianProcess(ABC):

    def __init__(self, **kwargs):
        pass

class GPRegression(GaussianProcess):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if 'kernel' not in kwargs:
            raise KeyError("Must specify a kernel function!")
        else:
            self.kernel = kwargs['kernel']

        if 'measurementNoiseCov' not in kwargs:
            self.noiseCov = 0.0
        else:
            self.noiseCov = kwargs['measurementNoiseCov']

    """
    trainModel(inpTrain, outTrain)
    The function is used to train the GP model given input (inpTrain) and  output (outTrain) training data.
    Training the GP model implies computation of prior mean and convariance distribution over the regressor function.

    Necessary function arguments:
    inpTrain - (numDim, numTrain) numpy array
    outTrain - (numTrain, 1) numpy array. Actually a 1D numpy array
    """
    def trainGP(self, inpTrain, outTrain):

        self.inpTrain = inpTrain.copy() # Please verify what happens without this copy()

        self.outTrain = outTrain.copy()

        self.numTrain = len(outTrain)

        # Compute the prior GP covariance matrix

        Ktrtr = np.zeros([self.numTrain,self.numTrain])

        for row in range(0,self.numTrain):
            for column in range(row,self.numTrain):
                Ktrtr[column, row] = Ktrtr[row, column] = self.kernel(inpTrain[:,row],inpTrain[:,column])

        self.priorCovariance = Ktrtr

    """
    trainModelIterative(inpTrain, outTrain)
    TODO Implement

    Necessary function arguments:
    inpTrain - (numDim, 1) numpy array
    outTrain - 1 float
     """
    def trainGPIterative(self, inpTrain, outTrain):
        pass

    """
    Evaluate GP to obtain mean and value at a testing point
    inpTest is (numDim,1) numpy array
    """
    def predict_value(self, inpTest):

        Ktrte = np.zeros([self.numTrain,1])

        for index in range(0,self.numTrain):
            Ktrte[index] = self.kernel(self.inpTrain[:,index], inpTest)

        Ktrtr = self.priorCovariance

        Ktete = self.kernel(inpTest, inpTest)

        mu_hat = Ktrte.T @ np.linalg.inv(Ktrtr + self.noiseCov*np.eye(self.numTrain))  @ self.outTrain

        var_hat = Ktete - Ktrte.T @ np.linalg.inv(Ktrtr + self.noiseCov*np.eye(self.numTrain)) @ Ktrte

        return mu_hat, var_hat

    """
    Evaluate GP on a grid
    xmin is (2,1) numpy array
    xmax is (2,1) numpy array
    gridSize is (2,1) numpy array or scalar    
    """
    def predict_grid_value(self, xmin, xmax, gridSize=10):
        if np.array(gridSize).size == 1:
            gridSize = np.ones(2)*gridSize
            
        x0 = np.linspace(xmin[0], xmax[0], gridSize[0])
        x1 = np.linspace(xmin[1], xmax[1], gridSize[1])
        X0, X1 = np.meshgrid(x0,x1)
        xTest = np.stack((X0.reshape(X0.shape[0]*X0.shape[1]), \
                            X1.reshape(X1.shape[0]*X1.shape[1]) ))
           
        numPts = int(gridSize[0]*gridSize[1])
        ypred = np.zeros(numPts)
        var_pred = np.zeros(numPts)
        for index in range(0,numPts):
            ypred[index], var_pred[index] = self.predict_value(xTest[:,index])

        return X0, X1, ypred, var_pred          


    def plot_grid(self, xmin, xmax, gridSize=10):
        if np.array(gridSize).size == 1:
            gridSize = np.ones(2)*gridSize

        X0, X1, ypred, var_pred =  self.predict_grid_value(xmin, xmax, gridSize)
        
        h = plt.figure("Gaussian Process")
        plt.clf()
        ax1 = plt.subplot(1,2,1)  
        p = ax1.pcolor(X0, X1, ypred.reshape([X0.shape[0],X0.shape[1]]), cmap=cm.jet, vmin=-20, vmax=20)
        cb = h.colorbar(p)
        plt.axis('equal')

        ax2 = plt.subplot(1,2,2)    
        p = ax2.pcolor(X0, X1, var_pred.reshape([X0.shape[0],X0.shape[1]]), cmap=cm.jet, vmin=0, vmax=1)
        cb = h.colorbar(p)
        plt.axis('equal')
        plt.pause(.1)