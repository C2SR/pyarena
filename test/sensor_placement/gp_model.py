import sys
sys.path.append('../..')

import pyArena.core as pyacore
import pyArena.algorithms.gaussian_process as GP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class gpModel:

    def __init__(self, area):

        self.area = area
        self.xmin = - area.x_dim
        self.ymin = - area.y_dim
        self.xmax = area.x_dim
        self.ymax = area.y_dim

        self.numTrain = area.numTrain
        self.numTest = 30
        self.numTruth = area.numTruth

        """ Setup scalar field """
        # Create a static Scalar Field as ground truth
        func = lambda x: (x[0] ** 2) / 1000 + (x[1] ** 2) / 4000 - (
                    ((x[0] - 10) ** 2) / 4000 + ((x[1] + 20) ** 2) / 2000) + (
                                     (x[0] * x[1]) / 2000 + (x[0] * x[1]) / 600)

        kwargsSensor = {'ScalarField': func, 'covariance': 0.1}
        self.sensorEnv = pyacore.sensors.StaticScalarField2D(**kwargsSensor)

        # Specify GP Model
        phi = 0.2
        tau_s = 10
        kernel = lambda x1, x2: np.exp(
            -(0.5 / np.float_power(tau_s, 2)) * np.float_power(np.linalg.norm(x1 - x2, axis=-1), 2))
        kwargsGP = {'kernel': kernel, 'measurementNoiseCov': phi}

        # Get training data
        xTrain, yTrain = self.sensorEnv.getFullPlotData(xmin=[self.xmin, self.ymin], xmax=[self.xmax, self.ymax], numGrid=self.numTrain)
        # TODO: This change in the sensor file
        xTrain = xTrain.T
        yTrain = yTrain.reshape(self.numTrain ** 2, 1)

        self.gp = GP.GPRegression(**kwargsGP)
        self.gp.train_offline(xTrain, yTrain)

    def plot(self):
        # Plot training and test data
        xTruth, yTruth = self.sensorEnv.getFullPlotData(xmin=[self.xmin, self.ymin], xmax=[self.xmax, self.ymax], numGrid=self.numTruth)
        X0 = xTruth[0].reshape(self.numTruth, self.numTruth)
        X1 = xTruth[1].reshape(self.numTruth, self.numTruth)
        Y = yTruth.reshape(self.numTruth, self.numTruth)

        h1 = plt.figure(1)
        ax1 = plt.subplot(2, 2, 1)
        p = ax1.pcolor(X0, X1, Y, cmap=cm.jet, vmin=-20, vmax=20)
        cb = h1.colorbar(p)

        X, Y = np.mgrid[self.xmin:self.xmax:self.numTest * 1j, self.ymin:self.ymax:self.numTest * 1j]
        inpPred = np.stack((X.reshape(X.shape[0] * X.shape[1]), \
                            Y.reshape(Y.shape[0] * Y.shape[1])), axis=-1)
        lenPred = len(inpPred)

        outPred = np.zeros([lenPred, 1])
        varPred = np.zeros([lenPred, 1])
        for index in range(0, lenPred):
            outPred[index], varPred[index] = self.gp.predict_value(np.array([inpPred[index]]))

        ax2 = plt.subplot(2, 2, 2)
        p = ax2.pcolor(X, Y, outPred.reshape([self.numTest, self.numTest]), cmap=cm.jet, vmin=-20, vmax=20)
        cb = h1.colorbar(p)

        ax3 = plt.subplot(2, 2, 3)
        p = ax3.pcolor(X, Y, varPred.reshape([self.numTest, self.numTest]), cmap=cm.jet)
        cb = h1.colorbar(p)

        plt.show()
