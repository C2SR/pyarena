import sys

sys.path.append('..')

import pyArena.core as pyacore
import pyArena.algorithms.gaussian_process as GP
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Static Scalar Field
func = lambda x: (x[0] ** 2) / 1000 + (x[1] ** 2) / 4000 \
                 - (((x[0] - 10) ** 2) / 4000 + ((x[1] + 20) ** 2) / 2000) \
                 + ((x[0] * x[1]) / 2000 + (x[0] * x[1]) / 600)

kwargsSensor = {'ScalarField': func, 'covariance': 0.1}
sensorEnv = pyacore.sensors.StaticScalarField2D(**kwargsSensor)

# Create SparseGP object
tau_s = 40
kernel = lambda x1, x2: np.exp(-(0.5 / np.float_power(tau_s, 2)) \
                               * np.float_power(np.linalg.norm(x1 - x2, axis=-1), 2))

kwargsGP = {'measurementNoiseCov': 0.1, 'kernel': kernel}
gp = GP.FITCSparseGP(**kwargsGP)

# Generate training data
xmax = ymax = 150
xmin = ymin = -150

numTrain = 10
inpTrain, outTrain = sensorEnv.getFullPlotData(xmin=[xmin, ymin], \
                                               xmax=[xmax, ymax], numGrid=numTrain)

# TODO: This change in the sensor file
inpTrain = inpTrain.T
outTrain = outTrain.reshape(numTrain ** 2, 1)

# Generate inducing inputs
def getRandomWayPoint(xmin=[-150, -150], xmax=[150, 150]):
    r = np.random.rand(2)
    return xmin + 2 * r * xmax

def GetInducingPoints(numPoints, xmin=[-150, -150], xmax=[150, 150], mode='static'):
    if mode == 'static':
        X, Y = np.mgrid[xmin[0]:xmax[0]:numPoints * 1j, xmin[1]:xmax[1]:numPoints * 1j]
        way_points = np.stack((X.reshape(X.shape[0] * X.shape[1]), \
                               Y.reshape(Y.shape[0] * Y.shape[1])), axis=-1)
    elif mode == 'random':
        way_points = np.zeros([numPoints, 2])
        for index in range(0, numPoints):
            way_points[index] = getRandomWayPoint(xmin, xmax)
    return way_points

# Play with inducing points by changing numPoints
numPoints = 5
inpInduce = GetInducingPoints(numPoints, [xmin, ymin], [xmax, ymax], mode='static')
gp.train_offline(inpTrain, outTrain, inpInduce)

# Predict and plot

# Plot ground truth
numTruth = 100
xTruth, yTruth = sensorEnv.getFullPlotData(xmin=[xmin, ymin], xmax=[xmax, ymax], numGrid=numTruth)
X0 = xTruth[0].reshape(numTruth, numTruth)
X1 = xTruth[1].reshape(numTruth, numTruth)
Y = yTruth.reshape(numTruth, numTruth)

h1 = plt.figure(1)
ax1 = plt.subplot(2, 2, 1)
p = ax1.pcolor(X0, X1, Y, cmap=cm.jet, vmin=-20, vmax=20)
cb = h1.colorbar(p)

# Plot prediction

numPred = 100
X, Y = np.mgrid[xmin:xmax:numPred * 1j, ymin:ymax:numPred * 1j]
inpPred = np.stack((X.reshape(X.shape[0] * X.shape[1]), \
                    Y.reshape(Y.shape[0] * Y.shape[1])), axis=-1)
lenPred = len(inpPred)

outPred = np.zeros([lenPred, 1])
varPred = np.zeros([lenPred, 1])
for index in range(0, lenPred):
    outPred[index], varPred[index] = gp.predict_value(inpPred[index])

ax2 = plt.subplot(2, 2, 2)
p = ax2.pcolor(X, Y, outPred.reshape([numPred, numPred]), cmap=cm.jet, vmin=-20, vmax=20)
cb = h1.colorbar(p)

ax3 = plt.subplot(2, 2, 3)
p = ax3.pcolor(X, Y, varPred.reshape([numPred, numPred]), cmap=cm.jet, vmin=-20, vmax=20)
cb = h1.colorbar(p)

ax4 = plt.subplot(2, 2, 4)
for index in range(0, len(inpInduce)):
    ax4.plot(inpInduce[index, 0], inpInduce[index, 1], '*b')

plt.show()