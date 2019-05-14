import sys
sys.path.append('..')

import pyArena.core as pyacore
from pyArena.algorithms import gaussian_process as GP
import numpy as np
import matplotlib.pyplot as plt

## Build a simulated 2D Scalar Field
func = lambda x: 10*np.sin(0.2*x[0]) + 5*np.cos(0.3*x[1])

kwargsSensor = {'ScalarField': func, 'covariance': 0.1}
sensorEnv = pyacore.sensors.StaticScalarField2D(**kwargsSensor)

## Build GP Model for regression for Scalar Field

# Specify GP Model
phi = 0.2
tau_s = 5
kernel = lambda x1, x2: np.exp(-(0.5/np.float_power(tau_s,2))*np.float_power(np.linalg.norm(x1-x2), 2))

kwargsGP = {'kernel': kernel, 'measurementNoiseCov': phi}

# Get training data
numTrain = 10
xTrain, yTrain = sensorEnv.getFullPlotData(xmin = [-5, -5], xmax = [5, 5], numGrid = numTrain)
print(xTrain)
mGP = GP.GPRegression(**kwargsGP)

mGP.trainGP(xTrain, yTrain)
print(xTrain.shape)
"""
# Get testing data
numTest = 30
xTest, yTest = sensorEnv.getFullPlotData(xmin = [-5, -5], xmax = [5, 5], numGrid = numTest)

# Prediction using GP
n = numTest**2
ypred = np.zeros(n)
ysense = np.zeros(n)
var_pred = np.zeros(n)
for index in range(0,n):
    ypred[index], var_pred[index] = mGP.predict_value(xTest[:,index])
    ysense[index] = sensorEnv.sense(None, xTest[:,index])

# Get ground truth for plotting
numTruth = 100
xTruth, yTruth = sensorEnv.getFullPlotData(xmin = [-5, -5], xmax = [5, 5], numGrid = numTruth)

h1 = plt.figure(1)
plt.imshow(yTruth.reshape(numTruth,numTruth), interpolation='bilinear')
plt.colorbar()

h2 = plt.figure(2)
plt.imshow(ypred.reshape(numTest,numTest), interpolation='bilinear')
plt.colorbar()

h3 = plt.figure(3)
plt.imshow(var_pred.reshape(numTest,numTest), interpolation='bilinear')
plt.colorbar()

h4 = plt.figure(4)
err = (ypred - ysense)
plt.plot(err)
plt.show()
"""