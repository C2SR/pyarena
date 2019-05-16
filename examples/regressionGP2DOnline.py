#!/usr/bin/env python
# coding: utf-8

# # Trajectory tracking control for unicyle robot
# TODO: Document here

# Do this once
import sys
sys.path.append('..')

## Necessary imports
import pyArena.core as pyacore
import pyArena.control.trajectorytracking as pyacontrol
import pyArena.vehicles.underactuatedvehicle as pyavehicle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

## Simulation parameters
nx = 3
nu = 2
Tsim = 15
dt = .1
x_init = np.array([0.0, 0.0, np.pi/2])

## Build a simulated 2D Scalar Field
func = lambda x: 10*np.cos(np.pi/5*x[0]) + 10*np.sin(np.pi/5*x[1])  
kwargsSensor = {'ScalarField': func, 'covariance': 0.1}
sensorEnv = pyacore.sensors.StaticScalarField2D(**kwargsSensor)

## GP parameters
phi = 0.2
tau_s = 5
kernel = lambda x1, x2: np.exp(-(0.5/np.float_power(tau_s,2))*np.float_power(np.linalg.norm(x1-x2), 2))
kwargsGP = {'kernel': kernel, 'measurementNoiseCov': phi}

## Specify desired trajectory
radius = 4
a = 2*np.pi/10
pd = lambda t: radius*np.array([np.cos(a*t), np.sin(a*t)])
pdDot = lambda t: radius*np.array([-a*np.sin(a*t), a*np.cos(a*t)])

# Trajectory tracking control law parameters
K = np.array([[1, 0.0],[0.0, 0.1]])
eps = np.array([1, 0])

## Specify the controller
kwargsController = {'pd': pd, 'pdDot': pdDot, 'gain': K, 'eps': eps, 'GaussianProcess': kwargsGP}
ttController = pyacontrol.TrajectoryTrackingWithGP(**kwargsController)

## Assembling system
kwargsSystem = {'initialCondition': x_init, 'controller': ttController, 'sensor': sensorEnv}
system = pyavehicle.UnicycleKinematics(**kwargsSystem)

## Create pyArena simulation object
kwargsSimulation = {'system': system, 'simTime': Tsim, 'dt': dt}
pyA = pyacore.simulator.pyArena(**kwargsSimulation)

# Plotting sensorEnv ground truth
numTruth = 100
xTruth, yTruth = sensorEnv.getFullPlotData(xmin = [-5, -5], xmax = [5, 5], numGrid = numTruth)
X0 = xTruth[0].reshape(numTruth,numTruth)
X1 = xTruth[1].reshape(numTruth,numTruth)
Y = yTruth.reshape(numTruth,numTruth)

h1 = plt.figure(1)
ax1 = plt.subplot(1,2,1)
p = ax1.pcolor(X0, X1, Y, cmap=cm.jet, vmin=-20, vmax=20)
cb = h1.colorbar(p)
plt.axis('equal')
plt.show(0)

# Run simulation
dataLog = pyA.run()

## Plot results
h1 = plt.figure(1)
ax2 = plt.subplot(1,2,2)
pdVec = pd(dataLog.time).T
plt.plot(pdVec[:,0], pdVec[:,1], 'k--')
plt.plot(dataLog.stateTrajectory[:,0], dataLog.stateTrajectory[:,1], 'r')
plt.axis('equal')
plt.show()
