## Do this once
import sys
sys.path.append('..')

## -----------------------------Necessary imports -----------------------------------------##
from pyArena.algorithms import gaussian_process as gp
import pyArena.core as pyacore
import pyArena.control.guidance2D as pyaguidance
import pyArena.vehicles.underactuatedvehicle as pyavehicle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

##--------------------------- Simulation Parameters----------------------------------------##

nx = 3
nu = 2
Tsim = 10000 # REALLY HIGH VALUE
dt = 1

x_init = np.array([5.0, -5.0, np.pi/3])

# Change numWayPoints and you can have more waypoints generated accordingly
# Make sure to increase Tsim for more way points
numWayPoints = 50

# GP parameters
length_scale = 40 # play with this to notice the difference

gpMeasurementCovariance = 0.2 # you may play with this as well

## --------------------------Create the sampling controller--------------------------------##
"""
Random way-point generator between area defined by xmin and xmax
"""
def getRandomWayPoint(xmin = [-150, -150], xmax = [150, 150]):
    r = np.random.rand(2)
    return xmin + 2*r*xmax


class RoboticSampling(pyacore.controller.StaticController):

    def __init__(self, **kwargs):

        super().__init__()

        # Create a waypoint controller

        # self.way_points = np.array([[-100.0, -100.0],\
        #                             [100.0, -100.0],\
        #                             [100.0, -50.0],\
        #                             [-100.0, -50.0],\
        #                             [-100.0, 0.0],\
        #                             [100.0, 0.0],\
        #                             [100.0, 50.0],\
        #                             [-100.0, 50.0],\
        #                             [-100.0, 100.0],\
        #                             [100.0, 100.0]])

        # self.numWP = len(self.way_points)

        self.numWP = numWayPoints

        temp_wp = list()

        for index in range(self.numWP):
            temp_wp = np.append(temp_wp, getRandomWayPoint())

        self.way_points = temp_wp.reshape(self.numWP,2)

        self.wpCounter = 1

        kwargsGuidance = {'wayPoint': self.way_points[self.wpCounter - 1], 'speed': 1}

        self.guidanceLaw = pyaguidance.LOSUnicycle(**kwargsGuidance)

        # Create a GP Regression process

        # GP parameters

        # TODO: Hyper-parameter estimation

        tau_s = length_scale

        kernel = lambda x1, x2: np.exp(-(0.5/np.float_power(tau_s,2))*np.float_power(np.linalg.norm(x1-x2), 2))

        kwargsGP = {'kernel': kernel, 'measurementNoiseCov': gpMeasurementCovariance}

        self.gp = gp.GPRegression(**kwargsGP)

        self.counter = 0

    def computeInput(self, t, x, *measurements):

        # Compute the control inputs using guidance law

        if self.guidanceLaw.isWayPointReached is True:

            # The following IF statement is to ensure no GP updates happen after last way point has been visited.
            # Saves a lot of computation time after last waypoint until Tsim runs out.

            if (self.wpCounter != self.numWP):
                self.gp.trainGPIterative( x[0:2], measurements[0], True)

                # Plotting sensorEnv ground truth
                num = 20

                xmin = [-150,-150]
                xmax = [150,150]
                x0 = np.linspace(xmin[0], xmax[0], num)
                x1 = np.linspace(xmin[1], xmax[1], num)
                X0, X1 = np.meshgrid(x0,x1)
                xTest = np.stack((X0.reshape(X0.shape[0]*X0.shape[1]), \
                                X1.reshape(X1.shape[0]*X1.shape[1]) ))

                ypred = np.zeros(num**2)
                var_pred = np.zeros(num**2)
                for index in range(0,num**2):
                    ypred[index], var_pred[index] = self.gp.predict_value(xTest[:,index])

                h1 = plt.figure(1)
                ax3 = plt.subplot(2,2,3)
                p = ax3.pcolor(X0, X1, ypred.reshape([num,num]), cmap=cm.jet, vmin=-20, vmax=20)
                ax4 = plt.subplot(2,2,4)
                p = ax4.pcolor(X0, X1, var_pred.reshape([num,num]), cmap=cm.jet)
                plt.pause(.1)

            # The following IF statement checks if last wayp point is reached. If not new way point is set

            if (self.wpCounter == self.numWP):
                self.wpCounter = self.wpCounter
            else:
                self.wpCounter = self.wpCounter + 1
                self.guidanceLaw.setWayPoint(self.way_points[self.wpCounter-1])
                print("Way Point number {} set!".format(self.wpCounter))

        control_input = self.guidanceLaw.run(t,x)

        return control_input
## --------------------------END sampling controller--------------------------------##

## --------------------------START Simulation---------------------------------------##

## Build a simulated 2D Scalar Field

#func = lambda x: 100*np.cos(np.pi/5*x[0]) + 100*np.sin(np.pi/5*x[1])

func = lambda x: (x[0]**2)/1000 + (x[1]**2)/4000 - (((x[0] - 10)**2)/4000 + ((x[1] + 20)**2)/2000) + ((x[0]*x[1])/2000 + (x[0]*x[1])/600)

kwargsSensor = {'ScalarField': func, 'covariance': 0.1}

sensorEnv = pyacore.sensors.StaticScalarField2D(**kwargsSensor)

## Define controller

controller = RoboticSampling()

## Create a system with Controller

kwargsSystem = {'initialCondition': x_init, 'controller': controller, 'sensor': sensorEnv}

system = pyavehicle.UnicycleKinematics(**kwargsSystem)

## Create pyArena simulation object and simulate

kwargsSimulation = {'system': system, 'simTime': Tsim, 'dt': dt}

pyA = pyacore.simulator.pyArena(**kwargsSimulation)

# Plot ground truth

numTruth = 100
xTruth, yTruth = sensorEnv.getFullPlotData(xmin = [-150, -150], xmax = [150, 150], numGrid = numTruth)
X0 = xTruth[0].reshape(numTruth,numTruth)
X1 = xTruth[1].reshape(numTruth,numTruth)
Y = yTruth.reshape(numTruth,numTruth)

h1 = plt.figure(1)
ax1 = plt.subplot(2,2,1)
p = ax1.pcolor(X0, X1, Y, cmap=cm.jet, vmin=-20, vmax=20)
cb = h1.colorbar(p)
plt.show(0)

# Run simulation

dataLog = pyA.run()

## Plot results

# Plot ground truth

h2 = plt.figure(2)
ax1 = plt.subplot(2,2,1)
p = ax1.pcolor(X0, X1, Y, cmap=cm.jet, vmin=-20, vmax=20)
cb = h1.colorbar(p)
plt.show(0)

# Plot the learned GP model

num = 100

xmin = [-150,-150]
xmax = [150,150]
x0 = np.linspace(xmin[0], xmax[0], num)
x1 = np.linspace(xmin[1], xmax[1], num)
X0, X1 = np.meshgrid(x0,x1)
xTest = np.stack((X0.reshape(X0.shape[0]*X0.shape[1]), \
X1.reshape(X1.shape[0]*X1.shape[1]) ))

ypred = np.zeros(num**2)
var_pred = np.zeros(num**2)

for index in range(0,num**2):
    ypred[index], var_pred[index] = controller.gp.predict_value(xTest[:,index])

#h2 = plt.figure(2)
ax3 = plt.subplot(2,2,3)
p = ax3.pcolor(X0, X1, ypred.reshape([num,num]), cmap=cm.jet, vmin=-20, vmax=20)
ax4 = plt.subplot(2,2,4)
p = ax4.pcolor(X0, X1, var_pred.reshape([num,num]), cmap=cm.jet)
plt.pause(.1)

# Plot the robot position

ax2 = plt.subplot(2,2,2)
plt.plot(dataLog.stateTrajectory[:,0], dataLog.stateTrajectory[:,1], 'r')

for index in range(controller.numWP):
    wp = controller.way_points[index]
    plt.plot(wp[0], wp[1], '*b')

plt.show()
