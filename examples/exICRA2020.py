## -----------------------------Necessary imports ------------------------------##
import sys
sys.path.append('..')

from pyArena.algorithms import gaussian_process as gp
import pyArena.core as pyacore
import pyArena.control.guidance2D as pyaguidance
import pyArena.vehicles.underactuatedvehicle as pyavehicle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

##--------------------------- Parameters----------------------------------------##
# TODO: What is nx, nu, and dt?
# Simulation parameters
Tsim = 2000  # Simulation time
dt = 1  # Sampling time
# Robot
x_init = np.array([-0.0, 0.0, np.pi / 3])  # Initial 2D pose of the robot
# Environment (2D scalar field)
world_size_x0 = [-150, 150]
world_size_x1 = [-150, 150]

func = lambda x: 100 * np.cos(np.pi / 5 * x[0]) + 100 * np.sin(np.pi / 5 * x[1])
func = lambda x: (x[0] ** 2) / 1000 + (x[1] ** 2) / 4000 - (((x[0] - 10) ** 2) / 4000 + ((x[1] + 20) ** 2) / 2000) + (
            (x[0] * x[1]) / 2000 + (x[0] * x[1]) / 600)
# GP parameters
tau_s = 40  # AKA "lenght scale"
kernel = lambda x1, x2: np.exp(-(0.5 / np.float_power(tau_s, 2)) * np.float_power(np.linalg.norm(x1 - x2), 2))
# !!!!!! What are these parameters? !!!!!!!
nx = 3
nu = 2


## --------------------------Create the sampling controller--------------------------------##
class RoboticSampling(pyacore.controller.StaticController):

    def __init__(self, **kwargs):

        super().__init__()

        # Initialize variables that are not listed in **kwargs
        self.numWP = 0
        self.way_points = list()
        self.arrivedAtWaypoint = False
        self.UpdateGPModel = False

        # Initialize variables listed in **kwargs
        self.max_WP_dist = kwargs['kwargsWPGenerator']['max_WP_dist']
        world_size_x1 = kwargs['kwargsWPGenerator']['world_size_x1']
        x0 = np.linspace(world_size_x0[0], world_size_x0[1], 20)
        x1 = np.linspace(world_size_x1[0], world_size_x1[1], 20)
        self.grid_X0, self.grid_X1 = np.meshgrid(x0, x1)

        ## Guidance law
        self.guidanceLaw = pyaguidance.LOSUnicycle(**kwargs['kwargsGuidance'])
        self.guidanceLaw.isWayPointReached = True

        # Create a GP Regression process
        # TODO: Hyper-parameter estimation
        self.gp = gp.GPRegression(**kwargs['kwargsGP'])

    def computeInput(self, t, x, *measurements):
        # Update flags
        self.arrivedAtWaypoint = self.guidanceLaw.isWayPointReached

        ### Sampling policy ###
        self.WPSamplingPolicy(t, x, *measurements)

        ### GP update model ###
        self.GPUpdateModel(t, x, *measurements)

        ### WayPoint generator ###
        self.WaypointGenerator(t, x, *measurements)

        ### Guidance Law ###
        control_input = self.guidanceLaw.run(t, x)

        return control_input

    ### Samling Policy ###
    def WPSamplingPolicy(self, t, x, *measurements):
        self.UpdateGPModel = self.arrivedAtWaypoint

    ### Model for updating GP ###
    def GPUpdateModel(self, t, x, *measurements):
        if self.UpdateGPModel is True:
            self.gp.trainGPIterative(x[0:2], measurements[0], True)

    ### WP Generator ###
    def WaypointGenerator(self, t, x, *measurements):
        # If no update on the model, do not compute new waypoint
        if self.UpdateGPModel is False:
            return

        # Setting initial values such next WP generator is excited
        next_WP = x[0:2]
        next_WP_var = 0

        xTest = np.stack((self.grid_X0.reshape(self.grid_X0.shape[0] * self.grid_X0.shape[1]), \
                          self.grid_X1.reshape(self.grid_X1.shape[0] * self.grid_X1.shape[1])))

        # Evaluate the GP model to choose next waypoint
        numel = self.grid_X0.shape[0] * self.grid_X0.shape[1]
        ypred = np.zeros(numel)
        var_pred = np.zeros(numel)
        for index in range(0, numel):
            ypred[index], var_pred[index] = self.gp.predict_value(xTest[:, index])
            dist = np.linalg.norm(x[0:2] - xTest[:, index])
            # Check if current point is elegible as next WP
            if ((var_pred[index] > next_WP_var) and (dist < self.max_WP_dist)):
                next_WP_var = var_pred[index]
                next_WP = xTest[:, index]

        ## Plotting ###
        self.plotGPGrid(ypred, var_pred)

        # Send next waypoint to the guidance controller
        self.guidanceLaw.setWayPoint(next_WP)
        self.way_points = np.append(self.way_points, next_WP)
        self.numWP += 1

    def plotGPGrid(self, ypred, var_pred):
        # Plotting prediction
        h1 = plt.figure(1)
        ax3 = plt.subplot(2, 2, 3)
        p = ax3.pcolor(self.grid_X0, self.grid_X1, ypred.reshape([self.grid_X0.shape[0], self.grid_X0.shape[1]]),
                       cmap=cm.jet, vmin=-20, vmax=20)
        # Plotting variance
        ax4 = plt.subplot(2, 2, 4)
        p = ax4.pcolor(self.grid_X0, self.grid_X1, var_pred.reshape([self.grid_X0.shape[0], self.grid_X0.shape[1]]),
                       cmap=cm.jet)
        plt.pause(.1)


## --------------------------END sampling controller--------------------------------##

## --------------------------START Simulation---------------------------------------##

# Environment
kwargsSensor = {'ScalarField': func, 'covariance': 0.1}
sensorEnv = pyacore.sensors.StaticScalarField2D(**kwargsSensor)

## Define adaptive sampling controller
kwargsGuidance = {'speed': 1}
kwargsGP = {'kernel': kernel, 'measurementNoiseCov': 0.2}
kwargsWPGenerator = {'max_WP_dist': 2 * tau_s, 'world_size_x0': world_size_x0, 'world_size_x1': world_size_x1}
kwargsAdaptiveSampling = {'kwargsGuidance': kwargsGuidance, 'kwargsGP': kwargsGP,
                          'kwargsWPGenerator': kwargsWPGenerator}
controller = RoboticSampling(**kwargsAdaptiveSampling)

## Create a system with Controller
kwargsSystem = {'initialCondition': x_init, 'controller': controller, 'sensor': sensorEnv}
system = pyavehicle.UnicycleKinematics(**kwargsSystem)

## Create pyArena simulation object and simulate
kwargsSimulation = {'system': system, 'simTime': Tsim, 'dt': dt}
pyA = pyacore.simulator.pyArena(**kwargsSimulation)

# Plot ground truth
numTruth = 100
xTruth, yTruth = sensorEnv.getFullPlotData(xmin=[-150, -150], xmax=[150, 150], numGrid=numTruth)
X0 = xTruth[0].reshape(numTruth, numTruth)
X1 = xTruth[1].reshape(numTruth, numTruth)
Y = yTruth.reshape(numTruth, numTruth)

h1 = plt.figure(1)
ax1 = plt.subplot(2, 2, 1)
p = ax1.pcolor(X0, X1, Y, cmap=cm.jet, vmin=-20, vmax=20)
cb = h1.colorbar(p)
plt.show(0)

# Run simulation
dataLog = pyA.run()

## Plot results

# Plot ground truth
h2 = plt.figure(2)
ax1 = plt.subplot(2, 2, 1)
p = ax1.pcolor(X0, X1, Y, cmap=cm.jet, vmin=-20, vmax=20)
cb = h1.colorbar(p)
plt.show(0)

# Plot the learned GP model

num = 100

xmin = [-150, -150]
xmax = [150, 150]
x0 = np.linspace(xmin[0], xmax[0], num)
x1 = np.linspace(xmin[1], xmax[1], num)
X0, X1 = np.meshgrid(x0, x1)
xTest = np.stack((X0.reshape(X0.shape[0] * X0.shape[1]), \
                  X1.reshape(X1.shape[0] * X1.shape[1])))

ypred = np.zeros(num ** 2)
var_pred = np.zeros(num ** 2)

for index in range(0, num ** 2):
    ypred[index], var_pred[index] = controller.gp.predict_value(xTest[:, index])

# h2 = plt.figure(2)
ax3 = plt.subplot(2, 2, 3)
p = ax3.pcolor(X0, X1, ypred.reshape([num, num]), cmap=cm.jet, vmin=-20, vmax=20)
ax4 = plt.subplot(2, 2, 4)
p = ax4.pcolor(X0, X1, var_pred.reshape([num, num]), cmap=cm.jet)
plt.pause(.1)

# Plot the robot position

ax2 = plt.subplot(2, 2, 2)
plt.plot(dataLog.stateTrajectory[:, 0], dataLog.stateTrajectory[:, 1], 'r')

way_points = controller.way_points.reshape(controller.numWP, 2)
print(way_points)
for index in range(controller.numWP):
    wp = way_points[index]
    plt.plot(wp[0], wp[1], '*b')

plt.show()
