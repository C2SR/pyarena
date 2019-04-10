from pyArena.basics.dynamical_system import DynamicalSystem
from pyArena.vehicles import unicycle
from pyArena.control.path_following import PathFollowing
from pyArena.simulator import simulator

import matplotlib.pyplot as plt
import numpy as np

# Specify simulation parameters
nx = 4 
nu = 3
Tsim = 100
dt = 0.05
K = np.eye(2)
eps = np.array([0.1, 0])
vd = 1
x_init = np.array([10.0, 0.0, 0.0, 0.0])

# Specify desired path
radius = 10
a = 0.1

pd = lambda l: np.array([radius*np.sin(a*l), radius*np.cos(a*l)])

# TODO: Learn Symbolic operations in Python
pdD = lambda l: np.array([a*radius*np.cos(a*l), -a*radius*np.sin(a*l)])

kwargsController = {'pd': pd, 'pdD': pdD, 'gain': K, 'eps': eps, 'vd': vd}
controller = PathFollowing(**kwargsController)

kwargsSystem = {'nx':nx, 'nu':nu,\
                'stateEquation': lambda t,x,u: unicycle.unicycle(t,x,u),\
                'initialCondition': x_init, 'controller': controller}

system = DynamicalSystem(**kwargsSystem)

kwargsSimulation = {'system': system, 'simTime': Tsim, 'dt': dt}
pyA = simulator.pyArena(**kwargsSimulation)
dataLog = pyA.run()

pdVec = pd(np.linspace(0,100,1000))
plt.plot(pdVec[0,:], pdVec[1,:])
plt.plot(dataLog.stateTrajectory[:,0], dataLog.stateTrajectory[:,1])
plt.show()
