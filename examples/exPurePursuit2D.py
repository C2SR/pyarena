import sys
sys.path.append('..')

import pyArena.core as pyacore
from pyArena.control import pure_pursuit_2D 
import pyArena.vehicles.underactuatedvehicle as pyavehicle

import matplotlib.pyplot as plt

import numpy as np

## Create a path following controller class
class Task(pyacore.controller.StaticController):

    def __init__(self, **kwargs):
        
        super().__init__()
        
        # Go2Point parameters
        kwargsController = kwargs['kwargsController']
        self.controller = pure_pursuit_2D.PurePursuit2D(**kwargsController)

    def computeInput(self, t, x, *args):
        return self.controller.run(x,[1,1])

# Simulation parameters
Tsim = 10
dt = .1
x_init = np.array([0.0, 0.0, np.pi/2])


# Initializing task parameters 
kwargs = {}
# controller
kwargsController = {'k_v': 1, 'k_w': 2}
kwargs['kwargsController'] =  kwargsController

ttTask = Task(**kwargs)

# [Dynamic system] Unicycle model
kwargsSystem = {'initialCondition': x_init, 'controller': ttTask}
system = pyavehicle.UnicycleKinematics(**kwargsSystem)

## Create pyArena simulation object
kwargsSimulation = {'system': system, 'simTime': Tsim, 'dt': dt}
pyA = pyacore.simulator.pyArena(**kwargsSimulation)
dataLog = pyA.run()

## Plot results
plt.plot(dataLog.stateTrajectory[:,0], dataLog.stateTrajectory[:,1], 'r')
plt.show()