# # Trajectory tracking control for unicyle robot
# TODO: Document here

# Do this once

## Necessary imports
import pyArena.control.trajectorytracking as pyacontrol
from pyArena.vehicles.unicycle import Unicycle
import matplotlib.pyplot as plt

import numpy as np

import rospy

# Vehicle parameters
x_init = np.array([10.0, 0.0, np.pi/2])
kwargsSystem = {'initialCondition': x_init, 'dt': .01, 'real_time': True}
vehicle = Unicycle(**kwargsSystem)

## Trajectory tracking parameters
K = np.array([[1, 0.0],[0.0, 0.1]])
eps = np.array([1, 0])
radius = 30   # trajectory's radius
w = 0.05
pd = lambda t: radius*np.array([np.cos(w*t), np.sin(w*t)])
pdDot = lambda t: radius*np.array([-w*np.sin(w*t), w*np.cos(w*t)])

## Specify the controller
kwargsController = {'pd': pd, 'pdDot': pdDot, 'gain': K, 'eps': eps, 'dt': .05, 'real_time': True, 
                    'plot': True, 'scale': 1.0, 'axis': np.array([-50,50,-50,50])}
ttController = pyacontrol.TrajectoryTracking2D(**kwargsController)

## Create pyArena simulation object and simulate
vehicle.run()
ttController.run()
rospy.spin()
