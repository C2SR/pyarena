# # Lawnmower control for unicyle robot
# TODO: Document here

# Do this once

## Necessary imports
from pyArena.vehicles.unicycle import Unicycle
from pyArena.planning.lawnmower import Lawnmower2D
from pyArena.sampling.standard_gp import StandardGP
import pyArena.control.guidance2D as pyacontrol
from pyArena.datasets.intel_berkeley_world import IntelBerkeleyWorld 
from pyArena.datasets.intel_berkeley_sensor import IntelBerkeleySensor 
import matplotlib.pyplot as plt

import numpy as np

import rospy

map_resolution = 1.

# World
world = IntelBerkeleyWorld()
world.run()

# Vehicle parameters
x_init = np.array([0.0, 0.0, 0])
kwargsSystem = {'initialCondition': x_init, 'dt': .01, 'real_time': True}
vehicle = Unicycle(**kwargsSystem)
vehicle.run()

# Sensors
sensor = IntelBerkeleySensor()
sensor.run()

# Plan
step = 2
start = world.origin + step + map_resolution/2.
end = world.origin + np.array([world.width, world.height]) - step - map_resolution/2.
planner = Lawnmower2D()
planner.compute_plan(start, end, step)
planner.run()

# Map
kwargsMap = {'width': world.width, 'height': world.height, 'resolution': map_resolution/4.}
mmap = StandardGP(**kwargsMap)

#mmap.update_training_input(planner.wp_plan[:,0:4])
wp=np.array([[2.5,2.5], [6, 2.5] ,[10, 2.5], [15,2.5],[20,2.5], [25,2.5],[30,2.5], [35,2.5], 
             [35,4.5], [30, 4.5], [25,4.5],[20,4.5], [15,4.5],[10,4.5], [6, 4.5], [2.5,4.5], [2.5,6.5]]).T
mmap.update_training_input(wp)
mmap.run()

## Waypoint controller parameters
axis = np.array([world.origin[0], world.origin[0]+world.width, world.origin[1], world.origin[1]+world.height ])
scale = 0.5
kwargsGuidance = {'plot': False, 'speed': .75, 'look_ahead': .75, 'dt': 0.05, 'axis': axis, 'scale': scale, 'real_time': True }
guidanceLaw = pyacontrol.LOSUnicycle(**kwargsGuidance)
guidanceLaw.run()

# Run ROS
rospy.spin()
