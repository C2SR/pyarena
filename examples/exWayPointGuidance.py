
## Necessary imports
import pyArena.control.guidance2D as pyacontrol
from pyArena.vehicles.unicycle import Unicycle

import numpy as np
import rospy


# Vehicle parameters
x_init = np.array([10.0,  10.0, np.pi/2])
kwargsSystem = {'initialCondition': x_init, 'dt': .01, 'real_time': True}
vehicle = Unicycle(**kwargsSystem)


## Waypoint controller parameters
wp = np.array([-5.0, 0.0])

kwargsGuidance = {'speed': 1, 'dt': 0.05, 'real_time': True}
guidanceLaw = pyacontrol.LOSUnicycle(**kwargsGuidance)
guidanceLaw.set_waypoint(wp, speed=1, lookahead=1.5)

## Create pyArena simulation object and simulate
vehicle.run()
guidanceLaw.run()
rospy.spin()

