import numpy as np
import time

from pyArena.vehicles.unicycle import Unicycle
from pyArena.world.landmark_world import LandmarkWorld
from pyArena.sensors.landmark_sensor import LandmarkSensor  
from pyArena.plots.landmark_localization import LandmarkLocalization

np.random.seed(0)

# Vehicle 
x0 = np.array([0.0,0.0,0])
kwargsUnicycle = {'x0': x0}
mvehicle = Unicycle(**kwargsUnicycle)

# World 
kwargsWorld = {'width': 10, 'height': 10, 'nb_landmarks': 10}
mworld = LandmarkWorld(**kwargsWorld)

# Sensor
kwargsLandmark = {'world': mworld,'max_range': 5.0}
msensor = LandmarkSensor(**kwargsLandmark)

# Plot
kwargsPlot = {'world': mworld}
mplot = LandmarkLocalization(**kwargsPlot)

# Loop
while(1):
    u = np.random.rand(2,1) + np.array([0,-.4]).reshape(2,1)
    x = mvehicle.run(dt=0.1,u=u)
    measurements = msensor.sample(x)
    mplot.update(x,x,measurements)
    #print(x)
    time.sleep(.05)

