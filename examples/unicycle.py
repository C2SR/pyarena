import numpy as np
import time

from pyArena.vehicles import unicycle
from pyArena.sensors import landmark 

# Vehicle parametes
x0 = np.array([0.0,0.0,0])
kwargsUnicycle = {'x0': x0}
vehicle = unicycle.Unicycle(**kwargsUnicycle)

# Sensor
kwargsLandmark2D = {'max_range': 5.0}
msensor = landmark.Landmark2D(**kwargsLandmark2D)
msensor.create_world(world_size=np.array([10,10]), nb_landmarks=10)

# Loop
while(1):
    u = np.random.rand(2)
    x = vehicle.run(dt=0.1,u=u)
    measurements = msensor.sample(x)
    print(measurements)
    #print(x)
    time.sleep(.2)

