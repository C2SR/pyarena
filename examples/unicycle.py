import numpy as np
import time

from pyArena.vehicles import unicycle

# Setting seed for replication of experiment
np.random.seed(0)

# Vehicle 
x0 = np.array([.0,.0,0])
kwargsUnicycle = {'x0': x0}
mvehicle = Unicycle(**kwargsUnicycle)

# Loop
while(1):
    u = np.random.rand(2)
    x = vehicle.run(dt=0.1,u=u)
    print(x)
    time.sleep(.2)

