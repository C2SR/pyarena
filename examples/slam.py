import numpy as np
import time

from pyarena.vehicles.unicycle import Unicycle
from pyarena.world.landmark_world import LandmarkWorld
from pyarena.sensors.landmark_sensor import LandmarkSensor  
from pyarena.sensors.velocity_sensor import VelocitySensor  
from pyarena.plots.landmark_slam import LandmarkSLAM
from pyarena.slam.ekf_slam_landmark import EKFSLAMLandmark

# Setting seed for replication of experiment
np.random.seed(0)

# Vehicle 
x0 = np.array([.0,.0,0])
kwargsUnicycle = {'x0': x0}
mvehicle = Unicycle(**kwargsUnicycle)

# World 
kwargsWorld = {'width': 10, 'height': 10, 'nb_landmarks': 10}
mworld = LandmarkWorld(**kwargsWorld)

# Velocity Sensor
noise = np.array([0.05,0.02])
kwargsVelocitySensor = {'noise': noise}
mvelocity_sensor = VelocitySensor(**kwargsVelocitySensor)

# Landmark Sensor
noise = np.array([0.05,0.05])
kwargsLandmarkSensor = {'world': mworld,'max_range': 5.0, 'noise': noise}
mlandmark_sensor = LandmarkSensor(**kwargsLandmarkSensor)

# State estimation
motion_noise = np.array([0.05,0.02])
measurement_noise = np.array([0.05,0.05])
Sigma0 = np.diag([.05,.05,.0])
kwargsEKFSLAM = {'nb_landmarks': 10, 'Sigma0': Sigma0, 'motion_noise':motion_noise, 'measurement_noise': measurement_noise}
mestimator = EKFSLAMLandmark(**kwargsEKFSLAM)

# Plot
kwargsPlot = {'world': mworld}
mplot = LandmarkSLAM(**kwargsPlot)

# Loop
while(1):
    # Simulate vehicle
    u = np.random.rand(2,1) + np.array([0.0,-.4]).reshape(2,1)
    x = mvehicle.run(dt=0.05,u=u)
    # Simulate sensors
    odometry = mvelocity_sensor.sample(dt=.05,x=x)
    if np.random.rand() < .1:
        measurements = mlandmark_sensor.sample(dt=0,x=x)
    else:
        measurements = None       
    # Localization
    x_est, Sigma = mestimator.run(dt=.05, u=odometry, measurements=measurements)
    # Plot
    mplot.update(x_ground_truth=x,x_est=x_est,measurements=measurements, Cov=Sigma)
    time.sleep(.01)

