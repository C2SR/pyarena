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
nb_landmarks = 50
kwargsWorld = {'width': 10, 'height': 10, 'nb_landmarks': nb_landmarks}
mworld = LandmarkWorld(**kwargsWorld)
#mworld.landmarks['coordinate'] = np.array([[2.,1.],[4.,1.],[6,1],[8,1],
#                                           [8.,4.],[8.,6.],[2.,4.],[2.,6.],[4,8],
#                                           [2.,9.],[4.,9.],[6,9],[8,9]]).T
# Velocity Sensor
motion_noise = np.array([0.05,0.02])
kwargsVelocitySensor = {'noise': motion_noise}
mvelocity_sensor = VelocitySensor(**kwargsVelocitySensor)

# Landmark Sensor
measurement_noise = np.array([0.05,0.05])
kwargsLandmarkSensor = {'world': mworld,'max_range': 4.0, 'noise': measurement_noise}
mlandmark_sensor = LandmarkSensor(**kwargsLandmarkSensor)
# State estimation
Sigma0 = np.diag([.0,.0,.0])
kwargsEKFSLAM = {'nb_landmarks': nb_landmarks, 'Sigma0': Sigma0, 'motion_noise':motion_noise, 'measurement_noise': measurement_noise}
mestimator = EKFSLAMLandmark(**kwargsEKFSLAM)

# Plot
kwargsPlot = {'world': mworld}
mplot = LandmarkSLAM(**kwargsPlot)

# Loop
state = 0
while(1):
    # Simulate vehicle
    u = np.random.rand(2,1) + np.array([0.0,-.425]).reshape(2,1)
    x = mvehicle.run(dt=0.05,u=u)
    # Simulate sensors
    odometry = mvelocity_sensor.sample(dt=.05,x=x)
    if np.random.rand() < .5:
        measurements = mlandmark_sensor.sample(dt=0,x=x)
    else:
        measurements = None       
    # Localization
    x_est, Sigma = mestimator.run(dt=.05, u=odometry, measurements=measurements)
    # Plot
    mplot.update(x_ground_truth=x,x_est=x_est,measurements=measurements, Cov=Sigma)
    time.sleep(.01)

