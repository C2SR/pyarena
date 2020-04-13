from ..core import sensor
import numpy as np

class LandmarkSensor(sensor.Sensor):
    def __init__(self,**kwargs):
        # retrieving parameters
        if 'world' not in kwargs:
            raise KeyError("[Sensor/LandmarkSensor] Must specify a world")

        world = kwargs['world']
        max_range = kwargs['max_range'] if 'max_range' in kwargs else 1.0
        noise = kwargs['noise'] if 'noise' in kwargs else np.zeros([2,1])
        
        # Storing parameters
        self.world = world
        self.max_range = max_range
        self.noise_std = noise.reshape(2,1)

        kwargsSensor = {'x_dimension': 3}
        kwargs.update(kwargsSensor)

        super().__init__(**kwargs) 

    """
    Sampling function
    """
    def sample(self, dt, x):
        # position and orientation of the robot
        pos = x[0:2].reshape([2,1])
        theta = x[2,0]
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        # distance from the robot to the landmarks
        landmarks2robot = R@(self.world.landmarks['coordinate']-pos) 
        distance = np.linalg.norm(landmarks2robot, axis=0) 
        # Detecting landmarks within range
        measurements = {}
        measurements['id'] = self.world.landmarks['id'][distance < self.max_range]
        measurements['coordinate'] = landmarks2robot[:,distance < self.max_range] 
        # Generating and pertubing with gaussian noise
        nb_detected_landmarks = np.sum(distance < self.max_range)
        measurement_noise = np.random.normal(scale=self.noise_std, size=(2,nb_detected_landmarks))
        measurements['coordinate'] += measurement_noise                                     
        return measurements

