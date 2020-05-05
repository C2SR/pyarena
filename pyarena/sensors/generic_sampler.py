from ..core import sensor
import numpy as np

class GenericSampler(sensor.Sensor):
    def __init__(self,**kwargs):
        # retrieving parameters
        if 'world' not in kwargs:
            raise KeyError("[Sensor/GenericSampler] Must specify a world")

        world = kwargs['world']
        noise = kwargs['noise'] if 'noise' in kwargs else np.zeros([2,1])
        
        # Storing parameters
        self.world = world
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
        measurement = self.map[pos[0,0], pos[0,1]]
        return measurement

