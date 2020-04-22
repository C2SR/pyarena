from ..core import sensor
import numpy as np

class VelocitySensor(sensor.Sensor):
    def __init__(self,**kwargs):
        # Retrieving parameters
        x0 = kwargs['x0'] if 'x0' in kwargs else np.zeros(3)
        noise = kwargs['noise'] if 'noise' in kwargs else np.zeros(2)

        # Storing parameters
        self.x = x0.reshape(3,1)
        self.noise_std = noise.reshape(2,1)

        kwargsSensor = {'x_dimension': 3}
        kwargs.update(kwargsSensor)
        super().__init__(**kwargs) 

    """
    Sampling function
    """
    def sample(self, dt, x):
        # Euler differentiation
        diff = (x - self.x) / dt
        # Computing velocities
        v_lin = np.linalg.norm(diff[0:2, 0])
        w_ang =  (diff[2,0] + np.pi) % (2 * np.pi) - np.pi
        velocity = np.array([v_lin, w_ang]).reshape(2,1)
        # Generating and perturbingaussian noise
        measurement_noise = np.random.normal(scale=self.noise_std, size=(2,1))
        velocity += measurement_noise
        # Saving state for next iteration
        self.x = np.copy(x)
        return velocity