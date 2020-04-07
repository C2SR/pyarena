# Python libraries
import numpy as np
from scipy.integrate import solve_ivp as ode45
from abc import ABC, abstractmethod

## Sensor (abstract) class ##
class Sensor(ABC):

    def __init__(self, **kwargs):
        # Checking for missing parameters
        if 'x_dimension' not in kwargs:
            raise KeyError("[Sensor] Must specify number of states x_dimension")
            
        # Retrieving parameters
        self.x_dimension = kwargs['x_dimension']
        
        # Initializing varibles
        self.x = kwargs['x0'] if 'x0' in kwargs else np.zeros(self.x_dimension)

    """
    Sampling equation that samples the world
    """
    @abstractmethod
    def sample(self, x):
        pass
