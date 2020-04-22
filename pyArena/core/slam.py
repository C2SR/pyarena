# Python libraries
import numpy as np
from abc import ABC, abstractmethod

## SLAM (abstract) class ##
class SLAM(ABC):

    def __init__(self, **kwargs):
        # Checking for missing parameters
        if 'x_dimension' not in kwargs:
            raise KeyError("[SLAM] Must specify number of states x_dimension")
            
        # Retrieving parameters
        x_dimension = kwargs['x_dimension']
        x0_est = kwargs['x0_est'] if 'x0_est' in kwargs else np.zeros(x_dimension)

        # Initializing parameters
        self.x_dimension = x_dimension
        self.x_est = x0_est.reshape(x_dimension,1)

    """
    SLAM routine that builds that map and estimates the pose of the vehicle using odometry
    and measurements.
    """
    @abstractmethod
    def run(self, dt, u, measurement=None):
        pass

