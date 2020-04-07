# Python libraries
import numpy as np
from scipy.integrate import solve_ivp as ode45
from abc import ABC, abstractmethod

## Vehicle (abstract) class ##
class Vehicle(ABC):

    def __init__(self, **kwargs):
        # Checking for missing parameters
        if 'x_dimension' not in kwargs:
            raise KeyError("[Vehicle] Must specify number of states x_dimension")
        if 'u_dimension' not in kwargs:
            raise KeyError("[Vehicle] Must specify number of inputs u_dimension")
            
        # Retrieving parameters
        self.x_dimension = kwargs['x_dimension']
        self.u_dimension = kwargs['u_dimension']
        
        # Initializing varibles
        self.x = kwargs['x0'] if 'x0' in kwargs else np.zeros(self.x_dimension)

    """
    State equation that defines the dynamics of the vehicle
    """
    @abstractmethod
    def stateEquation(self, t, x, u):
        pass

    """
    Run the system dynamics forward in time by a single time step.
    """
    def run(self, dt, u):
        # Iterating the state of the vehicle
        ode_fun = lambda t, x: self.stateEquation(dt, self.x.flatten(), u.flatten())
        sol = ode45(ode_fun, [0, dt], self.x.flatten())
        self.x = sol.y[:,-1].reshape(self.x_dimension,1)        

        return np.copy(self.x)