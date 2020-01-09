from ..core import system
import numpy as np

class Unicycle(system.DynamicSystem):

    def __init__(self,**kwargs):
        kwargsSystem = {'x_dimension': 3, 
                        'u_dimension': 2} 

        kwargs.update(kwargsSystem)

        super().__init__(**kwargs)

    # Unicycle kinematic model
    def stateEquation(self, t, x, u):
        return np.array([u[0]*np.cos(x[2]), 
                         u[0]*np.sin(x[2]), 
                         u[1]])    
