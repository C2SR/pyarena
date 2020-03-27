from ..core import vehicle
import numpy as np

class Unicycle(vehicle.Vehicle):
    def __init__(self,**kwargs):
        kwargsVehicle = {'x_dimension': 3, 
                         'u_dimension': 2} 
        
        kwargs.update(kwargsVehicle)

        super().__init__(**kwargs)
        
    # Unicycle kinematic model
    def stateEquation(self, t, x, u):
        return np.array([u[0]*np.cos(x[2]), 
                         u[0]*np.sin(x[2]), 
                         u[1]])
