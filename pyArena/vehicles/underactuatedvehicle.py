from ..core import system
import numpy as np

class UnicycleKinematics(system.InlineControlSystem):

    def __init__(self,**kwargs):

        def unicycle(t, x, u):
            return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])

        kwargsSystem = {'nx': 3, 'nu':2, 'inlineStateEquation': lambda t,x,u: unicycle(t,x,u)}

        kwargs.update(kwargsSystem)

        super().__init__(**kwargs)
