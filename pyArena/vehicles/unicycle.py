# TODOC Summary: ...

import numpy as np

# IDEA Should that become a class?
# Specify robot kinematics
def unicycle(t, x, u):
    return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1], u[2]])