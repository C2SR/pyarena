# TODOC
import numpy as np


## Create a path following controller class
class PurePursuit2D:

    def __init__(self, **kwargs):
        self.k_v = kwargs['k_v']
        self.k_w = kwargs['k_w'] 

    def run(self, x, x_des):

        p = x[0:2]
        p_des = x_des[0:2]
        dist = np.linalg.norm(p-p_des)

        theta = x[2]
        theta_des = np.arctan2(p_des[1]-p[1],p_des[0]-p[0])
        theta_diff = theta_des - theta

        if theta_diff > np.pi:    # TODO this should be a reusable function
            theta_diff -= 2*np.pi
        elif theta_diff < -np.pi:
            theta_diff += 2*np.pi

        u = np.zeros(2)
        u[0] = min(self.k_v*dist,1)
        u[1] = self.k_w*theta_diff
     
        return u










   