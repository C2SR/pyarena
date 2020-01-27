from ..core import planner

import numpy as np
import matplotlib.pyplot as plt

"""
Lawnmower planner for 2D vehicles
"""
class Lawnmower2D(planner.StaticPlanner):
    def __init__(self, **kwargs):
        # Initializing parent class
        kwargsController = {'x_dimension': 2, 'u_dimension': 2}
        kwargs.update(kwargsController)

        self.current_wp = None

        super().__init__(**kwargs)

        #vec_x = np.linspace(0,10,num=2,dtype='float')
        #vec_y = np.arange(0,5,1,dtype='float')  

    """
    Set next waypoint
    """
    def set_waypoint(self, waypoint, speed, lookahead=5.):
        self.wp_final = waypoint
        self.speed = speed
        self.look_ahead = lookahead

        self.wp_init = None
        self.has_reached_waypoint = False
        self.has_waypoint = True

    """
    
    """
    def compute_plan(self, origin, final, step=1):
        going_left = False

        x = np.array([origin[0], final[0]])
        y = np.arange(origin[1],final[1]+step, step, dtype='float')  

        self.num_wp = len(x)*len(y)
        self.wp_plan = np.zeros([2,self.num_wp])
        k = 0
        for yi in y:
            if not going_left:
                for xi in x:
                    self.wp_plan[:,k] = np.array([xi,yi])
                    k+=1
            else:
                for xi in x[::-1]:
                    self.wp_plan[:,k] = np.array([xi,yi])
                    k+=1
            
            going_left = not going_left

    """
    Planning algorithm
    """
    def compute_input(self, t, x):
        if (self.current_wp is None):
            self.current_wp_num = 0     
            print("Starting lawnmower plan")       
         # Check if we have reached next waypoint
        elif (np.linalg.norm(x-self.current_wp)<1):
            self.current_wp_num += 1
            print("Reached waypoint!")
        
        # Send waypoint
        if (self.current_wp_num < self.num_wp):
            self.current_wp = self.wp_plan[:,self.current_wp_num]
            self.send_plan(self.current_wp)

        

