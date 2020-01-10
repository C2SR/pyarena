from ..core import controller

import numpy as np
import matplotlib.pyplot as plt


"""
Trajectory tracking controller for 2D vehicles
"""
class TrajectoryTracking2D(controller.StaticController):

    def __init__(self, **kwargs):
        # Retrieving parameters
        if 'pd' not in kwargs:
            raise KeyError("Must specify DESIRED TRAJECTORY pd")
        if 'pdDot' not in kwargs:
            raise KeyError("Must specify DESIRED TRAJECTORY DERIVATIVE pdDot")

        self.funpd = kwargs['pd']
        self.funpdDot = kwargs['pdDot'] 
        self.K = kwargs['gain'] if 'gain' in kwargs else np.array([[1., 0.],[0., .1]])
        self.eps = kwargs['eps'] if 'eps' in kwargs else np.array([1., 0.])
        self.draw_plot = kwargs['plot'] if 'plot' in kwargs else True
        scale = kwargs['scale'] if 'scale' in kwargs else 1.0
        axis = kwargs['axis'] if 'axis' in kwargs else np.array([-50,50,-50,50])
        
        # Pre-computing constants
        self.invDelta = np.linalg.pinv(np.array([[1.0, -self.eps[1]], [0.0, self.eps[0]]]))

        # Plot configuration
        if (self.draw_plot):
            self.vehicle_contour= scale * np.array([[-1,2,-1,-1],[-1,0,1,-1]])
            plt.axis(axis)
            plt.ion()
            plt.show()
        
        # Flags
        self.is_plot_ready = False 

        # Initializing parent class
        kwargsController = {'x_dimension': 3, 'u_dimension': 2}
        kwargs.update(kwargsController)    
        
        super().__init__(**kwargs)

    """
    Update trajectory online
    """
    def update_trajectory(self, pd, pdDot):
        self.funpd = pd
        self.funpdDot = pdDot

    """
    Trajectory tracking algorithm
    """
    def compute_input(self, t, x):
        # Current pose of the vehicle
        pos = x[0:2]
        heading = x[2]
        Rot = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])

        # Trajectory
        pos_des = self.funpd(t)
        pos_des_dot = self.funpdDot(t)
        
        # Control law
        e =  (Rot.T)@(pos - pos_des) + self.eps
        u_ff = (Rot.T)@pos_des_dot
        u = self.invDelta@(-self.K@e + u_ff)

        # Call plot if requested by user
        if self.draw_plot:
            self.plot(pos_des, pos, Rot)
        
        return u

    """
    Plot routine for online vizualization
    """
    def plot(self, pd, p, R):
        if (self.is_plot_ready == False):
            # Drawing first frame
            current_vehicle_contour = R@self.vehicle_contour + p[0:2].reshape(2,1)
            self.traj_line, self.state_line = plt.plot(pd[0], pd[1],'--k', p[0], p[1], '-b')
            self.vehicle_marker = plt.plot(current_vehicle_contour[0], current_vehicle_contour[1], color='g')[0]
            self.traj_line.set_label("Desired trajectory")
            self.state_line.set_label("Real trajectory")
            plt.legend()
            plt.grid()
            # Update flag to not re-initialize plot
            self.is_plot_ready = True
        else:
            current_vehicle_contour = R@self.vehicle_contour + p[0:2].reshape(2,1)
            self.traj_line.set_xdata(np.append(self.traj_line._x, pd[0]))
            self.traj_line.set_ydata(np.append(self.traj_line._y, pd[1]))
            self.state_line.set_ydata(np.append(self.state_line._y, p[1]))  
            self.state_line.set_xdata(np.append(self.state_line._x, p[0]))
            self.vehicle_marker.set_xdata(current_vehicle_contour[0])
            self.vehicle_marker.set_ydata(current_vehicle_contour[1])                
            plt.draw()
            plt.pause(0.0001)
