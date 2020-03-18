from ..core import controller

import numpy as np
import matplotlib.pyplot as plt

"""
Waypoint control for 2D vehicles
"""
class LOSUnicycle(controller.StaticController):
    def __init__(self, **kwargs):
        # Retrieving parameters
        self.speed = kwargs['speed'] if 'speed' in kwargs else 1
        self.look_ahead = kwargs['look_ahead'] if 'look_ahead' in kwargs else 1        
        self.draw_plot = kwargs['plot'] if 'plot' in kwargs else True
        scale = kwargs['scale'] if 'scale' in kwargs else 0.2
        axis = kwargs['axis'] if 'axis' in kwargs else np.array([-15,15,-15,15])        
       
        # Initializing variables
        self.wp_final = None
        self.wp_init = None

        # Plot configuration
        if (self.draw_plot):
            self.vehicle_contour= scale * np.array([[-1,2,-1,-1],[-1,0,1,-1]])
            plt.axis(axis)
            plt.ion()
            plt.show()

        # Flags
        self.has_waypoint = False    
        self.is_plot_ready = False

        # Initializing parent class
        kwargsController = {'x_dimension': 3, 'u_dimension': 2}
        kwargs.update(kwargsController)
        super().__init__(**kwargs)

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

    def update_reference(self, t, ref):
        if (np.array_equal(ref, self.wp_final)):
            return None


        if (self.wp_final is not None):
            self.wp_init = self.wp_final
            self.wp_final = ref
            path = (self.wp_final - self.wp_init).reshape(2,1)
            self.proj_operator = path @ path.T / (path.T@path)
        else: 
            self.wp_final = ref
        
        print('Received a new waypoint:')
        print('init', self.wp_init, 'final', self.wp_final)
        self.has_reached_waypoint = False
        self.has_waypoint = True        

    """
    Guidance algorithm
    """
    def compute_input(self, t, x):
        # In case of no waypoiny, send zero velocity commands 
        if not self.has_waypoint:
            return np.array([0.,0.])

        # Set the initial robot position as initial waypoiny
        if self.wp_init is None:
            self.wp_init = np.array(x[0:2])
            path = (self.wp_final - self.wp_init).reshape(2,1)
            self.proj_operator = path @ path.T / (path.T@path)

        # Current pose of the vehicle
        pos = np.array(x[0:2])
        heading = np.array(x[2])
        Rot = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])

        # Line of sight (LoS) algorithm
        path_ref = self.proj_operator @ (pos - self.wp_init).reshape(2,1) + self.wp_init.reshape(2,1)
        los_angle = np.arctan2(self.wp_final[1] - self.wp_init[1], self.wp_final[0] - self.wp_init[0])
        Rot_los = np.array([[np.cos(los_angle), -np.sin(los_angle)],[np.sin(los_angle), np.cos(los_angle)]])
   
        pos_ref = Rot_los.T@(pos - self.wp_init)
        heading_desired = - np.arctan(pos_ref[1]/self.look_ahead) + los_angle

        if (np.sqrt((self.wp_final - pos)@(self.wp_final - pos)) < .1):
            v_lin = 0
            w_ang = 0
            self.has_reached_waypoint = True
        else:
            v_lin = self.speed
            w_ang = -0.6*(heading - heading_desired)
        
        if self.draw_plot:
            self.plot( path_ref, pos, Rot)
        return np.array([v_lin, w_ang])

    """
    Plot routine for online vizualization
    """
    def plot(self, pd, p, R):
        if (self.is_plot_ready == False):
            # Drawing first frame
            current_vehicle_contour = R@self.vehicle_contour + p[0:2].reshape(2,1)
            self.traj_line, self.state_line = plt.plot(pd[0], pd[1],'--k', p[0], p[1], '-b')
            self.vehicle_marker = plt.plot(current_vehicle_contour[0], current_vehicle_contour[1], color='g')[0]
            plt.plot(self.wp_final[0], self.wp_final[1],'xr',label='WP')
            self.traj_line.set_label("Reference path")
            self.state_line.set_label("Real ")
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
