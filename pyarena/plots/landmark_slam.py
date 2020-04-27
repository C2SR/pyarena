import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from . import drawings  

class LandmarkSLAM:
    def __init__(self, **kwargs):
         # Checking for missing parameters
        if 'world' not in kwargs:
            raise KeyError("[Plot/Landmark Localization] Must specify a world")
        world = kwargs['world']

        # Saving parameters
        self.world = world

        # Vehicle drawer       
        kwargsVehicle = {'vehicle': 'Unicycle','scale': 0.1}
        self.vehicle = drawings.VehicleDrawing(**kwargsVehicle)
        # Covariance drawer
        kwargsCovariance = {'nb_pts': 30}
        self.uncertainty_ellipse =  drawings.Covariance2DDrawing(**kwargsCovariance)
        
        # Artists
        self.fig, self.ax = plt.subplots()
        self.x_gt_marker = self.ax.plot([],[], linestyle='--', color='k')[0]
        self.x_est_marker = self.ax.plot([],[], linestyle='-', color='k')[0]
        self.ellipse_marker = []
        for i in range(0, world.nb_landmarks+1):
            ellipse_marker = self.ax.plot([],[], linestyle='-', color='k')[0]
            self.ellipse_marker.append(ellipse_marker)
        self.mapped_landmarks = self.ax.plot([],[],   marker='s',
                                                      markerfacecolor='b',
                                                      markeredgecolor='k',
                                                      markeredgewidth=2.0,
                                                      linestyle='None')[0]
        self.detected_landmarks = self.ax.plot([],[], marker='s',
                                                      markerfacecolor='r',
                                                      markeredgecolor='k',
                                                      markeredgewidth=2.0,
                                                      linestyle='None')[0]
        # Plotting the first frame
        self.first_frame()


    """
    Create the first plot that contains the known position of the landmarks in the world
    """
    def first_frame(self):  
        # First frame: world
        self.ax.plot(self.world.landmarks['coordinate'][0,:],
                     self.world.landmarks['coordinate'][1,:],
                     marker='s',
                     markerfacecolor='None',
                     markeredgecolor='k',
                     markeredgewidth=2.0,
                     linestyle='None')

        plt.ylim(0, self.world.width)
        plt.xlim(0, self.world.height)
        plt.grid(True)
        plt.show(0)
        plt.pause(.01)
        
        
    """
    Update the pose of the robot in the map
    """
    def update(self, x_ground_truth, x_est, measurements, Cov=np.zeros([3,3])):
        pl.figure(self.fig.number)
      
        # Computing vehicle countour
        x_gt_countour = self.vehicle.update(x_ground_truth)
        x_est_countour = self.vehicle.update(x_est)        
       
        # Updating vehicle plot
        self.x_gt_marker.set_xdata(x_gt_countour[0,:])
        self.x_gt_marker.set_ydata(x_gt_countour[1,:])
        self.x_est_marker.set_xdata(x_est_countour[0,:])
        self.x_est_marker.set_ydata(x_est_countour[1,:])  
        
        # Updating vehicle uncertainty
        x_est_ellipse = self.uncertainty_ellipse.update(Cov[0:2,0:2], x_est[0:2])
        self.ellipse_marker[0].set_xdata(x_est_ellipse[0,:])
        self.ellipse_marker[0].set_ydata(x_est_ellipse[1,:])

        # Updating landmark uncertainty
        for i in range(0, self.world.nb_landmarks):
            x_est_ellipse = self.uncertainty_ellipse.update(Cov[3+2*i:3+2*i+2, 3+2*i:3+2*i+2],
                                                            x_est[3+2*i:3+2*i+2])
            self.ellipse_marker[i+1].set_xdata(x_est_ellipse[0,:])
            self.ellipse_marker[i+1].set_ydata(x_est_ellipse[1,:])

        # Updating map plot
        self.mapped_landmarks.set_xdata(x_est[3::2,0])
        self.mapped_landmarks.set_ydata(x_est[4::2,0])          

        # Highlighting detected landmarks
        if (measurements is not None):
            self.detected_landmarks.set_xdata(self.world.landmarks['coordinate'][0,measurements['id']])
            self.detected_landmarks.set_ydata(self.world.landmarks['coordinate'][1,measurements['id']])        
        else:
            self.detected_landmarks.set_xdata([])
            self.detected_landmarks.set_ydata([])            
        plt.pause(.01)
                         