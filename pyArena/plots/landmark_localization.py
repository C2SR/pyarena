import numpy as np
import matplotlib.pyplot as plt

class Unicycle:
    def __init__(self, **kwargs):
        scale = kwargs['scale'] if 'scale' in kwargs else 1.0    
            
        self.pattern = scale*np.array([[-1,2,-1,-1],[-1,0,1,-1]])

    def update(self, x, covariance=np.zeros([3,3])):
        # Pose of the vehicle
        pos = x[0:2,0].reshape([2,1])
        theta = x[2,0]

        # Transforming contour
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [ np.sin(theta),  np.cos(theta)]])
        vehicle = R@self.pattern + pos
        ellipse = None

        return vehicle, ellipse

class LandmarkLocalization:
    def __init__(self, **kwargs):
         # Checking for missing parameters
        if 'world' not in kwargs:
            raise KeyError("[Plot/Landmark Localization] Must specify a world")

        self.world = kwargs['world']
        self.sensor = kwargs['sensor'] if 'sensor' in kwargs else None
        self.confidence = kwargs['confidence'] if 'confidence' in kwargs else .95

        # Vehicle drawer       
        kwargsVehicle = {'scale': 0.1}
        self.vehicle = Unicycle(**kwargsVehicle)
        
        
        # Artists
        _, self.ax = plt.subplots()
        self.x_gt_marker = self.ax.plot([],[], linestyle='-', color='k')[0]
        self.x_est_marker = self.ax.plot([],[], linestyle='--', color='g')[0]
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
        self.ax.plot(self.world['coordinate'][0,:],
                     self.world['coordinate'][1,:],
                     marker='s',
                     markerfacecolor='None',
                     markeredgecolor='k',
                     markeredgewidth=2.0,
                     linestyle='None')

        plt.ylim(0, self.world['size'][0])
        plt.xlim(0, self.world['size'][1])
        plt.grid(True)
        plt.show(0)
        plt.pause(.1)


        plt.pause(.01)
        
    """
    Update the pose of the robot in the map
    """
    def update(self, x_ground_truth, x_est, measurements, covariance=np.zeros([3,3])):
        # Computing vehicle countour
        x_gt_countour, _ = self.vehicle.update(x_ground_truth)
        x_est_countour, _ = self.vehicle.update(x_est)        
       
        # Updating vehicle plot
        self.x_gt_marker.set_xdata(x_gt_countour[0,:])
        self.x_gt_marker.set_ydata(x_gt_countour[1,:])
        self.x_est_marker.set_xdata(x_est_countour[0,:])
        self.x_est_marker.set_ydata(x_est_countour[1,:])  
        # Highlighting detected landmarks
        self.detected_landmarks.set_xdata(self.world['coordinate'][0,measurements['id']])
        self.detected_landmarks.set_ydata(self.world['coordinate'][1,measurements['id']])        

        plt.pause(.01)
                         