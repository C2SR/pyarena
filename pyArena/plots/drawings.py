import numpy as np

class VehicleDrawing:
    def __init__(self, **kwargs):
        vehicle_type = kwargs['vehicle'] if 'vehicle' in kwargs else 'unicycle'        
        scale = kwargs['scale'] if 'scale' in kwargs else 1.0    
        
        if vehicle_type.lower() == 'unicycle':
            self.pattern = scale*np.array([[-1,2,-1,-1],[-1,0,1,-1]])
        else:
            raise KeyError("[Plot/Vehicle] Unknown vehicle type. Options: vehicle='unicycle'")
        
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