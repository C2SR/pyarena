import numpy as np

class VehicleDrawing:
    def __init__(self, **kwargs):
        vehicle_type = kwargs['vehicle'] if 'vehicle' in kwargs else 'unicycle'        
        scale = kwargs['scale'] if 'scale' in kwargs else 1.0    
        
        # Vehicle countour
        if vehicle_type.lower() == 'unicycle':
            self.vehicle_drawing = scale*np.array([[-1,2,-1,-1],[-1,0,1,-1]])
        else:
            raise KeyError("[Plot/Vehicle] Unknown vehicle type. Options: vehicle='unicycle'")

        # Uncertainty ellipsoid countour (95% confidence)
        theta = np.linspace(0, 2*np.pi, 50)
        self.ellipse_drawing = np.vstack([np.cos(theta), np.sin(theta)])

    def update(self, x, Cov=np.zeros([3,3])):
        # Pose of the vehicle
        pos = x[0:2,0].reshape([2,1])
        theta = x[2,0]

        # Transforming vehicle contour
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [ np.sin(theta),  np.cos(theta)]])
        vehicle = R@self.vehicle_drawing + pos
        ellipse = None

        # Transforming ellipse contour
        eigval, eigvec  = np.linalg.eig(Cov[0:2,0:2])
        sorted_index = np.argsort(eigval)[::-1] # sorting eigenvalues' index  by descending order of the eigenvalues
        eigval, eigvec = eigval[sorted_index], eigvec[:,sorted_index]
        alpha = np.arctan2(eigvec[1,0], eigvec[0,0])
   
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])

        ellipse = R@np.diag(np.sqrt(5.991*eigval))@self.ellipse_drawing + x[0:2,0].reshape(2,1)        

        return vehicle, ellipse
