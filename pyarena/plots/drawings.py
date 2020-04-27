import numpy as np
import scipy

class VehicleDrawing:
    def __init__(self, **kwargs):
        vehicle_type = kwargs['vehicle'] if 'vehicle' in kwargs else 'unicycle'        
        scale = kwargs['scale'] if 'scale' in kwargs else 1.0    
        
        # Vehicle countour
        if vehicle_type.lower() == 'unicycle':
            self.vehicle_drawing = scale*np.array([[-1,2,-1,-1],[-1,0,1,-1]])
        else:
            raise KeyError("[Plot/Vehicle] Unknown vehicle type. Options: vehicle='unicycle'")


    def update(self, x):
        # Pose of the vehicle
        pos = x[0:2,0].reshape([2,1])
        theta = x[2,0]

        # Transforming vehicle contour
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [ np.sin(theta),  np.cos(theta)]])
        vehicle = R@self.vehicle_drawing + pos

        return vehicle

class Covariance2DDrawing:
    def __init__(self, **kwargs):
        # Uncertainty ellipsoid countour (95% confidence)
        nb_pts = kwargs['nb_pts'] if 'nb_pts' in kwargs else 50
        
        theta = np.linspace(0, 2*np.pi, nb_pts)
        self.ellipse_drawing = np.vstack([np.cos(theta), np.sin(theta)])

    def update(self, Cov, xc = np.array([0,0])):
        # Transforming ellipse contour
        """
        eigval, eigvec  = np.linalg.eig(Cov[0:2,0:2])
        sorted_index = np.argsort(eigval)[::-1] # sorting eigenvalues' index  by descending order of the eigenvalues
        eigval, eigvec = eigval[sorted_index], eigvec[:,sorted_index]
        alpha = np.arctan2(eigvec[1,0], eigvec[0,0])
   
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])

        ellipse = R@np.diag(np.sqrt(5.991*eigval))@self.ellipse_drawing + x[0:2,0].reshape(2,1)
        """
        ellipse = scipy.linalg.sqrtm(5.991*Cov[0:2,0:2])@self.ellipse_drawing + xc[0:2,0].reshape(2,1)
        return ellipse