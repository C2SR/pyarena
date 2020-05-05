from ..core import localization
import numpy as np

class LandmarkEKF(localization.Localization):
    def __init__(self, **kwargs):
        if 'map' not in kwargs:
            raise KeyError("[Localization/LandmarkEKF] Must specify a map")

        map = kwargs['map']
        Sigma0 = kwargs['Sigma0'] if 'Sigma0' in kwargs else 1.*np.eye(3)
        motion_noise = kwargs['motion_noise'] if 'motion_noise' in kwargs else np.array([.05,.01])
        measurement_noise = kwargs['measurement_noise'] if 'measurement_noise' in kwargs else np.array([.05,.05])

        # Storing parameters
        self.map = map
        # Filter variables
        self.Sigma = Sigma0
        self.R = np.diag(motion_noise)
        self.Q = np.diag(measurement_noise)        
        
        kwargsLocalization = {'x_dimension': 3}
        kwargs.update(kwargsLocalization)
        super().__init__(**kwargs)

    """
    Extended Kalman Filter routine
    """
    def run(self, dt, u, measurements=None):
        # Loading from previous iterations
        x_est = self.x_est
        Sigma = self.Sigma

        c, s = np.cos(x_est[2,0]), np.sin(x_est[2,0])
        # Computing matrices
        G = np.array([[1.,0.,-dt*u[0,0]*s],[0.,1.,dt*u[0,0]*c],[0.,0.,1.]])
        V = np.array([[dt*c,0.],[dt*s,0.],[0., dt]])
        # Prediction   
        x_est = np.array([x_est[0,0] + dt*u[0,0]*c,
                          x_est[1,0] + dt*u[0,0]*s,
                          x_est[2,0] + dt*u[1,0]]).reshape(3,1)
        x_est[2,0] = (x_est[2,0] + np.pi) % (2 * np.pi) - np.pi 
        Sigma =  G@Sigma@G.T + V@self.R@V.T

        # Update
        if measurements is not None and len(measurements['id']) > 0:
            R = np.array([[c, -s],[s, c]]) 
            dR = np.array([[-s, -c],[c, -s]])
            for idx, id in enumerate(measurements['id']):
                # storing measurement and map landmark in local variables
                zi = (measurements['coordinate'][:,idx]).reshape(2,1)
                mi = (self.map.landmarks['coordinate'][:,id]).reshape(2,1)   

                H = np.hstack([-R.T, dR.T@(mi - x_est[0:2])])
                V = R.T
                hi = R.T@(mi - x_est[0:2]) 

                K = Sigma @ H.T @ np.linalg.inv(H@Sigma@H.T + V@self.Q@V.T)
                x_est = x_est + K@(zi - hi)      
                Sigma = (np.eye(3) - K@H)@Sigma

        # Storing for next iterations
        self.x_est = np.copy(x_est)
        self.Sigma = np.copy(Sigma)
        
        return x_est, Sigma

