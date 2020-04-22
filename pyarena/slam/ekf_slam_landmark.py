from ..core import slam
import numpy as np

class EKFSLAMLandmark(slam.SLAM):
    def __init__(self, **kwargs):
        if 'nb_landmarks' not in kwargs:
            raise KeyError("[SLAM/LandmarkEKF] Must specify number of landmarks")

        nb_landmarks = kwargs['nb_landmarks']
        Sigma0 = kwargs['Sigma0'] if 'Sigma0' in kwargs else 1.*np.eye(3)
        motion_noise = kwargs['motion_noise'] if 'motion_noise' in kwargs else np.array([.05,.01])
        measurement_noise = kwargs['measurement_noise'] if 'measurement_noise' in kwargs else np.array([.05,.05])

        # Storing parameters
        self.nb_landmarks = nb_landmarks
        self.has_initialized_landmark = np.zeros(nb_landmarks,dtype=bool)
        # Filter variables
        self.Sigma = 100*np.eye(3+2*nb_landmarks)
        self.Sigma[0:3,0:3] = Sigma0
        self.R = np.diag(motion_noise)
        self.Q = np.diag(measurement_noise)
        self.F = np.vstack([np.eye(3), np.zeros([2*nb_landmarks,3])])        

        # Initializing parent class
        kwargsSLAM = {'x_dimension': 3+2*nb_landmarks}
        kwargs.update(kwargsSLAM)
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
        G = np.eye(2*self.nb_landmarks+3)
        V = np.zeros([2*self.nb_landmarks+3, 2])        
        G[0:3,0:3] = np.array([[1.,0.,-dt*u[0,0]*s],[0.,1.,dt*u[0,0]*c],[0.,0.,1.]])
        V[0:3,:] = np.array([[dt*c,0.],[dt*s,0.],[0., dt]])

        # Prediction   
        x_est = x_est + self.F@np.array([dt*u[0,0]*c,
                                        dt*u[0,0]*s,
                                        dt*u[1,0]]).reshape(3,1)
        x_est[2,0] = (x_est[2,0] + np.pi) % (2 * np.pi) - np.pi 
        Sigma =  G@Sigma@G.T + V@self.R@V.T

        # Update
        if measurements is not None and len(measurements['id']) > 0:
            R = np.array([[c, -s],[s, c]]) 
            dR = np.array([[-s, -c],[c, -s]])
            for idx, id in enumerate(measurements['id']):
                # storing measurement and map landmark in local variables
                zi = (measurements['coordinate'][:,idx]).reshape(2,1)
                if self.has_initialized_landmark[id] == False:
                    print('Initializing landmark #', id)
                    x_est[3+2*id:3+2*id+2] = x_est[0:2] + R@zi
                    self.has_initialized_landmark[id]= True
                mi = x_est[3+2*id:3+2*id+2]   
                H = np.zeros([2,3+2*self.nb_landmarks])
                H[:,0:3] = np.hstack([-R.T, dR.T@(mi - x_est[0:2])])
                H[:,3+2*id:3+2*id+2] = R.T                
                hi = R.T@(mi - x_est[0:2]) 

                K = Sigma @ H.T @ np.linalg.inv(H@Sigma@H.T + self.Q)
                x_est = x_est + K@(zi - hi)      
                Sigma = (np.eye(3+2*self.nb_landmarks) - K@H)@Sigma

        # Storing for next iterations
        self.x_est = np.copy(x_est)
        self.Sigma = np.copy(Sigma)
        
        return x_est, Sigma

