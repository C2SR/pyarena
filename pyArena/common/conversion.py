import numpy as np

def Euler2Quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr  
    w = cy * cp * cr + sy * sp * sr

    return x, y, z, w

def Yaw2Quaternion(yaw):
    x = .0
    y = .0
    z = np.sin(yaw * 0.5)   
    w = np.cos(yaw * 0.5) 

    return x, y, z, w