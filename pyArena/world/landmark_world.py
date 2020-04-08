import numpy as np

class LandmarkWorld:
    def __init__(self, **kwargs):
        if 'width' not in kwargs:
            raise KeyError("[World/LandmarkWorld] Please specify the width of the map")
        if 'height' not in kwargs:
            raise KeyError("[World/LandmarkWorld] Please specify the height of the map")

        # receiving parameters
        width = kwargs['width']
        height = kwargs['height']    
        nb_landmarks = kwargs['nb_landmarks'] if 'nb_landmarks' in kwargs else 10
        offset = kwargs['offset'] if 'offset' in kwargs else np.array([.0,.0])

        # storing
        self.width = width
        self.height = height
        self.nb_landmarks = nb_landmarks
        self.offset = offset.reshape(2,1)

        # generating landmarks
        self.landmarks = {}
        self.landmarks['id'] = np.linspace(0,self.nb_landmarks-1,self.nb_landmarks, dtype=int)
        self.landmarks['coordinate'] = np.diag([self.width, self.height])@np.random.rand(2,self.nb_landmarks) - self.offset.reshape(2,1)