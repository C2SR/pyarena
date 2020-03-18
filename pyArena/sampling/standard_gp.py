from ..core import map
from ..algorithms.gaussian_process import GPRegression 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import time
"""
Standard gaussian process mapping for 2D fields
"""
class StandardGP(map.StaticMap):
    def __init__(self, **kwargs):
        if 'width' not in kwargs:
            raise KeyError("[Map] Specify the map width")

        if 'height' not in kwargs:
            raise KeyError("[Map] Specify the height")

        self.width = kwargs['width'] 
        self.height = kwargs['height']    
        self.resolution = kwargs['resolution'] if 'resolution' in kwargs else 1.
        self.grid_size = np.array([self.height/self.resolution, self.width/self.resolution]).astype(int)

        # GP
        self.m_gp = GPRegression()

         # plot
        plt.ion()
        plt.show()
        plt.figure()
        self.ax_map = plt.subplot(111)
        self.ax_map.set(xlabel='x [m]', ylabel='y [m]')            
        norm = plt.cm.colors.Normalize(vmin=0,vmax=22)
        plt.colorbar(plt.cm.ScalarMappable(norm,cmap='jet'),ax=self.ax_map)
  


        # Initializing parent class
        kwargsMap = {'x_dimension': 2}
        kwargs.update(kwargsMap)

        super().__init__(**kwargs)    

    def update_training_input(self, input):
        self.inpTrain = input
        self.numTrain = input.shape[1]
        self.distance_2_input = 100*np.ones(self.numTrain)

        self.m_gp.update_grid(self.height, self.width, self.resolution)

    """
    Update rule after receiving a measurement 
    """
    def compute_map(self, t, x, measurement):
        # Finding the cell that needs to be updated
        distance = np.linalg.norm(x.reshape([2,1])-self.inpTrain,axis=0) 
        flag = distance < .25

        if (np.sum(flag)>0 and self.distance_2_input[flag]>.25):
            self.distance_2_input[flag] = distance[flag]
            pos = np.where(flag)[0]
            print('!!!!! Updating wp #:', pos[0])
            self.m_gp.trainGP(self.inpTrain[:,pos[0]], measurement)
            self.m_gp.update_grid(self.height, self.width, self.resolution)
    def get_map(self):
       
        self.ax_map.imshow(self.m_gp.map[::-1,:], vmin = 0, vmax=22, cmap='jet')
        x_tick_loc=np.arange(0,self.grid_size[1]+self.resolution,10)
        y_tick_loc=np.arange(0,self.grid_size[0]+self.resolution,10)
        x_tick_label = x_tick_loc*self.resolution
        y_tick_label = y_tick_loc[::-1]*self.resolution  
        plt.xticks(x_tick_loc, x_tick_label)
        plt.yticks(y_tick_loc, y_tick_label)
        plt.draw()
        plt.pause(0.01)
        
        