from ..core import map

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import time
"""
Standard gaussian process mapping for 2D fields
"""
class Occupancy(map.StaticMap):
    def __init__(self, **kwargs):
        if 'width' not in kwargs:
            raise KeyError("[Map] Specify the map width")

        if 'height' not in kwargs:
            raise KeyError("[Map] Specify the height")

        width = kwargs['width'] 
        height = kwargs['height']    
        x0 = kwargs['x0'] if 'x0' in kwargs else 0.
        y0 = kwargs['y0'] if 'y0' in kwargs else 0.
        self.resolution = kwargs['resolution'] if 'resolution' in kwargs else 1.


        # Create a grid map to vizualisation
        self.origin = np.array([x0, y0])
        self.end = self.origin + np.array([width, height])
        self.grid_size = np.array([height/self.resolution, width/self.resolution]).astype(int)
        self.grid = np.zeros(self.grid_size)
        self.grid_hits = np.zeros(self.grid_size)
   
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

    """
    Update rule after receiving a measurement 
    """
    def compute_map(self, t, x, measurement):
        # Finding the cell that needs to be updated
        meas_map_coordinate = x - self.origin
        meas_grid_coordinate = np.floor(meas_map_coordinate/self.resolution).astype(int)
        # Updating
        current_cell_value = self.grid[meas_grid_coordinate[1], meas_grid_coordinate[0]] 
        current_hits =  self.grid_hits[meas_grid_coordinate[1], meas_grid_coordinate[0]]
        
        new_cell_value = (current_cell_value*current_hits + measurement)/(current_hits+1)
        new_hits = current_hits + 1    

        self.grid[meas_grid_coordinate[1], meas_grid_coordinate[0]] = new_cell_value
        self.grid_hits[meas_grid_coordinate[1], meas_grid_coordinate[0]] = new_hits

    
    def get_map(self):
        self.ax_map.imshow(self.grid[::-1,:], vmin = 0, vmax=22, cmap='jet')
        x_tick_loc=np.arange(0,self.grid_size[1]+self.resolution,10)
        y_tick_loc=np.arange(0,self.grid_size[0]+self.resolution,10)
        x_tick_label = x_tick_loc*self.resolution
        y_tick_label = y_tick_loc[::-1]*self.resolution  
        plt.xticks(x_tick_loc, x_tick_label)
        plt.yticks(y_tick_loc, y_tick_label)
        plt.draw()
        plt.pause(0.01)
