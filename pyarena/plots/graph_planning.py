import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from . import drawings  

class GraphPlanning:
    def __init__(self, **kwargs):
         # Checking for missing parameters
        if 'world' not in kwargs:
            raise KeyError("[Plot/GraphPlanning] Must specify a GraphWorld")

        world = kwargs['world']
        
        # Storing parameters
        self.world = world
        self.num_nodes =  len(world.graph.nodes)

        # Artists
        self.fig, self.ax = plt.subplots()
        self.active_nodes = self.ax.plot([],[], marker='o',
                                                markerfacecolor='b',
                                                markeredgecolor='k',
                                                linestyle='None')[0]
        self.active_edges = []
        # Plotting the first frame
        self.first_frame()
        
    """
    Create the first plot that contains the known position of the landmarks in the world
    """
    def first_frame(self):  
        # First frame: world
        self.ax.plot([node.state[0,0] for node in self.world.graph.nodes],
                     [node.state[1,0] for node in self.world.graph.nodes],
                     marker='o',
                     markerfacecolor='None',
                     markeredgecolor='k',
                     linestyle='None')

        for edge in self.world.graph.edges:
            node_init = self.world.graph.nodes[edge.init_node_id]
            node_end = self.world.graph.nodes[edge.end_node_id]
            self.ax.plot([node_init.state[0,0], node_end.state[0,0]],
                         [node_init.state[1,0], node_end.state[1,0]],
                         linestyle='--',
                         linewidth='.5', 
                         color='k')


        plt.grid(True)
        plt.show(0)
        plt.pause(.01)

    """
    Update the active edges and nodes in the graph
    """
    def update(self, active_edges_id):
        # Cleaning edges that were active
        for active_edge in self.active_edges:
            active_edge.remove()
        self.active_edges = []

        # Highlighting active edges
        is_active_node = np.zeros(self.num_nodes, dtype='Bool')
        for edge_id in active_edges_id:
            edge = self.world.graph.edges[edge_id]
            is_active_node[edge.init_node_id] = is_active_node[edge.end_node_id] = True
            node_init = self.world.graph.nodes[edge.init_node_id]
            node_end = self.world.graph.nodes[edge.end_node_id]
            active_edge = self.ax.plot([node_init.state[0,0], node_end.state[0,0]],
                                        [node_init.state[1,0], node_end.state[1,0]],
                                        linestyle='-',
                                        linewidth='1.', 
                                        color='b')[0]
            self.active_edges.append(active_edge)       

        # Highlighting active nodes
        active_nodes_id = np.where(is_active_node==True)[0]
        self.active_nodes.set_xdata([self.world.graph.nodes[id].state[0,0] for id in active_nodes_id])
        self.active_nodes.set_ydata([self.world.graph.nodes[id].state[1,0] for id in active_nodes_id])

        plt.pause(5.0)