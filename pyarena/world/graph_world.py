import numpy as np
from ..tools import graph

class GraphWorld:
    def __init__(self, **kwargs):
        if 'size' not in kwargs:
            raise KeyError("[World/GraphWorld] Please specify the size of the map")
        
        size = kwargs['size'].reshape(2,)
        origin = kwargs['origin'] if 'origin' in kwargs else np.zeros(2,)
        step = kwargs['step'] if 'step' in kwargs else 1.
        max_neighbor_distance = kwargs['max_neighbor_distance'] if 'max_neighbor_distance' in kwargs else step


        # Create the graph
        x = np.arange(origin[0], step*(size[0]-origin[0]), step)
        y = np.arange(origin[1], step*(size[1]-origin[1]), step)        
        XX, YY = np.meshgrid(x,y)
        states = np.vstack([XX.flatten(), YY.flatten()])
        mgraph = graph.Graph()

        id_edge = 0
        for id_node, state in enumerate(states.T):
            diff = states - state.reshape([2,1])
            distances = np.sqrt(np.sum(diff**2, axis=0))
            edges = []  # list of edges that leads to neighboring nodes
            for id_neighbor, distance in enumerate(distances):
                if 0 < distance <= max_neighbor_distance:
                    # Creating and inserting an edge in the grapg
                    edge = graph.Edge(id=id_edge, id_node_init=id_node, id_node_end=id_neighbor, weight=distance)
                    mgraph.insert_edge(edge)
                    edges.append(id_edge)
                    id_edge += 1
            # Creating and inserting a node in the graph
            node = graph.Node(id=id_node, state=state.reshape([2,1]), edges=edges)
            mgraph.insert_node(node)
        
        # Storing
        self.graph = mgraph