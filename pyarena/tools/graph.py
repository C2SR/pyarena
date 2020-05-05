import numpy as np

class Node:
    def __init__(self, id, state, reachable=True, edges=None):
        self.id = id
        self.state = state
        self.reachable = reachable
        self.edges = edges

class Edge:
    def __init__(self, id, id_node_init, id_node_end, weight=1.):
        self.id = id
        self.init_node_id = id_node_init
        self.end_node_id = id_node_end 
        self.weight = weight       

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def insert_node(self, new_node):
        self.nodes.append(new_node)

    def insert_edge(self, new_edge):
        self.edges.append(new_edge)
        
    def find_closest_node(self, state):
        # Search for node that is closest to state
        node_id  = -1
        min_dist = np.inf
        for node in self.nodes:
            dist = np.linalg.norm(node.state.reshape(2,1) - state.reshape(2,1)) 
            if (dist < min_dist):
                node_id = node.id
                min_dist = dist

        return node_id, min_dist