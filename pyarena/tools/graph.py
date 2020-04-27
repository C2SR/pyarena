import numpy as np

class Edge:
    def __init__(self, id, id_node_init, id_node_end, weight):
        self.id = id
        self.id_node_init = id_node_init
        self.id_node_end = id_node_end 
        self.weight = weight       

class Node():
    def __init__(self, mid, state, reachable=True, edges=None):
        self.id = mid
        self.state = state
        self.reachable = reachable
        self.edges = edges

    def update_edges(self, edge):
        if self.edges is None:
            self.edges = np.array([edge])
        else:
            self.edges = np.stack([self.edges, edge])

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def insert_node(self, new_node):
        self.nodes.append(new_node)
        print(new_node.id)

    def insert_edge(self, new_edge):
        self.edges.append(new_edge)
        

step = 1
x = np.arange(0,3,step)
y = np.arange(0,3,step)
XX, YY = np.meshgrid(x,y)
states = np.vstack([XX.flatten(), YY.flatten()])
mgraph = Graph()

id_edge = 0
for id_node, state in enumerate(states.T):
    # Creating node
    node = Node(mid=id_node, state=state.reshape([2,1]))
    mgraph.insert_node(Node)
    # Creating edges
    diff = states - state.reshape([2,1])
    distances = np.sqrt(np.sum(diff**2, axis=0))
    for id_neighbor, distance in enumerate(distances):
        if 0 < distance <= step:
            edge = Edge(id=id_edge, id_node_init=id_node, id_node_end=id_neighbor, weight=distance)
            mgraph.insert_edge(edge)
            id_edge += 1

print('number of nodes', len(mgraph.nodes))
print('number of edges:', id_edge)
print('number of edges:', len(mgraph.edges))