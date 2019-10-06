import numpy as np

class Vertice:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.edge = []
        self.cost = []
        self.children = []
        
    """ Add the id of a child to the children list 
    """    
    def add_child(self, child_id, edge, cost = 1):
        self.children.append(child_id)
        self.edge.append(edge)
        self.cost.append(cost)

    """ Print the vertice and its children to terminal 
    """    
    def print(self):
        print(self.id,': ', self.children)

class Graph:
    """ Initialization function
    """
    def __init__(self, mx, my, neighborhood):
        self.mx = mx
        self.my = my
        self.neighborhood = neighborhood
        # Vertice list
        self.vertices = []
        X, Y = np.meshgrid(np.arange(0,mx),np.arange(0,my))
        X = X.flatten()
        Y = Y.flatten()

        for id in range(mx * my):
            current = np.array([X[id], Y[id]])
            self.vertices.append(Vertice(id, current))
            for edge in neighborhood:
                child = current + edge
                if ((0<=child[0]<mx) and (0<=child[1]<my)) :  
                    self.vertices[id].add_child(child[0] + child[1]*mx,
                                                edge,
                                                sum(edge))
    
    """ Print the graph to terminal 
    """    
    def print(self):
        print("[id] [children]")
        for vertice in self.vertices:
            vertice.print()

   
                    