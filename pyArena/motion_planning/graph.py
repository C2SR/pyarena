import numpy as np

class Vertice:
    def __init__(self, id):
        self.id = id
        self.children = []

    """ Add the id of a child to the children list 
    """    
    def add_child(self, child_id):
        self.children.append(child_id)

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
            self.vertices.append(Vertice(id))
            current = np.array([X[id], Y[id]])
            for neighbor in neighborhood:
                child = current + neighbor
                if ((0<=child[0]<mx) and (0<=child[1]<my)) :  
                    self.vertices[id].add_child(child[0] + child[1]*mx)
    
    """ Print the graph to terminal 
    """    
    def print(self):
        print("[id] [children]")
        for vertice in self.vertices:
            vertice.print()

# Graph
mx = 2
my = 2
neighborhood = np.array([[0,0],[-1,0],[0,-1],[1,0],[0,1]])

graph = Graph(mx,my, neighborhood)
graph.print()    
                    