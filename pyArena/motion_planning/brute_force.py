import numpy as np
from graph import Graph


class BruteForce:
    def __init__(self, graph):
        # Graph
        self.graph = graph
        self.hasTree = False
    """ Builds the tree to find feasible paths from start to goal positions. 
    In the end, the variable self.path stores feasible paths from start to goal
    """
    def build_tree(self, id_start, id_goal, depth):
        # Saving parameters
        self.path = []
        self.id_goal = id_goal
        self.depth = depth
        # Check for feasible paths to the goal
        self.explore([id_start])
        self.hasTree = True

    """ Explore the graph considering the depth (budget) and goal constraints
    """
    def explore(self, path):
        id_current = path[-1]
        neighbors = self.graph.vertices[id_current].children
        # Check if the goal has been reached
        if (id_current == self.id_goal):
            self.path.append(np.array(path))
        # Explore if there is budget to explore
        if (len(path) < self.depth+1):
            for id in neighbors: 
                self.explore(path +[id])

    """ Compute optimal path for Informative Path Planning problem 
    using the average variance reduction (AVR) cost function and minimum 
    budget
    """
    def average_variance_reduction(self, id_start, id_goal, budget, l =1):
        min_energy = budget
        max_reward = -np.inf
        optimal_path = []

        if not self.hasTree:
            self.build_tree(id_start, id_goal, budget)
        
        # 
        n = len(self.graph.vertices)
        K_XX = np.zeros([n,n])
        for i in range(0,n):
            for j in range(i,n):
                xi = self.graph.vertices[i].position
                xj = self.graph.vertices[j].position
                K_XX[i,j] = K_XX[j,i] = np.exp(-np.linalg.norm(xi-xj)/(2*l*l) )
        
        for full_path in self.path:
            # Given a path, we want to compute the AVR reward and energy cost
            energy = len(full_path)
            path = np.unique(full_path)  # @NOTE this order the elements in crescent 
            m = len(path)
            K_XA = np.zeros([n,m])
            for i in range(0,n):
                for j in range(0,m):
                    xi = self.graph.vertices[i].position
                    xj = self.graph.vertices[path[j]].position
                    K_XA[i,j]  = np.exp(-np.linalg.norm(xi-xj)/(2*l*l) )

            K_AA = np.zeros([m,m])
            for i in range(0,m):
                for j in range(0,m):
                    xi = self.graph.vertices[path[i]].position
                    xj = self.graph.vertices[path[j]].position
                    K_AA[i,j]  = K_AA[j,i] = np.exp(-np.linalg.norm(xi-xj)/(2*l*l) )

            Sigma_post = K_XX - K_XA@np.linalg.inv(K_AA)@(K_XA.T)
            f_arv = 1/m * (np.sum(np.diag(K_XX))) - np.sum(np.diag(Sigma_post))

            if ( (f_arv == max_reward and energy < min_energy) or f_arv > max_reward):
                optimal_path = full_path
                min_energy = energy
                max_reward = f_arv 

        return optimal_path, max_reward, min_energy
# Graph
mx = 3
my = 3
neighborhood = np.array([[0,0],[-1,0],[0,-1],[1,0],[0,1]])

brute_force = BruteForce(Graph(mx,my, neighborhood))
optimal_path, max_reward, min_energy = brute_force.average_variance_reduction(
        id_start=0,
        id_goal=8,
        budget=8)

print("-----------------------------------------------------------------------")
print("Path", optimal_path, " reward: ", max_reward, "cost: ", min_energy)
print("-----------------------------------------------------------------------")