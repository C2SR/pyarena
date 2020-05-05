import numpy as np
from ..core import planning
from ..tools import graph

"Informative Path Planning using Branch and Bound"
class IPPBnB(planning.DiscretePlanner):
    def __init__(self,**kwargs):
         # Checking for missing parameters
        if 'world' not in kwargs:
            raise KeyError("[Planning/IPPBnB] Must specify a GraphWorld")
        world = kwargs['world']
        length = kwargs['kernel_length'] if 'kernel_length' in kwargs else 1.
        # Storing
        self.world = world
        self.num_nodes =  len(world.graph.nodes)
        self.length = length

        # Covariance matrix for all possible locations
        self.K_XX = np.zeros([self.num_nodes, self.num_nodes])
        for i in range(0, self.num_nodes):
            state_i = world.graph.nodes[i].state
            for j in range(i, self.num_nodes):
                state_j = world.graph.nodes[j].state
                self.K_XX[i,j] = self.K_XX[j,i] = np.exp( -np.linalg.norm(state_i-state_j)/(2*length*length) )

        super().__init__(**kwargs) 

    """
    Motion planning default function for computing a plan
    """        
    def run(self, x_start, x_goal, **kwargs):
        # Parameters
        # Checking for missing parameters
        if 'budget' not in kwargs:
            raise KeyError("[Planning/IPPBnB] Must specify a budget")        
        budget = kwargs['budget']
        path = kwargs['path'] if 'path' in kwargs else []
        horizon = kwargs['horizon'] if 'horizon' in kwargs else np.inf
        
        # Initialize parameters
        self.optimal_objective = 0
        self.optimal_path = path
        # Search for the two nodes that are closest to start and goal states
        start_node_id, _ = self.world.graph.find_closest_node(x_start)
        goal_node_id, _ = self.world.graph.find_closest_node(x_goal)        
        # IPP-BnB algorithm
        self.ipp_bnb(start_node_id, goal_node_id, budget, path, horizon)

        return self.optimal_path, self.optimal_objective  

    """
    Informative Path Planning using Branch and Bound: Recursive function that evaluate
    the upper bound of the current candidate solution and expands the tree. If the path
    leads to the goal node, the it is considered a leaf.
    """
    def ipp_bnb(self, start_node_id, goal_node_id, budget, path, horizon):
        is_leaf = True
        new_path = path
        if horizon > 0:
            for edge_id in self.world.graph.nodes[start_node_id].edges:
                # Updating candidate path and remaining budget
                new_path = path + [edge_id]
                new_node_id = self.world.graph.edges[edge_id].end_node_id 
                new_budget = budget - self.world.graph.edges[edge_id].weight
                # Check if budget allows reaching the goal
                if self.euclidean_cost(new_node_id, goal_node_id) <= new_budget:
                    if new_node_id != goal_node_id:
                        is_leaf = False
                        # Prunning if upper-bound is lower than current optimal
                        if self.ubound(new_node_id, goal_node_id, new_budget, new_path) > self.optimal_objective:
                            self.ipp_bnb(new_node_id, goal_node_id, new_budget, new_path, horizon-1)
                    else:
                        self.check_optimality(new_path)

        if is_leaf and horizon == 0:
            self.check_optimality(new_path)

    # check if candidate path is better than current best soluroscoretion
    def check_optimality(self, path): 
            # Find the nodes in the candidate path and compute reward
            is_node_in_path = np.zeros(self.num_nodes, dtype='Bool')
            for edge_id in path:
                edge = self.world.graph.edges[edge_id]
                is_node_in_path[edge.init_node_id] = True
                is_node_in_path[edge.end_node_id] = True 
            nodes_id = np.where(is_node_in_path==True)[0]
            objective = self.evaluate_objective(nodes_id)
            # Update solutinon if candidate is better than current best solution
            if (objective > self.optimal_objective):
                self.optimal_path = path
                self.optimal_objective = objective    

    """
    Computes an upper bound for a path
    """
    def ubound(self, start_node_id, goal_node_id, budget, path):
        # Compute all nodes that can be reached by the current (path, budget)
        nodes_reachable = np.zeros(self.num_nodes, dtype='Bool')
        for edge in self.world.graph.edges:
            cost_to_goal = self.euclidean_cost(start_node_id, edge.init_node_id) + \
                                edge.weight + \
                                self.euclidean_cost(edge.end_node_id, goal_node_id)
            if cost_to_goal <= budget or edge.id in path:
                nodes_reachable[edge.init_node_id] = True
                nodes_reachable[edge.end_node_id] = True 
        candidate_nodes_id = np.where(nodes_reachable==True)[0]
        # Compute and retunr the bound
        return self.evaluate_objective(candidate_nodes_id)

    """
    Evaluate the average variance reduction (AVR) objective function for a candidate path
    """
    def evaluate_objective(self, candidate_nodes_id):
        # Covariance matrix for candidates
        num_candidates = len(candidate_nodes_id)
        if (num_candidates <= 0):
            return 0
        K_AA = np.zeros([num_candidates, num_candidates])
        K_AX = np.zeros([num_candidates, self.num_nodes])
        for i in range(0, num_candidates):
            node_id_i = candidate_nodes_id[i]
            state_i = self.world.graph.nodes[node_id_i].state
            # Bulding K_AA
            for j in range(i, num_candidates):
                node_id_j = candidate_nodes_id[j]
                state_j = self.world.graph.nodes[node_id_j].state
                K_AA[i,j] = K_AA[j,i] = np.exp( -np.linalg.norm(state_i-state_j)/(2*self.length*self.length) )
        
            # Bulding K_AX
            for j in range(0, self.num_nodes):
                node_id_j = j
                state_j = self.world.graph.nodes[node_id_j].state
                K_AX[i,j] = np.exp( -np.linalg.norm(state_i-state_j)/(2*self.length*self.length) )
            
        # Compute posterior covariance
        Sigma_post = self.K_XX - K_AX.T @ np.linalg.inv(K_AA + .001*np.eye(num_candidates)) @ K_AX

        # Average variance reduction (AVG) objective function 
        return 1./self.num_nodes * (np.sum(np.diag(self.K_XX)) - np.sum(np.diag(Sigma_post))) 

    """
    Euclidean cost to reach end_node_id from init_node_id
    """
    def euclidean_cost(self, init_node_id, end_node_id):
        return np.linalg.norm(self.world.graph.nodes[end_node_id].state - \
                              self.world.graph.nodes[init_node_id].state, 
                              1)