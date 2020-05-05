# Python libraries
import numpy as np
from abc import ABC, abstractmethod

## Discrete Planner (abstract) class ##
class DiscretePlanner(ABC):
    def __init__(self, **kwargs):          
        # Retrieving parameters
        x_start = kwargs['x_start'] if 'x_start' in kwargs else None
        x_goal = kwargs['x_goal'] if 'x_goal' in kwargs else None
        budget = kwargs['budget'] if 'budget' in kwargs else np.inf

        # Initializing parameters
        self.x_start = x_start
        self.x_goal  = x_goal
        self.budget  = budget

    "Set basic planner parameters"
    def set_parameters(self, x_start, x_goal, budget=np.inf):
        self.x_start = x_start
        self.x_goal  = x_goal
        self.budget  = budget

    """
    Discrete planner routine that computes waypoints
    to reach the goal state
    """
    @abstractmethod
    def run(self, x_start, x_goal, **kwargs):
        pass

