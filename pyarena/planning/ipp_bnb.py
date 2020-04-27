from ..core import planning

"Informati Path Planning using Branch and Bound"
class IPPBnB(planning.DiscretePlanner):
    def __init__(self,**kwargs):
        super().__init__(**kwargs) 
        
    def run(self, x_start=None, x_goal=None, budget=None):
        pass