from abc import ABC, abstractmethod

class StaticController(ABC):

    def __init__(self):
        if type(self) is StaticController:
            raise Exception("Cannot create an instance of abstract class BaseController")

    @abstractmethod
    def computeInput(self,t,x):
        pass

class InlineStaticController(StaticController):

    def __init__(self, **kwargs):

        super().__init__()

        if 'controlLaw' not in kwargs:
            raise KeyError("Must specify control law")

        self.controlLaw = kwargs['controlLaw']
    # END of __init__

    def computeInput(self,t,x):
        return self.controlLaw(t,x)
