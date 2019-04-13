# TODOC Summary: explain...

# create a system class
#class DynamicalSystem:
#    def __init__(self, **kwargs):
#        self.nx = kwargs['nx']
#        self.nu = kwargs['nu']
#        self.stateEquation = kwargs['stateEquation']
#        self.isMeasurementEquation = False
#        self.initialCondition = kwargs['initialCondition']
#        self.controller = kwargs['controller']

from abc import ABC, abstractmethod

class DynamicSystem(ABC):

    def __init__(self):
        if type(self) is DynamicSystem:
            raise Exception("Cannot create an instance of Abstract class - DynamicSystem")

    @abstractmethod
    def stateEquation(self,t,x,u):
        pass

    @abstractmethod
    def outputEquation(self,t,x,u):
        pass
## END of class DynamicSystem


class InlineDynamicSystem(DynamicSystem):

    def __init__(self, **kwargs):

        super().__init__()

        if 'nx' not in kwargs:
            raise KeyError("Must specify number of states nx")

        self.nx = kwargs['nx']

        if 'nu' not in kwargs:
            raise KeyError("Must specify number of inputs nu")

        self.nu = kwargs['nu']

        if 'inlineStateEquation' not in kwargs:
            raise KeyError("Must specify the inline state equation")

        self.inlineStateEquation = kwargs['inlineStateEquation']

        if 'ny' not in kwargs:

            self.ny = self.nx

            self.isOutputEquation = False

        else:

            self.ny = kwargs['ny']

            self.isOutputEquation = True

            if 'inlineOutputEquation' not in kwargs:
                raise KeyError("Must specify the inline output equation")

            self.inlineOutputEquation = kwargs['inlineOutputEquation']

        if 'initialCondition' not in kwargs:
            raise KeyError("Must specify the initial condition")

        self.initialCondition = kwargs['initialCondition']

    # END of __init__()

    def stateEquation(self,t,x,u):
        return self.inlineStateEquation(t,x,u)
    # END of stateEquation()

    def outputEquation(self,t,x,u):
        if self.isOutputEquation is True:
            y = self.inlineOutputEquation(t,x,u)
        else:
            y = x
        return y
    # END of outputEquation()
## END of InlineDynamicSystem()

class InlineControlSystem(InlineDynamicSystem):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if 'controller' not in kwargs:
            raise KeyError("Must specify the controller")

        self.controller = kwargs['controller']
    # END of __init__()
## END of InlineControlSystem()
