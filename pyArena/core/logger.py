import numpy as np
from abc import ABC, abstractmethod

class AbstractLog(ABC):

    def __init__(self):
        if type(self) is AbstractLog:
            raise Exception("Cannot create an instance of Abstract class - AbstractLog")

    @abstractmethod
    def updateLog(self, t, x, u, *args):
        pass

class StateVectorLog(AbstractLog):

    def __init__(self, **kwargs):

        self.name = 'stateTrajectory'

        self.size = kwargs['nx']

        self.data = list()

    def updateLog(self, t, x, u, *args):

        self.data = np.append(self.data, x)

class InputVectorLog(AbstractLog):

    def __init__(self, **kwargs):

        self.name = 'inputTrajectory'

        self.size = kwargs['nu']

        self.data = list()

    def updateLog(self, t, x, u, *args):

        self.data = np.append(self.data, u)

class TimeLog(AbstractLog):

    def __init__(self):

        self.name = 'time'

        self.size = 1

        self.data = list()

    def updateLog(self, t, x, u, *args):

        self.data = np.append(self.data, t)

class InlineVectorLog(AbstractLog):

    def __init__(self, **kwargs):

        super().__init__()

        self.name = kwargs['name']

        self.function = kwargs['logFunction']

        self.size = kwargs['size']

        self.data = list()

    def updateLog(self, t, x, u, *args):

        self.data = np.append(self.data, self.function(t, x, u, *args))



# class pyLog:
#
#     def __init__(self, **kwargs):
#
#         self.stateTrajectory = np.zeros(kwargs['nx'])
#
#         self.inputTrajectory = np.zeros(kwargs['nu'])
#
#         self.time = np.zeros(1)
#
#         if 'numMeasurements' in kwargs:
#             self.
#
#         self.__kwargs = kwargs
#         # TODO: Need additional code for extra logs
#     ## END of __init__()
#
#     def updateLog(self, t, x, u):
#
#         if t == 0:
#             self.stateTrajectory = x
#
#             self.inputTrajectory = u
#
#             self.time = t
#         else:
#             self.stateTrajectory = np.append(self.stateTrajectory, x, axis=0)
#
#             self.inputTrajectory = np.append(self.inputTrajectory, u, axis=0)
#
#             self.time = np.append(self.time, t)
#     ## END of updateLog()
#
#     def reshapeLog(self):
#
#         self.stateTrajectory = self.stateTrajectory.reshape(-1, self.__kwargs['nx'])
#
#         self.inputTrajectory = self.inputTrajectory.reshape(-1, self.__kwargs['nu'])
#     ## END of reshapeLog()
#
# ## END of class pyLog
