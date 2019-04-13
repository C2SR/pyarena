import numpy as np

class pyLog:

    def __init__(self, **kwargs):

        self.stateTrajectory = np.zeros(kwargs['nx'])

        self.inputTrajectory = np.zeros(kwargs['nu'])

        self.time = np.zeros(1)

        self.__kwargs = kwargs
        # TODO: Need additional code for extra logs
    ## END of __init__()

    def updateLog(self, t, x, u):

        if t == 0:
            self.stateTrajectory = x

            self.inputTrajectory = u

            self.time = t
        else:
            self.stateTrajectory = np.append(self.stateTrajectory, x, axis=0)

            self.inputTrajectory = np.append(self.inputTrajectory, u, axis=0)

            self.time = np.append(self.time, t)
    ## END of updateLog()

    def reshapeLog(self):
        
        self.stateTrajectory = self.stateTrajectory.reshape(-1, self.__kwargs['nx'])

        self.inputTrajectory = self.inputTrajectory.reshape(-1, self.__kwargs['nu'])
    ## END of reshapeLog()

## END of class pyLog
