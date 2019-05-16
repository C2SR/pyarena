# @TODO Summary: ..

import numpy as np
from scipy.integrate import solve_ivp as ode45
from . import logger
from . import utils

class pyArena:
    ## TODO:  Do I need *args?
    def __init__(self, *args, **kwargs):

        if 'system' in kwargs:
            self.system = kwargs['system']
        else:
            raise KeyError('No system specified for simulation!')

        if 'simTime' in kwargs:
            self.simTime = kwargs['simTime']
        else:
            raise KeyError('Must specify simulation time!')

        if 'dt' in kwargs:
            self.dt = kwargs['dt']
        else:
            self.dt = 0.1
            print('Set default sampling time of {} seconds'.format(0.1))

        # Create a list of log objects
        self.logObjList = list()

        self.logObjList.append(logger.StateVectorLog(**{'nx': self.system.nx}))

        self.logObjList.append(logger.InputVectorLog(**{'nu': self.system.nu}))

        self.logObjList.append(logger.TimeLog())

        if self.system.isOutputEquation:
            pass # TODO: Must implement

        # TODO: Potential non-sense IF statement - consider moving to loglist section
        if self.system.sensor is not None:

            tempKwargsLog = {'name': self.system.sensor.name, \
            'size': self.system.sensor.numMeasurements,\
            'logFunction': lambda t, x, u, *args: self.system.sensor.measurement}

            self.logObjList.append(logger.InlineVectorLog(**tempKwargsLog))

        if 'loglist' in kwargs:
            # code for additional logs
            pass

        self.sysLog = utils.Structure()

    ## END of __init__()

    def run(self):

        numIndex =  int(self.simTime/self.dt)

        xSys = self.system.initialCondition

        for index in range(numIndex+1):

            time = index*self.dt

            if self.system.sensor is None:
                measurements = list()
            else:
                measurements = [self.system.sensor.sense(time, xSys)]

            uSys = self.system.controller.computeInput(time, xSys, *measurements)

            for index, logObj in enumerate(self.logObjList):
                logObj.updateLog(time, xSys, uSys)

            ode_fun = lambda t,x: self.system.stateEquation(t, x, uSys)

            sol = ode45(ode_fun, [index*self.dt, (index+1)*self.dt], xSys)

            xSys = sol.y[:,-1]

        for index, logObj in enumerate(self.logObjList):
            if logObj.size is 1:
                setattr(self.sysLog, logObj.name, logObj.data)
            else:
                setattr(self.sysLog, logObj.name, logObj.data.reshape(-1, logObj.size))

        return self.sysLog
    ## END of run()

## END of class pyArena
