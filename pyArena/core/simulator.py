# @TODO Summary: ..

import numpy as np
from scipy.integrate import solve_ivp as ode45
from . import logger

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

        # Create an attribute for data logging

        if self.system.isOutputEquation:
            kwargsLog = {'nx': self.system.nx, 'nu': self.system.nu, 'ny': self.system.ny}

        kwargsLog = {'nx': self.system.nx, 'nu': self.system.nu}

        self.sysLog  = logger.pyLog(**kwargsLog)
        
        if 'loglist' in kwargs:
            # code for additional logs
            pass
        else:
            self.sysLog = logger.pyLog(**kwargsLog)

    ## END of __init__()

    def run(self):

        numIndex =  int(self.simTime/self.dt)

        xSys = self.system.initialCondition

        for index in range(numIndex+1):

            time = index*self.dt

            uSys = self.system.controller.computeInput(time, xSys)

            self.sysLog.updateLog(time, xSys, uSys)

            ode_fun = lambda t,x: self.system.stateEquation(t, x, uSys)

            sol = ode45(ode_fun, [index*self.dt, (index+1)*self.dt], xSys)

            xSys = sol.y[:,-1]

        self.sysLog.reshapeLog()

        return self.sysLog
    ## END of run()

## END of class pyArena
