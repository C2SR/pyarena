# TODOC Summary: explain...

# create a system class
class DynamicalSystem:
    def __init__(self, **kwargs):
        self.nx = kwargs['nx']
        self.nu = kwargs['nu']
        self.stateEquation = kwargs['stateEquation']
        self.isMeasurementEquation = False
        self.initialCondition = kwargs['initialCondition']
        self.controller = kwargs['controller']