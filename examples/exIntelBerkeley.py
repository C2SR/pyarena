import sys
sys.path.append('..')

from pyArena.datasets import intel_berkeley as ib
import numpy as np
import matplotlib.pyplot as plt

Tsim = 1000 # simulation time in seconds
dt = 100 # sampling time in seconds

kwargsDataSet = {'gridResolution': 2, \
                 'dt': dt, \
                 'Tsim': Tsim}
DataSet = ib.IntelBerkeley(**kwargsDataSet)

for index in range(0, Tsim, dt):
    DataSet.plotFullGroundTruth(index)
