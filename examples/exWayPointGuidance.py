# Do this once
import sys
sys.path.append('..')

## Necessary imports
import pyArena.core as pyacore
import pyArena.control.guidance2D as pyaguidance
import pyArena.vehicles.underactuatedvehicle as pyavehicle
import matplotlib.pyplot as plt
import numpy as np

## Define a Waypoint Guidance controller based on LOS guidance

class WayPointGuidance(pyacore.controller.StaticController):

    def __init__(self, **kwargs):

        super().__init__()

        # Create a Way point array or table

        self.way_points = np.array([[0.0, 0.0],\
                                    [100, 0.0],\
                                    [100, 50],\
                                    [0.0, 50],\
                                    [0.0, 100],\
                                    [100, 100]])

        self.numWP = len(self.way_points)

        self.wpCounter = 1

        kwargsGuidance = {'wayPoint': self.way_points[self.wpCounter - 1], 'speed': 1}

        self.guidanceLaw = pyaguidance.LOSUnicycle(**kwargsGuidance)

    def computeInput(self, t, x, *measurements):

        if self.guidanceLaw.isWayPointReached is True:

            if (self.wpCounter == self.numWP):
                self.wpCounter = self.wpCounter
            else:
                self.wpCounter = self.wpCounter + 1
                self.guidanceLaw.setWayPoint(self.way_points[self.wpCounter-1])
                print("Way Point number {} set!".format(self.wpCounter))

        control_input = self.guidanceLaw.run(t,x)

        return control_input

## Simulation parameters

nx = 3
nu = 2
Tsim = 600
dt = 0.1

x_init = np.array([5.0, -5.0, np.pi/3])

## Define controller

controller = WayPointGuidance()

## Create a system with Controller

kwargsSystem = {'initialCondition': x_init, 'controller': controller}

system = pyavehicle.UnicycleKinematics(**kwargsSystem)

## Create pyArena simulation object and simulate

kwargsSimulation = {'system': system, 'simTime': Tsim, 'dt': dt}

pyA = pyacore.simulator.pyArena(**kwargsSimulation)

dataLog = pyA.run()

## Plot data

plt.plot(dataLog.stateTrajectory[:,0], dataLog.stateTrajectory[:,1], 'r')

for index in range(controller.numWP):
    wp = controller.way_points[index]
    plt.plot(wp[0], wp[1], '*b')

plt.show()
