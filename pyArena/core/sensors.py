from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class AbstractSensor(ABC):

    def __init__(self):
        if type(self) is AbstractSensor:
            raise Exception("Cannot create an instance of Abstract class - AbstractSensor")

    @abstractmethod
    def sense(self, t, x, *args):
        pass

class StaticScalarField2D(AbstractSensor):

    def __init__(self, **kwargs):

        super().__init__()

        if 'ScalarField' not in kwargs:
            raise KeyError("Input arguments must have ScalarField")

        self.scalar_field = kwargs['ScalarField']

        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'scalar_field';

        if 'covariance' in kwargs:
            self.noiseCov = kwargs['covariance']
        else:
            self.noiseCov = 0.1 # a default value

        self.numMeasurements = 1

        self.measurement = 0

    def sense(self, t, x, *args):

        # TODO: may have to remove this hard coding, or make it more intuitive
        position = x[0:2]

        self.measurement = self.scalar_field(position) + self.noiseCov*np.random.randn()

        return self.measurement

    def getGroundTruth(self, t, x, *args):

        # TODO: may have to remove this hard coding, or make it more intuitive
        position = x[0:2]

        return self.scalar_field(position)

    def getFullPlotData(self, xmax = [1,1], xmin = [-1,-1], numGrid = 10):

        x1 = np.linspace(xmin[0], xmax[1], numGrid)

        x2 = np.linspace(xmin[1], xmax[1], numGrid)

        XX1, XX2 = np.meshgrid(x1,x2)

        x = np.stack((XX1.reshape(XX1.shape[0]*XX1.shape[1]), \
                XX2.reshape(XX2.shape[0]*XX2.shape[1]) ))

        y = self.scalar_field(x)

        return x, y
