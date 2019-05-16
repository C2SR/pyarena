import numpy as np
from abc import ABC, abstractmethod


class GaussianProcess(ABC):

    def __init__(self, **kwargs):
        pass

class GPRegression(GaussianProcess):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if 'kernel' not in kwargs:
            raise KeyError("Must specify a kernel function!")
        else:
            self.kernel = kwargs['kernel']

        if 'measurementNoiseCov' not in kwargs:
            self.noiseCov = 0.0
        else:
            self.noiseCov = kwargs['measurementNoiseCov']

        self.xTrain = []
        self.yTrain = []

    """
    trainModel(inpTrain, outTrain)
    The function is used to train the GP model given input (inpTrain) and  output (outTrain) training data.
    Training the GP model implies computation of prior mean and convariance distribution over the regressor function.

    Necessary function arguments:
    inpTrain - (numDim, numTrain) numpy array
    outTrain - (numTrain, 1) numpy array. Actually a 1D numpy array
    """
    def trainGP(self, inpTrain, outTrain):

        self.inpTrain = inpTrain.copy() # Please verify what happens without this copy()

        self.outTrain = outTrain.copy()

        self.numTrain = len(outTrain)

        # Compute the prior GP covariance matrix

        Ktrtr = np.zeros([self.numTrain,self.numTrain])

        for row in range(0,self.numTrain):
            for column in range(row,self.numTrain):
                Ktrtr[column, row] = Ktrtr[row, column] = self.kernel(inpTrain[:,row],inpTrain[:,column])

        self.priorCovariance = Ktrtr

    """
    trainModelIterative(inpTrain, outTrain)
    The function is used to train the GP model given input (inpTrain) and  output (outTrain) training data iteratively.
    The input and output training data is stored. Training is performed only when the trainingFlag is active
    Training the GP model implies computation of prior mean and convariance distribution over the regressor function.

    Necessary function arguments:
    inpTrain - (numDim, 1) numpy array
    outTrain - 1 float
    trainingFlag - bool. 0: do not train (only store new data). 1: store and train GP
    """
    def trainGPIterative(self, inpTrain, outTrain, trainingFlag):
        self.numTrain = len(self.yTrain) + 1

        self.yTrain =  np.append(self.yTrain, outTrain)

        self.xTrain = np.append(self.xTrain, inpTrain)
        self.xTrain = self.xTrain.reshape([self.numTrain, 2])
   
        if(trainingFlag):
            self.trainGP(self.xTrain.T, self.yTrain)
        return

    """
    Evaluate GP to obtain mean and value at a testing point
    inpTest is (numDim,1) numpy array
    """
    def predict_value(self, inpTest):

        Ktrte = np.zeros([self.numTrain,1])

        for index in range(0,self.numTrain):
            Ktrte[index] = self.kernel(self.inpTrain[:,index], inpTest)

        Ktrtr = self.priorCovariance

        Ktete = self.kernel(inpTest, inpTest)

        mu_hat = Ktrte.T @ np.linalg.inv(Ktrtr + self.noiseCov*np.eye(self.numTrain))  @ self.outTrain

        var_hat = Ktete - Ktrte.T @ np.linalg.inv(Ktrtr + self.noiseCov*np.eye(self.numTrain)) @ Ktrte

        return mu_hat, var_hat
