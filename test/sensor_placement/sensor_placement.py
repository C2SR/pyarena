import sys
sys.path.append('../..')

import numpy as np
import pyArena.core as pyacore
import gp_model
import matplotlib.pyplot as plt


class SensorPlacement:

    def __init__(self, numSensors, gp):

        self.numSensors = numSensors
        self.gp = gp
        self.optimal_placements_entropy_id = -np.inf*np.ones(self.numSensors)
        self.optimal_placements_entropy = np.zeros([self.numSensors,2])

        self.optimal_placements_mutualinfo_id = -np.inf*np.ones(self.numSensors)
        self.optimal_placements_mutualinfo = np.zeros([self.numSensors,2])

        grid_x = self.gp.area.grid_x
        grid_y = self.gp.area.grid_y

        X, Y = np.mgrid[self.gp.xmin:self.gp.xmax:grid_x * 1j, self.gp.ymin:self.gp.ymax:grid_y * 1j]
        self.inp_locs = np.stack((X.reshape(X.shape[0] * X.shape[1]), \
                             Y.reshape(Y.shape[0] * Y.shape[1])), axis=-1)

    def run(self):
        self.optimizeEntropy()
        self.optimizeMutualInformation()


    def optimizeEntropy(self):

        print("Sensor placements using Entropy")

        numGrids = self.gp.area.numGrids
        grid_x = self.gp.area.grid_x
        grid_y = self.gp.area.grid_y

        X, Y = np.mgrid[self.gp.xmin:self.gp.xmax:grid_x * 1j, self.gp.ymin:self.gp.ymax:grid_y * 1j]
        inp_locs = np.stack((X.reshape(X.shape[0] * X.shape[1]), \
                            Y.reshape(Y.shape[0] * Y.shape[1])), axis=-1)

        conditional_entropy = -np.inf*np.ones(numGrids)

        # Find first optimal placement
        print("Placing sensor 1 ...")
        for location_index in range(0, numGrids):
            mean, var = gp.gp.predict_value(np.array([inp_locs[location_index]]))
            conditional_entropy[location_index] = 0.5*np.log(var) + 0.5*np.log(2*np.pi) + 0.5

        max_id = np.argwhere(conditional_entropy == np.max(conditional_entropy))
        if len(max_id) > 1:
            self.optimal_placements_entropy_id[0] = max_id[0]
            self.optimal_placements_entropy[0] = inp_locs[max_id[0]]
        else:
            self.optimal_placements_entropy_id[0] = max_id
            self.optimal_placements_entropy[0] = inp_locs[max_id]

        previous_conditional_entropy = np.max(conditional_entropy)

        for sensor_index in range(1, self.numSensors):
            print("Placing sensor {} ...".format(sensor_index + 1))
            joint_entropy = -np.inf * np.ones(numGrids)
            conditional_entropy = -np.inf * np.ones(numGrids)
            for location_index in range(0,numGrids):
                if location_index not in self.optimal_placements_entropy_id:
                    # Compute conditional entropy as H(y|X_{A_k-1}, D) = H(y, X_{A_k-1}| D) - H(X_{A_k-1}| D)
                    joint_sensor_locations = np.append(self.optimal_placements_entropy[0:sensor_index], [inp_locs[location_index]], axis=0)
                    mean, var = gp.gp.predict_value(joint_sensor_locations)
                    joint_entropy[location_index] = 0.5*np.log(np.linalg.det(var)) + (sensor_index + 1)*0.5*(1 + np.log(2*np.pi))
                    conditional_entropy[location_index] = joint_entropy[location_index] - previous_conditional_entropy

            max_id = np.argwhere(conditional_entropy == np.max(conditional_entropy))
            if len(max_id) > 1:
                self.optimal_placements_entropy_id[sensor_index] = max_id[0]
                self.optimal_placements_entropy[sensor_index] = inp_locs[max_id[0]]
            else:
                self.optimal_placements_entropy_id[sensor_index] = max_id
                self.optimal_placements_entropy[sensor_index] = inp_locs[max_id]

            previous_conditional_entropy = np.max(conditional_entropy)
    # End of function optimizeEntropy()

    def optimizeMutualInformation(self):
        print("Sensor placements using Mutual Information")
        numGrids = self.gp.area.numGrids

        # First optimal sensor placement
        mutual_information = -np.inf*np.ones(numGrids)

        print("Placing sensor 1 ...")
        for location_index in range(0, numGrids):

            # Compute conditional entropy H(y|X_{A_k}, D)
            mean, var = gp.gp.predict_value(np.array([self.inp_locs[location_index]]))
            conditional_entropy_isSensor = 0.5*np.log(np.linalg.det(var)) + (1)*0.5*(1 + np.log(2*np.pi))

            # Compute conditional entropy H(y|X_{B_k}, D)
            # Step 1: Compute conditional entropy H(X_{B_k}|D)
            non_sensor_locations = np.delete(self.inp_locs, location_index, axis=0)
            mean, var = gp.gp.predict_value(non_sensor_locations)
            entropy_notSensor = 0.5*np.log(np.linalg.det(var)) + len(non_sensor_locations)*0.5*(1 + np.log(2*np.pi))

            # Step 2: Compute joint entropy H(y, X_{B_k}|D)
            joint_locations = np.append(np.array([self.inp_locs[location_index]]), non_sensor_locations, axis=0)
            mean, var = gp.gp.predict_value(joint_locations)
            joint_entropy_notSensor = 0.5*np.log(np.linalg.det(var)) + len(joint_locations)*0.5*(1 + np.log(2*np.pi))

            #  Step 3: Compute conditional entropy H(y|X_{B_k}, D) = H(y, X_{B_k}|D) - H(X_{B_k}|D)
            conditional_entropy_notSensor = joint_entropy_notSensor - entropy_notSensor

            mutual_information[location_index] = conditional_entropy_isSensor - conditional_entropy_notSensor

        max_id = np.argwhere(mutual_information == np.max(mutual_information))
        if len(max_id) > 1:
            self.optimal_placements_mutualinfo_id[0] = max_id[0]
            self.optimal_placements_mutualinfo[0] = self.inp_locs[max_id[0]]
        else:
            self.optimal_placements_mutualinfo_id[0] = max_id
            self.optimal_placements_mutualinfo[0] = self.inp_locs[max_id]

        for sensor_index in range(1, self.numSensors):
            print("Placing sensor {} ...".format(sensor_index + 1))
            mutual_information = -np.inf * np.ones(numGrids)
            for location_index in range(0, numGrids):
                if location_index not in self.optimal_placements_mutualinfo_id:

                    # Step 1: Compute conditional entropy H(y|X_{A_k-1}, D)
                    ## Step 1a: Compute entropy H(X_{A_k-1}|D)
                    placed_sensors = self.optimal_placements_mutualinfo[0:sensor_index]
                    mean, var = gp.gp.predict_value(placed_sensors)
                    entropy_isSensor =  0.5*np.log(np.linalg.det(var)) + len(placed_sensors)*0.5*(1 + np.log(2*np.pi))

                    ## Step 1b: Compute joint entropy H(y, X_{A_k-1}|D)
                    joint_locations = np.append(np.array([self.inp_locs[location_index]]), placed_sensors, axis=0)
                    mean, var = gp.gp.predict_value(joint_locations)
                    joint_entropy_isSensor =  0.5*np.log(np.linalg.det(var)) + len(joint_locations)*0.5*(1 + np.log(2*np.pi))

                    ## Step 1c: Compute conditional entropy H(y|X_{A_k-1}, D) = H(y, X_{A_k-1}|D) - H(X_{A_k-1}|D)
                    conditional_entropy_isSensor = joint_entropy_isSensor - entropy_isSensor

                    # Step 2: Compute conditional entropy H(y|X_{B_k}, D)
                    ## Step 2a: Compute conditional entropy H(X_{B_k}|D)
                    non_sensor_locations_id = np.append(self.optimal_placements_mutualinfo_id[0:sensor_index], location_index)
                    non_sensor_locations = np.delete(self.inp_locs, non_sensor_locations_id.astype(int), axis=0)
                    mean, var = gp.gp.predict_value(non_sensor_locations)
                    entropy_notSensor = 0.5 * np.log(np.linalg.det(var)) + len(non_sensor_locations)*0.5*(1 + np.log(2 * np.pi))

                    ## Step 2b: Compute joint entropy H(y, X_{B_k}|D)
                    joint_locations = np.append(np.array([self.inp_locs[location_index]]), non_sensor_locations, axis=0)
                    mean, var = gp.gp.predict_value(joint_locations)
                    joint_entropy_notSensor = 0.5 * np.log(np.linalg.det(var)) + len(joint_locations)*0.5*(1 + np.log(2 * np.pi))

                    ## Step 2c: Compute conditional entropy H(y|X_{B_k}, D) = H(y, X_{B_k}|D) - H(X_{B_k}|D)
                    conditional_entropy_notSensor = joint_entropy_notSensor - entropy_notSensor

                    # Step 3: Mutual information
                    mutual_information[location_index] = conditional_entropy_isSensor - conditional_entropy_notSensor

            max_id = np.argwhere(mutual_information == np.max(mutual_information))
            if len(max_id) > 1:
                self.optimal_placements_mutualinfo_id[sensor_index] = max_id[0]
                self.optimal_placements_mutualinfo[sensor_index] = self.inp_locs[max_id[0]]
            else:
                self.optimal_placements_mutualinfo_id[sensor_index] = max_id
                self.optimal_placements_mutualinfo[sensor_index] = self.inp_locs[max_id]

    # End of function optimizeMutualInformation()
    def plot(self):
        self.gp.plot()
        h2 = plt.figure(2)
        plt.plot(self.optimal_placements_entropy[:,0],self.optimal_placements_entropy[:,1], '*b')
        plt.plot(self.optimal_placements_mutualinfo[:,0],self.optimal_placements_mutualinfo[:,1], '+r')
        plt.show()


if __name__ == "__main__":
    # Simulation parameters
    numSensors = 10

    # Simulation area
    area = pyacore.utils.Structure()
    area.x_dim = 150
    area.y_dim = 150
    area.grid_x = 10
    area.grid_y = 10
    area.numGrids = area.grid_x*area.grid_y
    area.numTrain = 30
    area.numTruth = 100

    # Create a GP model
    gp = gp_model.gpModel(area)

    # Run sensor placement
    sp = SensorPlacement(numSensors, gp)
    sp.run()
    sp.plot()