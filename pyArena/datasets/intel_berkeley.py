import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt

# Careful about disabling the warning, may not be a problem in this specific case
pd.set_option('mode.chained_assignment', None)


class IntelBerkeley:

    def __init__(self, **kwargs):

        if 'path' in kwargs:
            path = kwargs['location']
        else:
            path = '/home/romulo/Documents/dataset'

        if 'dt' in kwargs:
            self.dt = kwargs['dt']
        else:
            self.dt = 1.0

        if 'T_sim' in kwargs:
            self.T_sim = kwargs['T_sim']
        else:
            self.T_sim = 1000  # seconds

        if 'noise_covariance' in kwargs:
            self.noise_cov = kwargs['noise_covariance']
        else:
            self.noise_cov = np.array([0.1, 0.1, 0.1, 0.1])

        if 'grid_resolution' in kwargs:
            self.grid_resolution = kwargs['grid_resolution']
        else:
            self.grid_resolution = 1

        # Path to sensor data and location
        sensor_data_path = path + '/IntelBerkeley.txt'
        sensor_position_path = path + '/mote_locs.txt'
        
        # Retrieving labeled data
        column_names = ['Date', 'Time', 'Epoch', 'ID', 'Temperature', 'Humidity', 'Light', 'Voltage']
        full_sensor_data = pd.read_table(sensor_data_path, sep=' ', names=column_names)
        self.sensor_position_data = pd.read_table(sensor_position_path, sep=' ', names=['ID', 'x', 'y'])

        # Sensor location limits
        self.x_max = self.sensor_position_data['x'].max()
        self.y_max = self.sensor_position_data['y'].max()
        self.x_min = self.sensor_position_data['x'].min()
        self.y_min = self.sensor_position_data['y'].min()

        # Required for spatial interpolation
        self.base_position = self.sensor_position_data.loc[:, 'x':'y'].to_numpy()

        # Resolution settings to get full ground truth data
        xRes = (self.x_max - self.x_min) / self.grid_resolution
        yRes = (self.y_max - self.y_min) / self.grid_resolution
        gridX, gridY = np.mgrid[self.x_min:self.x_max:xRes * 1j, self.y_min:self.y_max:yRes * 1j]

        X = gridX.reshape(gridX.shape[0] * gridX.shape[1])
        Y = gridY.reshape(gridY.shape[0] * gridY.shape[1])

        self.stacked_positions = np.stack((X, Y), axis=-1)
        self.num_stacked_positions = len(self.stacked_positions)

        # For now extract only one day data
        onedaydata = full_sensor_data[ full_sensor_data['Date'] < '2004-02-29'  ]
        onedaydata = onedaydata[ onedaydata['Date'] >= '2004-02-28'  ]

        self.numSensors = len(self.sensor_position_data)

        self.sensorData = list()
        start_time_list = list()

        for sensorID_index in range(self.numSensors):
            # Extract individual sensor data
            temp_dataframe = onedaydata[onedaydata['ID'] == (sensorID_index + 1)]

            # Rename index with time stamps - useful for interpolation
            temp_dataframe.rename(index=pd.to_datetime(temp_dataframe.loc[:, 'Time']), inplace=True)

            # Sort the index to replicate increasing time stamps
            temp_dataframe = temp_dataframe.sort_index()
  
            start_time_list.append(temp_dataframe.index.min())

            # Append to list of individual sensor data
            self.sensorData.append(temp_dataframe)
            print("Sensor ID {}: Number of readings {}".format(sensorID_index + 1, len(temp_dataframe)))

        # Compute start time and end time for simulation of recorded data
        self.start_time = min(start_time_list)
        start_time_index = start_time_list.index(self.start_time)

        self.end_time = self.start_time + pd.DateOffset(seconds=self.T_sim)

        # Add the start time to all the sensor data and resample with given dt
        print('Resampling, Interpolation and Truncation!')
        for sensor_index in range(self.numSensors):
            if sensor_index != start_time_index:
                self.sensorData[sensor_index].loc[self.start_time] = np.nan

            # Resample
            self.sensorData[sensor_index] = self.sensorData[sensor_index] \
                .resample(str(self.dt) + 'S').mean()

            # Interpolate
            self.sensorData[sensor_index] = self.sensorData[sensor_index] \
                .interpolate(method='linear', limit_direction='both')

            self.sensorData[sensor_index] = self.sensorData[sensor_index] \
                .loc[self.sensorData[sensor_index].index < self.end_time]

            self.sensorData[sensor_index] = self.sensorData[sensor_index].fillna(value=0.0)

        print('Done!!')

    # End of __init__() of class IntelBerkeley

    """
    Spatial interpolation function
    at_position - 1 x 2 numpy array - position at which interpolation values are to be found.
    base_position - num_pos x 2 numpy array = positions at which the base_readings are known
    base_readings - num_pos x num_readings numpy array - 
                    various readings/measurement corresponding to a single base_position.
    return - 1 x num_readings numpy array - interpolated value at at_position

    Example:

    # Given 4 base_positions and 3 readings corresponding to single base_position
    base_position = np.array([[0,0],[5,5],[0,5],[5,0]])
    base_readings = np.array([[15,200,0.20],[25,250,0.10],[20,150,0.15],[22,220,0.13]])

    # Position at which interpolated readings (3 readings) need to be found
    at_position = np.array([2,2])

    # Interpolated readings at at_position
    __spatial_interpolate(at_position, base_position, base_readings)

    Note: Inverse distance weighing method for interpolation used at present.
    Other methods need to be explored.
    """

    def spatial_interpolate(self, at_position, base_position, base_readings):

        distances = np.sum((at_position - base_position) ** 2, axis=-1) ** (1. / 2)
        inverse_distances = np.minimum(1./distances, 1000)
        normalized_inverse_distances = inverse_distances / (np.sum(inverse_distances))

        return normalized_inverse_distances @ base_readings

    # End of spatial_interpolate

    def get_single_ground_truth(self, t, position):

        timestamp = (self.start_time + pd.DateOffset(seconds=t)).round(str(self.dt) + 'S')

        base_readings = list()

        for sensor_index in range(self.numSensors):
            base_readings.append(self.sensorData[sensor_index] \
                                 .loc[timestamp, 'Temperature':'Voltage'].to_numpy())

        reading = self.spatial_interpolate(position, self.base_position, base_readings)

        return reading

    # End of get_single_ground_truth

    def get_full_ground_truth(self, t):

        groundTruth = list()

        for index in range(0, self.num_stacked_positions):
            reading = self.get_single_ground_truth(t, self.stacked_positions[index])
            groundTruth.append(np.append(self.stacked_positions[index], reading))

        snapShot = pd.DataFrame(data=groundTruth, \
                                columns=['x', 'y', 'Temperature', 'Humidity', 'Light', 'Voltage'])

        print(self.base_position)
        return snapShot

    def plot_full_ground_truth(self, t, sensor_type='Temperature'):

        # Get snapshot of data at time t
        snapShot = self.get_full_ground_truth(t)

        # Use Seaborn to obtain heatmap
        piv = pd.pivot_table(snapShot, values=[sensor_type], index=['y'], columns=['x'])
        fig, ax = plt.subplots()
        ax = sns.heatmap(piv, ax=ax, xticklabels='auto', yticklabels='auto', cbar=True)
        ax.invert_yaxis()
        ax.set(xlabel='x [m]', ylabel='y [m]')
        plt.show()

    def get_measurement(self, t, position):

        readings = self.get_single_ground_truth(t, position)

        # @BUG what is this self.num_readings?? It is not defined anywhere
        return readings + self.noise_cov * np.random.randn(self.num_readings)


if __name__ == "__main__":
    dataset = IntelBerkeley()
    dataset.plot_full_ground_truth(0, 'Temperature')