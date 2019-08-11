import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Careful about disabling the warning, may not be a problem in this specific case
pd.set_option('mode.chained_assignment', None)


class IntelBerkeley:

    def __init__(self, **kwargs):

        if 'path' in kwargs:
            path = kwargs['location']
        else:
            path = '/home/praveen/Documents/MEGA/codes/pyArena/test/dataset'

        if 'dt' in kwargs:
            self.dt = kwargs['dt']
        else:
            self.dt = 1.0

        if 'Tsim' in kwargs:
            self.Tsim = kwargs['Tsim']
        else:
            self.Tsim = 1000  # seconds

        if 'noiseCovariance' in kwargs:
            self.noiseCov = kwargs['noiseCov']
        else:
            self.noiseCov = np.array([0.1, 0.1, 0.1, 0.1])

        if 'gridResolution' in kwargs:
            self.gridResolution = kwargs['gridResolution']
        else:
            self.gridResolution = 1.0

        sensorDataPath = path + '/IntelBerkeley.txt'
        sensorPositionPath = path + '/mote_locs.txt'

        column_names = ['Date', 'Time', 'Epoch', 'ID', 'Temperature', 'Humidity', 'Light', 'Voltage']
        fullSensorData = pd.read_table(sensorDataPath, sep=' ', names=column_names)

        self.sensorPositionData = pd.read_table(sensorPositionPath, sep=' ', names=['ID', 'x', 'y'])
        self.xmax = self.sensorPositionData['x'].max()
        self.ymax = self.sensorPositionData['y'].max()
        self.xmin = 0.0
        self.ymin = 0.0

        # Required for spatial interpolation
        self.base_position = self.sensorPositionData.loc[:, 'x':'y'].to_numpy()

        # Settings to get full ground truth data
        xRes = (self.xmax - self.xmin) / self.gridResolution
        yRes = (self.ymax - self.ymin) / self.gridResolution
        gridX, gridY = np.mgrid[self.xmin:self.xmax:xRes * 1j, self.ymin:self.ymax:yRes * 1j]

        X = gridX.reshape(gridX.shape[0] * gridX.shape[1])
        Y = gridY.reshape(gridY.shape[0] * gridY.shape[1])

        self.stackedPositions = np.stack((X, Y), axis=-1)
        self.numStackedPositions = len(self.stackedPositions)

        # For now extract only one day data
        onedaydata = fullSensorData[fullSensorData['Date'] < '2004-02-29']

        self.numSensors = len(self.sensorPositionData)

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

        self.end_time = self.start_time + pd.DateOffset(seconds=self.Tsim)

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
    __spatialInterpolate(at_position, base_position, base_readings)

    Note: Inverse distance weighing method for interpolation used at present.
    Other methods need to be explored.
    """

    def spatialInterpolate(self, at_position, base_position, base_readings):

        distances = np.sum((at_position - base_position) ** 2, axis=-1) ** (1. / 2)
        normalized_distances = distances / (np.sum(distances))

        return normalized_distances @ base_readings

    # End of spatialInterpolate

    def getSingleGroundTruth(self, t, position):

        timestamp = (self.start_time + pd.DateOffset(seconds=t)).round(str(self.dt) + 'S')

        base_readings = list()

        for sensor_index in range(self.numSensors):
            base_readings.append(self.sensorData[sensor_index] \
                                 .loc[timestamp, 'Temperature':'Voltage'].to_numpy())

        reading = self.spatialInterpolate(position, self.base_position, base_readings)

        return reading

    # End of getSingleGroundTruth

    def getFullGroundTruth(self, t):

        groundTruth = list()

        for index in range(0, self.numStackedPositions):
            reading = self.getSingleGroundTruth(t, self.stackedPositions[index])
            groundTruth.append(np.append(self.stackedPositions[index], reading))

        snapShot = pd.DataFrame(data=groundTruth, \
                                columns=['x', 'y', 'Temperature', 'Humidity', 'Light', 'Voltage'])

        return snapShot

    def plotFullGroundTruth(self, t):

        # Get snapshot of data at time t
        snapShot = self.getFullGroundTruth(t)

        # Use Seaborn to obtain heatmap
        piv = pd.pivot_table(snapShot, values=['Temperature'], index=['y'], columns=['x'])
        ax = sns.heatmap(piv, xticklabels=False, yticklabels=False)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        plt.show()

    def getMeasurement(self, t, position):

        readings = self.getSingleGroundTruth(t, position)

        return readings + self.noiseCov * np.random.randn(self.numReadings)
