import numpy as np

class LOSUnicycle:

    def __init__(self, **kwargs):

        if 'wayPoint' in kwargs:
            self.way_point = kwargs['wayPoint']
        else:
            self.way_point = None

        if 'lookahead' in kwargs:
            self.look_ahead = kwargs['lookahead']
        else:
            self.look_ahead = 5

        self.speed = kwargs['speed']

        self.ref = None

        self.isWayPointReached = False

    def setWayPoint(self, way_point):

        self.way_point = way_point

        self.ref = None

        self.isWayPointReached = False

    def run(self, t, x, *measurements):

        if self.ref is None:
            # Set the initial robot position as reference position
            self.ref = x[0:2]

        position = x[0:2]

        heading = x[2]

        los_angle = np.arctan2(self.way_point[1] - self.ref[1], self.way_point[0] - self.ref[0])

        Rot_los = np.array([[np.cos(los_angle), -np.sin(los_angle)],[np.sin(los_angle), np.cos(los_angle)]])

        # Compute the desired heading for the vehicle

        wp_ref = Rot_los.T@(self.way_point - self.ref)

        position_ref = Rot_los.T@(position - self.ref)

        heading_desired = - np.arctan(position_ref[1]/self.look_ahead) + los_angle

        if (np.sqrt((self.way_point - position)@(self.way_point - position)) < 1):
            v = 0
            omega = 0
            self.isWayPointReached = True
        else:
            v = self.speed
            omega = -0.6*(heading - heading_desired)

        return np.array([v,omega])
