import numpy as np
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

from std_msgs.msg import Float32MultiArray
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray

class IntelBerkeleySensor:
    def __init__(self, **kwargs):
        # Parameters
        self.dt = 1

        # ROS node/publisher/subscribers
        rospy.init_node('sensor', anonymous=True)
        self.sensor_pub = rospy.Publisher('sensor_data', Float32MultiArray, queue_size=10)
        rospy.Subscriber("world", PointCloud2, self.world_callback)        
        rospy.Subscriber("state", Float32MultiArray, self.state_callback)

        # Initialize variables
        self.x = np.zeros(2)   # (x,y) position of the robot
        self.msg_measurement = Float32MultiArray()
        
        # Flags
        self.has_world_message = False
        self.has_state_message = False

    def world_callback(self, msg):
        pt_cloud_data = np.frombuffer(msg.data)
        self.x = np.unique(pt_cloud_data[0::3]) 
        self.y = np.unique(pt_cloud_data[1::3])        
        data = pt_cloud_data[2::3].reshape(len(self.x), len(self.y))

        # Create interpolating function [It took about 3 ms for a grid with .25 m cell resolution]
        self.get_measurement = interpolate.interp2d(self.x, self.y, data.T, kind='linear')
        self.has_world_message = True

        # @TODO This is a standard way to read the point cloud. Perhaps should investigate more.
        # It accounts for different data dtype size. It will take longer if we need to keep a for-loop.
        #unpacked = pc2.read_points(msg, field_names = ("x", "y", "temperature"), skip_nans=True)
        #for p in unpacked:
        #    print(" x : %f  y: %f  temp: %f" %(p[0],p[1],p[2]) )

    def state_callback(self,msg):
        for i in range(0, 2):
            self.x[i] = msg.data[i+1]
        self.has_state_message = True

    def publish_sensor_data(self, timer):
        if self.has_state_message and self.has_world_message:
            measurement = self.get_measurement(self.x[0],self.x[1])
            self.msg_measurement.data = np.array(measurement)
            self.sensor_pub.publish(self.msg_measurement)
        else:
            if not self.has_state_message:
                rospy.logwarn("[Sensor] Waiting for STATE message to arrive")
            if not self.has_world_message:
                rospy.logwarn("[Sensor] Waiting for WORLD message to arrive")

    def run(self):
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.publish_sensor_data)

if __name__ == "__main__":
    sensor = IntelBerkeleySensor()

    sensor.run()

    rospy.spin()