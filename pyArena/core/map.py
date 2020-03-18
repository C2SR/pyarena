# Python libraries
import numpy as np
import time
from abc import ABC, abstractmethod

# ROS libraries
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

## StaticController (abstract) class ##
class StaticMap(ABC):
    @abstractmethod
    def compute_map(self, t, x, measurement):
        pass

    @abstractmethod
    def get_map(self):
        pass

    def __init__(self, **kwargs):
        if type(self) is StaticMap:
            raise Exception("[StaticPlanner] Cannot create an instance of abstract class StaticMap")

        # Checking for missing parameters
        if 'x_dimension' not in kwargs:
            raise KeyError("[Map] Must specify number of states x_dimension")

        if 'width' not in kwargs:
            raise KeyError("[Map] Specify the map width")

        if 'height' not in kwargs:
            raise KeyError("[Map] Specify the height")

        width = kwargs['width'] 
        height = kwargs['height']        

        self.x_dimension = kwargs['x_dimension']
        self.dt = 0

        # ROS node/publisher/subscribers
        rospy.init_node('anonymous', anonymous=True)
        self.pt_cloud_pub = rospy.Publisher('map', PointCloud2, queue_size=10)
        rospy.Subscriber("state", Float32MultiArray, self.state_callback)
        rospy.Subscriber("sensor_data", Float32MultiArray, self.sensor_callback)
        self.map_pub = rospy.Publisher('map', PointCloud2, queue_size=10)

        # Initialization
        self.x = np.zeros(self.x_dimension)

        # Assemble single frame message
        self.cloud_msg = PointCloud2()
        self.cloud_msg.width = width 
        self.cloud_msg.height = height 
        
        # x-axis (This should be always constant)
        self.cloud_msg.fields.append(PointField())
        self.cloud_msg.fields[0].name = 'x'
        self.cloud_msg.fields[0].offset = 0
        self.cloud_msg.fields[0].datatype = np.dtype(np.float).itemsize 
        self.cloud_msg.fields[0].count = self.cloud_msg.width*self.cloud_msg.height 
        # x-axis (This should be always constant)
        self.cloud_msg.fields.append(PointField())
        self.cloud_msg.fields[1].name = 'y'
        self.cloud_msg.fields[1].offset = np.dtype(np.float).itemsize
        self.cloud_msg.fields[1].datatype = np.dtype(np.float).itemsize 
        self.cloud_msg.fields[1].count = self.cloud_msg.width*self.cloud_msg.height 
        # Temperature (This may vary)
        self.cloud_msg.fields.append(PointField())        
        self.cloud_msg.fields[2].name = 'Temperature'
        self.cloud_msg.fields[2].offset = np.dtype(np.float).itemsize + np.dtype(np.float).itemsize
        self.cloud_msg.fields[2].datatype = np.dtype(np.float).itemsize 
        self.cloud_msg.fields[2].count = self.cloud_msg.width*self.cloud_msg.height                 
        
        self.cloud_msg.is_bigendian = False
        self.cloud_msg.point_step = np.dtype(np.float).itemsize + \
                                    np.dtype(np.float).itemsize + \
                                    np.dtype(np.float).itemsize
        self.cloud_msg.row_step = self.cloud_msg.point_step*self.cloud_msg.width 
        self.cloud_msg.is_dense = True        


    def state_callback(self,msg):
        self.t = msg.data[0]
        
        for i in range(0, self.x_dimension):
            self.x[i] = msg.data[i+1]

    def sensor_callback(self, msg):
        t = self.t
        x = np.array(self.x)
        measurement = msg.data[0]

        self.compute_map(t, x, measurement)

    def publish_map(self, timer):
        self.get_map()    

    def run(self):
        self.timer = rospy.Timer(rospy.Duration(.5), self.publish_map)