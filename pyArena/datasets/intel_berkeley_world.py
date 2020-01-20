# Summary: Load a a pandas dataFrame and publishes as a pointCloud

import numpy as np
import pandas as pd

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

class IntelBerkeleyWorld:

    def __init__(self, **kwargs):
        # Parameters

        self.dt = 0.5

        # Open file and load as a pandas dataFrame
        path = kwargs['path'] if 'path' in kwargs else '~/frame_0.pkl'
        data = pd.read_pickle(path)

        # Publisher
        rospy.init_node('world', anonymous=True)
        self.pt_cloud_pub = rospy.Publisher('world', PointCloud2, queue_size=10)

        # Assemble single frame message
        self.cloud_msg = PointCloud2()
        self.cloud_msg.width = len(data.x.unique()) 
        self.cloud_msg.height = len(data.y.unique()) 
        
        # x-axis (This should be always constant)
        self.cloud_msg.fields.append(PointField())
        self.cloud_msg.fields[0].name = 'x'
        self.cloud_msg.fields[0].offset = 0
        self.cloud_msg.fields[0].datatype = data['x'].dtype.itemsize 
        self.cloud_msg.fields[0].count = self.cloud_msg.width*self.cloud_msg.height 
        # x-axis (This should be always constant)
        self.cloud_msg.fields.append(PointField())
        self.cloud_msg.fields[1].name = 'y'
        self.cloud_msg.fields[1].offset = data['x'].dtype.itemsize
        self.cloud_msg.fields[1].datatype = data['y'].dtype.itemsize 
        self.cloud_msg.fields[1].count = self.cloud_msg.width*self.cloud_msg.height 
        # Temperature (This may vary)
        self.cloud_msg.fields.append(PointField())        
        self.cloud_msg.fields[2].name = 'Temperature'
        self.cloud_msg.fields[2].offset = data['x'].dtype.itemsize + data['y'].dtype.itemsize
        self.cloud_msg.fields[2].datatype = data['Temperature'].dtype.itemsize 
        self.cloud_msg.fields[2].count = self.cloud_msg.width*self.cloud_msg.height                 
        
        self.cloud_msg.is_bigendian = False
        self.cloud_msg.point_step = data['x'].dtype.itemsize + \
                                    data['y'].dtype.itemsize + \
                                    data['Temperature'].dtype.itemsize
        self.cloud_msg.row_step = self.cloud_msg.point_step*self.cloud_msg.width 
        self.cloud_msg.data = data.loc[:,['x','y','Temperature']].to_numpy().tostring() 
        self.cloud_msg.is_dense = True
        
    
    def publish(self, timer):
        print('Publishing')
        self.pt_cloud_pub.publish(self.cloud_msg)

    def run(self):
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.publish)

if __name__ == "__main__":
    world = IntelBerkeleyWorld()

    world.run()

    rospy.spin()

    pass