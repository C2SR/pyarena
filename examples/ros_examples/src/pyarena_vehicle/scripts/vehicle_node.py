#!/usr/bin/env python
# Python libraries
import numpy as np
import sys

# ROS libraries
import rospy
from nav_msgs.msg import Odometry

from pyarena.vehicles.unicycle import Unicycle

class VehicleNode:
    def __init__(self, node_name):
        # ROS node
        rospy.init_node('vehicle', anonymous=True)

        # Global parameters
        world_frame = rospy.get_param('/world_frame', 'world')
        # Private parameters
        child_frame_id = rospy.get_param('~base_frame', 'base_link')
        # Mandatory parameters
        try:
            vehicle_type = rospy.get_param('~type') 
            x0 = rospy.get_param('~x0')
            dt = rospy.get_param('~dt')        
        except KeyError:
            rospy.logerr('[%s] Error retrieving one or more private parameters. '+
                        'Check if the following parameters are being properly initialized:\n' + 
                        '-- [%s/type]: type of vehicle (string);\n'
                        '-- [%s/x0]: initial state of the vehicle (list of floats);\n' +
                        '-- [%s/dt]: sampling time (float).', 
                         node_name, node_name, node_name, node_name)
            return
        # ROS publishers and subscribers
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=10)
   
        # ROS messages        
        self.odom = Odometry()
        self.odom.header.frame_id = world_frame
        self.odom.child_frame_id = child_frame_id

        # Initializing pyarena vehicle
        kwargsVehicle = {'x0': np.array(x0)}  # parameters shared by all vehicles
        if vehicle_type == 'unicycle':
            self.vehicle = Unicycle(**kwargsVehicle)
        else:
            rospy.logerr('[%s] Unknown vehicle type. Current support:\n'+
                         '-- unicycle (2d)',
                         node_name)
            return
        # Control input
        self.u = np.zeros(self.vehicle.u_dimension)

        # Miscellaneous
        self.dt = dt  # sampling time

        # ROS loop
        self.spin()

    def iterate(self, event):
        # Simulate vehicle        
        self.u = np.random.rand(self.vehicle.u_dimension)        
        print(self.u)
        x = self.vehicle.run(dt=self.dt,u=self.u)
        # Publish
        self.odom_publisher(x)

    def odom_publisher(self,x):
        # @TODO encapsulate this in a library
        self.odom.pose.pose.position.x = x[0]
        self.odom.pose.pose.position.y = x[1]    
        self.odom.pose.pose.orientation.x = 0
        self.odom.pose.pose.orientation.y = 0
        self.odom.pose.pose.orientation.z = np.sin(x[2] * 0.5)  
        self.odom.pose.pose.orientation.w = np.cos(x[2] * 0.5)                         
        self.odom_pub.publish(self.odom)
        
    def spin(self):
        rospy.Timer(rospy.Duration(self.dt), self.iterate)       
        rospy.spin()

if __name__ == "__main__":
    node = VehicleNode('~')