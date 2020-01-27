# Python libraries
import numpy as np
import time
from abc import ABC, abstractmethod

# ROS libraries
import rospy
from std_msgs.msg import Float32MultiArray

## StaticController (abstract) class ##
class StaticPlanner(ABC):
    @abstractmethod
    def compute_plan(self, t, x):
        pass

    def __init__(self, **kwargs):
        if type(self) is StaticPlanner:
            raise Exception("[StaticPlanner] Cannot create an instance of abstract class StaticPlanner")

        # Checking for missing parameters
        if 'x_dimension' not in kwargs:
            raise KeyError("[Controller] Must specify number of states x_dimension")

        self.x_dimension = kwargs['x_dimension']
        self.dt = 0

        # ROS node/publisher/subscribers
        rospy.init_node('anonymous', anonymous=True)
        self.planner_pub = rospy.Publisher('reference', Float32MultiArray, queue_size=10)
        rospy.Subscriber("state", Float32MultiArray, self.state_callback)

        # Initialization
        self.x = np.zeros(self.x_dimension)

        # Message
        self.msg_planner = Float32MultiArray()
        
    def send_plan(self, plan): 
        self.msg_planner.data = np.array(plan, dtype='float')
        self.planner_pub.publish(self.msg_planner)

    def state_callback(self,msg):
        self.t = msg.data[0]
        
        for i in range(0, self.x_dimension):
            self.x[i] = msg.data[i+1]
        self.has_new_state_message = True

        self.compute_input(self.t, self.x)
    def run(self):
        pass