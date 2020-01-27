# Python libraries
import numpy as np
import time
from abc import ABC, abstractmethod

# ROS libraries
import rospy
from std_msgs.msg import Float32MultiArray

## StaticController (abstract) class ##
class StaticController(ABC):
    @abstractmethod
    def compute_input(self, t, x):
        pass

    @abstractmethod
    def update_reference(self, t, ref):
        pass

    def __init__(self, **kwargs):
        if type(self) is StaticController:
            raise Exception("Cannot create an instance of abstract class BaseController")

        # Checking for missing parameters
        if 'x_dimension' not in kwargs:
            raise KeyError("[Controller] Must specify number of states x_dimension")
        if 'u_dimension' not in kwargs:
            raise KeyError("[Controller] Must specify number of inputs u_dimension")
        if 'real_time' not in kwargs:
            raise KeyError('[Controller] Must specify real_time mode (True/False) for simulation!')
        if 'dt' not in kwargs:
            raise KeyError('[Controller] Please specify the SAMPLING TIME dt!')
            
        # Retrieving parameters            
        self.x_dimension = kwargs['x_dimension']
        self.u_dimension = kwargs['u_dimension']
        self.real_time = kwargs['real_time']
        self.dt = kwargs['dt']

        # Initializing varibles
        self.x = np.zeros(self.x_dimension)
        self.u = np.zeros(self.u_dimension)
        self.t = 0

        # ROS node/publisher/subscribers
        rospy.init_node('anonymous', anonymous=True)
        self.input_pub = rospy.Publisher('input', Float32MultiArray, queue_size=10)
        rospy.Subscriber("state", Float32MultiArray, self.state_callback)
        rospy.Subscriber("reference", Float32MultiArray, self.reference_callback)        

        # Message
        self.msg_input = Float32MultiArray()
        self.msg_input.data = np.zeros(self.u_dimension)
        self.has_new_state_message = False

    def iterate(self, timer): 
        if self.has_new_state_message:
            self.u = self.compute_input(self.t, self.x)
            self.has_new_state_message = False

        # Assemble and send message
        for i in range(0, self.u_dimension):
            self.msg_input.data[i] = self.u[i]
        self.input_pub.publish(self.msg_input) 

    def state_callback(self,msg):
        self.t = msg.data[0]
        for i in range(0, self.x_dimension):
            self.x[i] = msg.data[i+1]
        self.has_new_state_message = True

        if not self.real_time:
            self.iterate( rospy.Time.from_sec(0))

    def reference_callback(self, msg):
        ref = np.array(msg.data)
        self.update_reference(self.t, ref)

    def run(self):
        if self.real_time:
            self.timer = rospy.Timer(rospy.Duration(self.dt), self.iterate)
