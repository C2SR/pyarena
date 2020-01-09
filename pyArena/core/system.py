"""
Summary: Dynamic System class implements a generic dynamic system described by its
state (x) and input (u). The dimension of the state is given by the parameter x_dimension
and the dimension of the input by u_dimension. The evolution in time of the system is
given by the state equation to be implemented using the abstract method stateEquation().

This abstract class implements four basic functions:
- iterate()
- publish_state()
- input_callback()
- run()

"""


# Python libraries
import numpy as np
import time
from scipy.integrate import solve_ivp as ode45
from abc import ABC, abstractmethod

# ROS libraries
import rospy
from std_msgs.msg import Float32MultiArray

## DynamicSystem (abstract) class ##
class DynamicSystem(ABC):

    def __init__(self, **kwargs):

        # Checking for missing parameters
        if 'x_dimension' not in kwargs:
            raise KeyError("Must specify number of states x_dimension")
        if 'u_dimension' not in kwargs:
            raise KeyError("Must specify number of inputs u_dimension")
        if 'real_time' not in kwargs:
            raise KeyError('Must specify real_time mode (True/False) for simulation!')
        if 'dt' not in kwargs:
            raise KeyError('No real_time mode specified for simulation!')
            
        # Retrieving parameters
        self.real_time = kwargs['real_time']
        self.dt = kwargs['dt']
        self.x_dimension = kwargs['x_dimension']
        self.u_dimension = kwargs['u_dimension']
        self.x = kwargs['initialCondition']
        
        # Initializing varibles
        self.u = np.zeros(self.u_dimension)
        self.t0 = 0
        self.t = 0

        # ROS node/publisher/subscribers
        rospy.init_node('system', anonymous=True)
        self.state_pub = rospy.Publisher('state', Float32MultiArray, queue_size=10)
        rospy.Subscriber("input", Float32MultiArray, self.input_callback)

        # Message
        self.msg_state = Float32MultiArray()
        self.msg_state.data = np.zeros(self.x_dimension+1)

    """
    State equation that defines the dynamics of the system
    """
    @abstractmethod
    def stateEquation(self, t, x, u):
        pass

    """
    Iterate the system dynamics forward in time by a single time step.
    """
    def iterate(self, timer):
        if self.real_time:
            self.t = timer.current_real.to_sec() - self.t0 
        # Iterating the state of the vehicle
        ode_fun = lambda t, x: self.stateEquation(t, self.x, self.u)
        sol = ode45(ode_fun, [0, self.dt], self.x)
        self.x = sol.y[:,-1]        
        self.publish_state(rospy.Time.from_sec(0))

    """
    Publishes the current state on ROS
    """
    def publish_state(self, timer):
        # Assemble and send message
        self.msg_state.data[0] = self.t
        for i in range(0, self.x_dimension):
            self.msg_state.data[i+1] = self.x[i]
        self.state_pub.publish(self.msg_state)       
    
    """
    Receives via ROS the input of the system
    """
    def input_callback(self, msg):
        for i in range(0, self.u_dimension):
            self.u[i] = msg.data[i]
        
        if not self.real_time:
            self.t += self.dt
            self.iterate(rospy.Time.from_sec(self.t))

    """
    Runs simulation in an infinite loop
    real_time == TRUE: call the iterate function based on time step
    real_time == FALSE: call the iterate function as soon as receives an input
    """
    def run(self):
        if self.real_time:
            self.t0 = rospy.Time.now().to_sec()
            self.timer = rospy.Timer(rospy.Duration(self.dt), self.iterate)
        else:
            self.timer = rospy.Timer(rospy.Duration(1), self.iterate)
