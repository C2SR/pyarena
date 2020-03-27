cmake_minimum_required(VERSION 2.8.3)
project(pyarena_vehicle)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package()

