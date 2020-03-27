# pyArena

After clonning this repository, **install** pyArena:
  > *pyArena$* sudo pip install -e .

For ROS usage, navigate to **ros_examples** workspace  and compilte it with **python3**
  > *pyArena/examples/ros_examples$* catkin_make --cmake-args -DPYTHON_VERSION=3
  
  After that, source the project
  > *pyArena/examples/ros_examples$* source devel/setup.bash
  
  Now you are ready to try the ROS examples. We reccomend using and modifying the launch files, e.g.,
  > roslaunch pyarena_vehicles single_vehicle.launch
