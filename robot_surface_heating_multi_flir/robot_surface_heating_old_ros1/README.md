## Package Break Down 
There are 4 packages needed. 
- thermal_camera
- robot_move_scripts
- motoman
- heater_moveit_config

The thermal camera contains drivers and scripts for the thermal camera <br>
The robot move scripts contains the launch files and python move scripts for running the robot <br>
The motoman conatins the gp12 config and drivers needed for the yaskawa <br>
the heater moveit config contains the files generated for the rviz config for the yaskawa <br>

## For Camera Calibration
Place cold washer on the corner of the carbon fiber sheet. Run uvc-radiometry.py. This code will record the coldest temperature location. Take this location and record it in the 
section of code that states # CALIBRATE THESE VALUES FOR PERSPECTIVE TRANSFORM #. Do this for each corner of the sheet. 

## For Coldest Node Movement

# To open Rviz
cd ~/Documents/GitHub/robot_surface_heating <br>
source devel/setup.bash <br>
roslaunch robot_move_scripts py_launch_heater.launch <br>

# To Run The Thermal Camera
Open new tab <br>
sudo -s <br>
source devel/setup.bash <br>
rosrun thermal_camera cold-node-transform-grid-uvc-radiometry-lab.py <br>

# To Set Up Robot Arm Movement
Open new tab <br>
source devel/setup.bash <br>
rosrun robot_move_scripts heat_to_Xdeg_cold_node_move_group_python_interface.py <br>


# NOTE: YOU MAY NEED TO UNPLUG AND PLUG THE CAMERA IF THERE IS A UVC ERROR

move_group_python_interface_tutorial.py is random policy <br>
cold_node is heating the coldest node for X seconds <br>
heat_to_Xdeg is heating the coldest node to X temp <br>


