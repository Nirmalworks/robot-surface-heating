# Robot Surface Heating

### Main Dependencies

**ROS**: ROS 2 Humble

**Python Libraries**
```bash
pip install open3d numpy opencv-python trimesh
```

### Thermal Camera Setup Info

#### Dependency Installation
Follow the README in [this GitHub](https://github.com/groupgets/purethermal1-uvc-capture) to set up the libraries and dependencies needed for the FLIR Lepton 3.5 thermal cameras.

Afterwards, 
1) Find the path to libuvc using `find /usr/local -name "libuvc*"`. It should end with only `.so`.
2) Put the following at the end of `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libuvc/
```

#### Running as Root with ROS
**IMPORTANT**: By default, the FLIR Lepton 3.5 cameras must be run as root. ROS 2 cannot be run as root due to security/permissions issues. Follow these steps to allow for running the Lepton cameras without being root:
1) Use `lsusb` to get a list of USB devices on the computer, the FLIR Lepton 3.5 cameras will most likely be listed as `Bus 001 Device 006: ID 1e4e:0100 Cubeternet WebCam`.
2) Create a new rule file for USB access with `sudo nano /etc/udev/rules.d/99-flir-libusb.rules`. In it, paste `SUBSYSTEM=="usb", ATTR{idVendor}=="1e4e", ATTR{idProduct}=="0100", MODE="0666", GROUP="plugdev"`, where `idVendor` and `idProduct` should match the ID string `<vendor>:<product>`from the previous step.
3) Reload the udev rules with
    ```bash
    sudo udevadm control --reload
    sudo udevadm trigger
    ```
    then unplug and replug each camera.
4) If your user is not part of the `plugdev` group (check with `groups $USER`), add it with `sudo usermod -aG plugdev $USER`.
5) Open a new terminal, or use `reset` on the current one. After this point, the cameras should work without running as root.




<!-- ### Running and Using the Thermal Pointcloud with Realsense

In five separate terminals cd'd and sourced to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, run
```bash
ros2 launch ur_robot_driver thermal_ur_control.launch.py
ros2 launch thermal_moveit_config dual_moveit.launch.py 
ros2 launch realsense2_camera rs_launch.py spatial_filter.enable:=true temporal_filter.enable:=true align_depth.enable:=true pointcloud.enable:=true
ros2 launch thermal_camera thermal_pub.launch.py 
ros2 launch thermal_camera test_pointcloud.launch.py 
```

In RViz, add a Pointcloud2 panel to the displays, and set its topic to `/thermal_camera_0/points`. Under the topic dropdown, set the reliability police from `Reliable` to `Best Effort`. Currently, this pointcloud has update issues relating to message QoS inconsistencies, so to more readily update the pointcloud, restart the `thermal_pub.launch.py` thermal camera publisher node. -->

### Camera Extrinsic Calibration
First, in `robot_surface_heating/src/thermal_camera/config/pnp_calibration_config.yaml`, change `camera_count` and `extrinsics_dir_prefix` to set the number and name of output directories with extrinsic calibration results.

As an initial setup, in four separate terminals cd'd and sourced (`source install/setup.bash`) to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, run 
```bash
ros2 launch ur_robot_driver thermal_ur_control.launch.py
ros2 launch thermal_moveit_config dual_moveit.launch.py
ros2 launch thermal_camera thermal_pub.launch.py
ros2 launch thermal_camera lepton_visual.launch.py
```
to confirm that the cameras are able to see the desired images.

Then, kill the `thermal_pub.launch.py` and `lepton_Visual.launch.py` processes. In one of the two now-vacant terminals, run
```bash
ros2 launch thermal_camera pnp_extrinsic.launch.py
```

Then follow the instructions in the `pnp_extrinsic` terminal. All keyboard inputs are captured on the OpenCV popup tabs. After finishing calibration, in `robot_surface_heating/src/thermal_camera/config/camera_params.yaml`, set `extrinsics_prefix` under `/camera_frame_broadcaster` to the same one set initially in `robot_surface_heating/src/thermal_camera/config/pnp_calibration_config.yaml`. This will allow the system to automatically use the newly obtained camera transforms.

#### Calibration Tips
- Get at least 15 images per camera, dropping any images that have a misidentified centroid.
- Get a large diversity of points in terms of their real world position and their projection onto the camera image. Don't worry about getting all images for all cameras with each capture, it's better to get good images for each camera with a larger amount of captures.

### Running and Using the Thermal Pointcloud with CAD Registration
In four separate terminals cd'd and sourced to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, run
```bash
ros2 launch ur_robot_driver thermal_ur_control.launch.py
ros2 launch thermal_moveit_config dual_moveit.launch.py
ros2 launch thermal_camera thermal_pub.launch.py 
ros2 launch thermal_camera cad_launch.launch.py
```

In RViz, add a Pointcloud2 panel to the displays, and set its topic to `/colored_cad_pointcloud`. You can also visualize the CAD marker with a Marker display on `/cad_model` and the non-projected pointcloud with a Pointcloud2 display on `/cad_pointcloud`.


### Full RSS Demo Procedure
1) Turn on the UR10e robot and set it to the desired pose (using the teach pendant jog mode or the Move tab).
2) In four separate terminals cd'd and sourced to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, run
```bash
ros2 launch ur_robot_driver thermal_ur_control.launch.py
ros2 launch thermal_moveit_config dual_moveit.launch.py
ros2 launch thermal_camera thermal_pub.launch.py 
ros2 launch thermal_camera cad_launch.launch.py
```
Afterwards, Start the Program on the teach pendant. The Robot Driver node should indicate the updated connection.
3) In the launched RViz window, add a Pointcloud2 panel to the displays, and set its topic to `/colored_cad_pointcloud`. 
4) In a fifth terminal, cd'd and sourced like the other terminals in (2), run
```bash
ros2 launch robot_interface demo_interface.launch.py
```
5) Monitor the robot and display the RViz panels. Turn on the heat gun when prompted.
6) Optionally, in a sixth terminal, cd'd and sourced like the other terminals in (2), run
```bash
ros2 launch thermal_camera lepton_visual.launch.py
```
This will pull up the live thermal camera feed as 2D image streams in separate windows.
7) Turn off the heat gun before killing the robot control loop.


### Coldest Node Policy Demo
1) In five separate terminals cd'd and sourced to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, run
```bash
ros2 launch ur_robot_driver thermal_ur_control.launch.py
ros2 launch thermal_moveit_config thermal_moveit.launch.py
ros2 launch thermal_camera cams_and_cad.launch.py 
ros2 launch thermal_camera pcl_analysis.launch.py
```
2) A pop-up of the pointcloud will appear. Follow the instructions in the `pcl_analysis.launch.py` terminal window. Close the pop-up window when finished to select the pointcloud surface for operations.
3) In RViz, add a Pointcloud2 panel to the displays, and set its topic to `/selected_surface_points`. You can also visualize the coldest pose with a Pose display on `/debug_cold_pose`, and the CAD marker with a Marker display on `/cad_model`. Or, if saved, load the `thermal_rviz_pcl_analysis.rviz` config for a quick setup.
4) In another terminal cd'd and sourced to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, run
```bash
ros2 launch thermal_camera policy_launch.launch.py
```
This is the coldest node policy.
5) Make sure the robot is ready to operate before continuing. In another terminal cd'd and sourced to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, run
```bash
ros2 run robot_interface overheat_safety_robot_interface.py
```
This starts the control script for robot.

### Coldest Node Data Collection
1) Do **Coldest Node Policy Demo** steps 1-4.
2) Prepare another terminal cd'd and sourced to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, with the command 
```bash
ros2 run thermal_camera rosbag_writer
```
This is the recording program that saves the runtime data to a rosbag located in `robot_surface_heating/thermal_data_bag`.
3) Do **Coldest Node Policy Demo** step 5, and immediately after run the prepared program from step 2.
4) When finished with data collection, terminate the `rosbag_writer` terminal.
5) In `robot_surface_heating/src/thermal_camera/thermal_camera/rosbag_reader.py`, update `self.save_file_path` to contain the desired output directory for the written data.
6) In two terminals cd'd and sourced to the `~/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating` directory, prepare these two commands
```bash
ros2 bag play thermal_data_bag
ros2 run thermal_camera rosbag_reader
```
7) When ready, run both commands from step 5. When the `rosbag play` process is finished, terminate the `rosbag_reader` process. The data has been stored in an NPZ with the following structure:
- The selected pointcloud is stored in ```data['selected_points']``` in an array of shape ```(<number of timesteps>,<number of points>,4)```, where the last dimension is ```[X,Y,Z,thermal_value (Kelvin*100)]```
- The rest of the pointcloud is stored in ```data['other_points']``` in an array of shape ```(<number of timesteps>,<number of points>,4)```, where the last dimension is ```[X,Y,Z,thermal_value (Kelvin*100)]```
- The robot poses captured during data collection are stored in ```data['robot_poses']``` in an array of shape ```(<number of timesteps>,7)```, where the last dimensions is ```[X,Y,Z,qX,qY,qZ,qW]```
- The timesteps for each of the last three data arrays is stored in ```data['timesteps]``` in an array of shape ```(3,max(<timestep amounts for each other data array>))```. Subarrays with less than the maximum timesteps are padded with NaNs.