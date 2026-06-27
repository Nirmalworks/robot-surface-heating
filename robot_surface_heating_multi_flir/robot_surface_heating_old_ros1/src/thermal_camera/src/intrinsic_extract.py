import numpy as np
data = np.load('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/thermal_camera/refactored_test3/thermal_intrinsics.npz')
print(data.files)
print(data['rms'])