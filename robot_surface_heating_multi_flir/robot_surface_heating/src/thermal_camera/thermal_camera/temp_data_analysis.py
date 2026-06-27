import numpy as np
import matplotlib.pyplot as plt

# data = np.load('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_data.npz')
data = np.load('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_data_33C.npz')

# datas = [
#     np.load('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_data_33C.npz'),
#     np.load('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_data_29C.npz'),
#     np.load('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_data_40C.npz')
# ]

plt.figure(figsize=(12, 8))

t = [i for i in range(0,len(data['avg_temps']))]
desired_temp = data['desired_temp']

# plot average temperature over time
plt.subplot(1, 2, 1)
# for i in range(datas):
#     des_temp = datas[i]['desired_temp']
plt.plot(t, [elem.item() for elem in data['avg_temps']])
plt.xlabel("Timestep")
plt.ylabel("Temperature (Celsius)")
plt.title("Average Part Temperature", pad=20)
# plt.legend()

# plot temperature error from desired temperature
plt.subplot(1, 2, 2)
plt.plot(t, [elem.item() for elem in data['mean_error']])
plt.errorbar(t, [elem.item() for elem in data['mean_error']], [elem.item() for elem in data['std_dev_error']], linestyle='None', marker='o', ecolor='blue')
plt.xlabel("Timestep")
plt.ylabel("Temperature (Celsius)")
plt.title(f"Average Temperature Error from Desired ({desired_temp[0]} C)", pad=20)

plt.savefig('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_plot.png')
