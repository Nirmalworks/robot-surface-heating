import csv
import matplotlib.pyplot as plt
import os, shutil

perspective_no = 3
point_no = 4

new_folder_name = "point_"+str(point_no)+"_perspective_"+str(perspective_no)

os.mkdir('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/'+new_folder_name)
shutil.copy('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/log.csv', '/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/'+new_folder_name+'/log.csv')
shutil.copy('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/log1.csv', '/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/'+new_folder_name+'/log1.csv')

data_file_1 = open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/'+new_folder_name+'/log.csv')
data_file_2 = open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/'+new_folder_name+'/log1.csv')

data_file_1_reader = csv.reader(data_file_1, delimiter=',')
data_file_2_reader = csv.reader(data_file_2, delimiter=',')

data_file_1_time = []
data_file_1_temp = []

data_file_2_time = []
data_file_2_temp = []

offset = 8.0

for row in data_file_1_reader:
    if(row[0] != 'Timestamp'):
        print(row)
        data_file_1_time.append(float(row[0]))
        data_file_1_temp.append(float(row[1]))

for row in data_file_2_reader:
    if(row[0] != 'Timestamp'):
        print(row)
        data_file_2_time.append(float(row[0]))
        data_file_2_temp.append(float(row[1])+offset)

min_arr_len = min(len(data_file_2_temp), len(data_file_1_temp))
max_error = -float('inf')
for i in range(1, min_arr_len):
    error =abs(data_file_1_temp[-i] - data_file_2_temp[-i])
    max_error = max(max_error, error)

print(len(data_file_1_temp))
print(len(data_file_2_temp))

print(max_error)

fig = plt.figure()
plt.plot(data_file_1_time, data_file_1_temp, 'r')
plt.plot(data_file_2_time, data_file_2_temp)
plt.title("max. error: "+str(max_error)+", offset: "+str(offset)+", point "+str(point_no)+", perspective "+str(perspective_no))
fig.savefig('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/'+new_folder_name+'/plot.png')
plt.show()