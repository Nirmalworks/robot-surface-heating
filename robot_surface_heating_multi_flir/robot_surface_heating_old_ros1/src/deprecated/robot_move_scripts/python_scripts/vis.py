import csv
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle

csvfile = pd.read_csv('/home/cam/ND_ws/robot_surface_heating_ws/src/data/zigzag1.csv')
time_list = csvfile['Time_Seconds'].to_list()
not_heated_time_list = []
node_index_to_temp_list = {}

min_temp = float('inf')
max_temp = -float('inf')

fig = plt.figure() 
for key in csvfile.keys():
    if('Node_' in key and '_x' not in key and '_y' not in key):
       temp_list = csvfile[key].to_list()
       for temp in temp_list:
              if(temp < min_temp):
                min_temp = temp
              if(temp > max_temp):
                max_temp = temp
       plt.plot(time_list, temp_list, label=key)



currptr1 = None
currptr2 = None

for index, row in csvfile.iterrows():
    if(row['Node_Index_x'] == -1 and row['Node_Index_y'] == -1):
        if(currptr1 == None and currptr2 == None):
            currptr1 = row['Time_Seconds']

        elif(currptr1 != None and currptr2 == None):
            currptr2 = row['Time_Seconds']
            
    elif(row['Node_Index_x'] != -1 and row['Node_Index_y'] != -1):
        if(currptr1 != None and currptr2 != None):
            currptr2 = row['Time_Seconds']
            not_heated_time_list.append(currptr2-currptr1)
            print(currptr2-currptr1)

            plt.gca().add_patch(Rectangle((currptr1,min_temp),currptr2-currptr1,max_temp,fill=True, color='g', alpha=0.5, zorder=100, figure=fig))
            currptr1 = None
            currptr2 = None

        not_heated_time_list.append(row['Time_Seconds'])

plt.show()