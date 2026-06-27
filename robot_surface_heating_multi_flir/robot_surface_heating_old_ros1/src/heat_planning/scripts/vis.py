import csv
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle

csvfile = pd.read_csv('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/greedy56.csv')
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
heated = None

heated_durations = []
non_heated_durations = []

# for index, row in csvfile.iterrows():
#     if(row['Node_Index_x'] == -1 and row['Node_Index_y'] == -1):
#         if(heated == True):
#             plt.gca().add_patch(Rectangle((currptr1,min_temp),currptr2-currptr1,max_temp,fill=True, color='g', alpha=0.5, zorder=100, figure=fig))
#             heated_durations.append(currptr2-currptr1)
#             currptr1 = row['Time_Seconds']
#             currptr2 = None
#             heated = False
        
#         elif(currptr1 == None and currptr2 == None):
#             currptr1 = row['Time_Seconds']
#             heated = False
        
#         else:
#             currptr2 = row['Time_Seconds']
            
#     elif(row['Node_Index_x'] != -1 and row['Node_Index_y'] != -1):
#         if(heated == False):
#             plt.gca().add_patch(Rectangle((currptr1,min_temp),currptr2-currptr1,max_temp,fill=True, color='r', alpha=0.5, zorder=100, figure=fig))
#             non_heated_durations.append(currptr2-currptr1)
#             currptr1 = row['Time_Seconds']
#             currptr2 = None
#             heated = True
#         elif(currptr1 == None and currptr2 == None):
#             currptr1 = row['Time_Seconds']
#             heated = True

#         else:
#             currptr2 = row['Time_Seconds']

# if(heated == True):
#     plt.gca().add_patch(Rectangle((currptr1,min_temp),currptr2-currptr1,max_temp,fill=True, color='g', alpha=0.5, zorder=100, figure=fig))
#     heated_durations.append(currptr2-currptr1)
# else:
#     plt.gca().add_patch(Rectangle((currptr1,min_temp),currptr2-currptr1,max_temp,fill=True, color='r', alpha=0.5, zorder=100, figure=fig))
#     non_heated_durations.append(currptr2-currptr1)
            
print("List of heated durations")
for i in range(len(heated_durations)):
    print(f"{i+1}. {heated_durations[i]}")
print()
print("List of non-heated durations")
for i in range(len(non_heated_durations)):
    print(f"{i+1}. {non_heated_durations[i]}")
plt.ylim(75,200)
plt.show()