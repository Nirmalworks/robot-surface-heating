import matplotlib.pyplot as plt
import numpy as np
import csv

# Path to the CSV file
csv_filename = "/home/cam/csv_data/110_120_130_Normal_Al_15_flow_3_600C.csv"

# Initialize lists to hold your data
time_list = []
std_dev_list = []

# Read data from the CSV file
with open(csv_filename, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        time_list.append(float(row['Time_Seconds']))
        std_dev_list.append(float(row['STD DEV TEMP']))

# Convert lists to numpy arrays for more efficient numerical operations
time_array = np.array(time_list)
std_dev_array = np.array(std_dev_list)

# Plotting the standard deviation over time
plt.figure(figsize=(10, 6))
plt.plot(time_array, std_dev_array, marker='o', linestyle='-', color='b', label='Standard Deviation')
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Standard Deviation of Temperature', fontsize=14)
plt.title('Standard Deviation of Temperature Over Time', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
