import csv
import matplotlib.pyplot as plt

def plot_csv_data(csv_filename):
    # Lists to store time and average temperature data
    time_seconds, avg_temp, max_temp = [], [], []

    # Read the CSV file and extract data
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_seconds.append(float(row['Time_Seconds']))
            avg_temp.append(float(row['AVG_TEMP']))
            max_temp.append(float(row['MAX_TEMP']))

    # Plot data from the CSV file
    plt.figure(figsize=(10, 6))
    plt.plot(time_seconds, avg_temp, marker='o', linestyle='-', color='b', label='Average Temperature')
    plt.plot(time_seconds, max_temp, marker='x', linestyle='-', color='r', label='Max Temperature')

    # Common settings for the plot
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Temperature (°F)', fontsize=14)
    plt.title('Temperature Analysis Over Time', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    # Set y-axis ticks from 0 to 200 with an interval of 10
    plt.yticks(range(0, 201, 10), fontsize=12)
    plt.tight_layout()
    plt.show()

# Provide the path to

# Provide the path to your CSV file here
csv_filename = "/home/cam/st_heat/Experiments_04_17/140_150_160_Normal_Al_15_30_flow_full_400C_ts_160_14_3_twice_density(10,5)_Defined_Motion_vel_0.1.csv"  # Replace with the path to your CSV file
plot_csv_data(csv_filename)
