import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


csv_name = "/home/cam/csv_data/140_150_160_Normal_Al_15_30_flow_full_400C_ts_160_9_1_EXP_2.csv"
df = pd.read_csv(csv_name)
# df.drop(index=0)
# df = df.iloc[113:185]

# df = df.iloc[:90]

# df.to_csv("data/data_april_9/processed_"+csv_name, index=False)

plt.figure(figsize=(10, 6))

for column in df.columns[1:26]:  # Skip the first (time) and last (heated_node) columns

    plt.plot(np.array(df['Time_Seconds']), np.array(df[column]), label="right", color='blue')

for column in df.columns[26:51]:  # Skip the first (time) and last (heated_node) columns

    plt.plot(np.array(df['Time_Seconds']), np.array(df[column]), label="left", color='green')

plt.xlabel('Time (seconds)')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # Move the legend outside the plot

plt.show()