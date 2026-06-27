import csv
import matplotlib.pyplot as plt

def read_and_prepare_data(csv_filename):
    times = []
    state_changes = []
    
    with open(csv_filename, mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            times.append(float(row["Time"]))
            # Convert states from 'S1', 'S2', ... to 1, 2, ...
            state_changes.append(int(row["State"][1:]))
    
    return times, state_changes

def plot_state_changes(times, state_changes):
    plt.figure(figsize=(10, 6))
    plt.plot(times, state_changes, marker='o', linestyle='-', color='b')
    
    # Assuming the state changes are sequential and start from S1
    plt.yticks(range(1, max(state_changes) + 1), ['S' + str(i) for i in range(1, max(state_changes) + 1)])
    
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.title('State Changes Over Time')
    plt.grid(True)
    plt.show()

def main():
    # Update the path to your CSV file as needed
    csv_filename = "/home/cam/st_heat/state_csv/110_120_130_Normal_Al_15_flow_3_600C.csv"
    times, state_changes = read_and_prepare_data(csv_filename)
    plot_state_changes(times, state_changes)

if __name__ == '__main__':
    main()

