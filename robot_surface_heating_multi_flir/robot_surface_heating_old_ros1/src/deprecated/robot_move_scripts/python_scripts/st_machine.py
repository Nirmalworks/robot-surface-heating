#!/usr/bin/env python3

import rospy
import numpy as np
import csv
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from transitions import Machine
import yaml

class HeatingStateMachine(object):
    states = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']

    def __init__(self):
        self.param_file = open('/home/cam/st_heat/src/param.yaml', 'r')
        self.parameters = yaml.safe_load(self.param_file)
        self.machine = Machine(model=self, states=HeatingStateMachine.states, initial='S1', after_state_change=['print_state', 'record_state'])
        self.target_temp = self.parameters['target_temp']
        self.buffer_temp = self.parameters['buffer_temp']
        self.switch_temp = self.parameters['switch_temp']
        self.state_time = []                    
        self.state_history = []
        self.start_time = rospy.Time.now().to_sec()
        self.csv_filename = self.parameters["csv_file_name_st_machine"]  # Define the log file name
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "State"])  # Write header

    def record_state(self):
        print(f"Current State: {self.state}")
        current_time = rospy.Time.now().to_sec() - self.start_time
        self.state_time.append(current_time)
        self.state_history.append(self.state)
        
    # Log the state transition and timestamp to a CSV file
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, self.state])  # Write current time and state                     

    
    def print_state(self):
        print(f"Current State: {self.state}")

class TempSubscriber:           
    def __init__(self):
        # Create a subscriber for the "/Temp_Array" topic
        self.temp_subscriber = rospy.Subscriber("/Temp_Array", Float32MultiArray, self.temp_callback)

        # Initialize temperature data
        self.temperature_data = None

    def temp_callback(self, data):
        # This function will be called every time a message is received on the "/Temp_Array" topic
        # Access the data from the received message
        self.temperature_data = data.data
        #rospy.loginfo("Received temperature data: {}".format(self.temperature_data))

    def get_temperature_data(self):
        # Function to get the latest temperature data
        return self.temperature_data

class EE_nodeSubscriber:

    def __init__(self):
        # Create a subscriber for the "/Temp_Array" topic
        self.EE_node_subscriber = rospy.Subscriber("/EE_node", Int32MultiArray, self.EE_node_callback)
        self.EE_node_data = None

    def EE_node_callback(self, data):
        # This function will be called every time a message is received on the "/Temp_Array" topic
        # Access the data from the received message
        self.EE_node_data = data.data

    def get_EE_node_data(self):
        # Function to get the latest temperature data
        return self.EE_node_data
        


def main():
    rospy.init_node('heating_state_machine')
    state_machine = HeatingStateMachine()

    node_index_subscriber = EE_nodeSubscriber()
    temp_array_subscriber = TempSubscriber()

    rate = rospy.Rate(10)
    first_iteration = True
    while not rospy.is_shutdown():
        if first_iteration:
            node_index_subscriber = EE_nodeSubscriber()
            timeout = rospy.Time.now() + rospy.Duration(5)  # 5 seconds timeout
            while node_index_subscriber.get_EE_node_data() is None:
                rospy.sleep(0.1)  # Sleep for a short duration
                if rospy.Time.now() > timeout:
                    print("Timeout waiting for /EE_node data.")
                    return
            first_iteration = False  # Ensure this block doesn't execute again
            index_data = node_index_subscriber.get_EE_node_data()
        
        while True:
 
            data = temp_array_subscriber.get_temperature_data()


            # print(data)
            node_num_x = state_machine.parameters['node_num_x']
            node_num_y = state_machine.parameters['node_num_y']
            data_numpy = np.array(data)
            reshaped_data = data_numpy.reshape(node_num_x+2, node_num_y+2)
            heated_nodes_data = reshaped_data[1:-1, 1:-1] # get only the heated nodes 
            current_temps = heated_nodes_data
            flat_current_temps = heated_nodes_data.flatten()
            # print("Temp Array", current_temps) 
            
            # Categorize temperatures into three sets
            temps_0_to_target = {temp for temp in flat_current_temps if 0 < temp < state_machine.target_temp}
            temps_target_to_buffer = {temp for temp in flat_current_temps if state_machine.target_temp <= temp < state_machine.buffer_temp}
            
            if not temps_0_to_target:
                break


            x_index_adjusted = index_data[0] - 1        
            y_index_adjusted = index_data[1] - 1
            # print("X_index and Y_index", x_index_adjusted, y_index_adjusted)

            current_heated_node_temp = current_temps[x_index_adjusted][y_index_adjusted]
            # print("Current Heated Node Temp", current_heated_node_temp)


            if current_heated_node_temp < state_machine.target_temp and temps_0_to_target:
                state_machine.to_S2()
            elif state_machine.target_temp <= current_heated_node_temp < state_machine.buffer_temp and temps_0_to_target:
                state_machine.to_S3()
            elif state_machine.buffer_temp <=current_heated_node_temp < state_machine.switch_temp and temps_0_to_target:
                state_machine.to_S4()
            elif current_heated_node_temp >= state_machine.switch_temp and temps_0_to_target:
                state_machine.to_S5()
            
            # state_machine.print_state()


            # Condition to break the inner loop if state reaches S5
            # if state_machine.state == 'S4':
            #     print("Reached state S4, breaking the inner loop.")

            #     node_index_subscriber = EE_nodeSubscriber()
            #     timeout = rospy.Time.now() + rospy.Duration(5)   # 5 seconds timeout
            #     while node_index_subscriber.get_EE_node_data() is None:
            #         rospy.sleep(0.1)  # Sleep for a short duration
            #         if rospy.Time.now() > timeout:
            #             print("Timeout waiting for /EE_node data.")
            #             return

            #     index_data = node_index_subscriber.get_EE_node_data()
            #     print("Index data", index_data[:2])
            #     state_machine.to_S1()
            #     break

            if state_machine.state == 'S5':
                print("Reached state S5, breaking the inner loop.")

                
                node_index_subscriber = EE_nodeSubscriber()
                timeout = rospy.Time.now() + rospy.Duration(5)   # 5 seconds timeout
                while node_index_subscriber.get_EE_node_data() is None:
                    rospy.sleep(0.1)  # Sleep for a short duration
                    if rospy.Time.now() > timeout:
                        print("Timeout waiting for /EE_node data.")
                        return

                index_data = node_index_subscriber.get_EE_node_data()
                print("Index data", index_data[:2])
                state_machine.to_S1()
                break



        while not temps_0_to_target and temps_target_to_buffer:

            temps_target_to_buffer = {temp for temp in flat_current_temps if state_machine.target_temp <= temp < state_machine.buffer_temp}
            
            x_index_adjusted = index_data[0] - 1        
            y_index_adjusted = index_data[1] - 1
            # print("X_index and Y_index", x_index_adjusted, y_index_adjusted)

            current_heated_node_temp = current_temps[x_index_adjusted][y_index_adjusted]
            # print("Current Heated Node Temp", current_heated_node_temp)

            if state_machine.target_temp<= current_heated_node_temp < state_machine.buffer_temp and temps_target_to_buffer and not temps_0_to_target:
                state_machine.to_S7()
            elif state_machine.buffer_temp<= current_heated_node_temp < state_machine.switch_temp and temps_target_to_buffer and not temps_0_to_target :
                state_machine.to_S8()
            elif current_heated_node_temp >= state_machine.switch_temp and temps_target_to_buffer and not temps_0_to_target :
                state_machine.to_S9()
            
            # state_machine.print_state()
            
            # Condition to break the inner loop if state reaches S8
            if state_machine.state == 'S9':
                print("Reached state S9, breaking the inner loop.")

                node_index_subscriber = EE_nodeSubscriber()
                timeout = rospy.Time.now() + rospy.Duration(5)  # 5 seconds timeout
                while node_index_subscriber.get_EE_node_data() is None:
                    rospy.sleep(0.1)  # Sleep for a short duration
                    if rospy.Time.now() > timeout:
                        print("Timeout waiting for /EE_node data.")
                        return

                index_data = node_index_subscriber.get_EE_node_data()
                print("Index data", index_data[:2])
                
                state_machine.to_S6()
            

            data = temp_array_subscriber.get_temperature_data()
                                                                 
            # print(data)
            node_num_x = state_machine.parameters['node_num_x']
            node_num_y = state_machine.parameters['node_num_y']
            data_numpy = np.array(data)
            reshaped_data = data_numpy.reshape(node_num_x+2, node_num_y+2)
            heated_nodes_data = reshaped_data[1:-1, 1:-1] # get only the heated nodes 
            current_temps = heated_nodes_data
            flat_current_temps = heated_nodes_data.flatten()
            

            
            # Categorize temperatures into three sets
            temps_0_to_target = {temp for temp in flat_current_temps if 0 < temp < state_machine.target_temp}
            temps_target_to_buffer = {temp for temp in flat_current_temps if state_machine.target_temp <= temp < state_machine.buffer_temp}
            


        while not temps_target_to_buffer:

            temps_buffer_to_switch = {temp for temp in flat_current_temps if state_machine.buffer_temp <= temp < state_machine.switch_temp}
            
            x_index_adjusted = index_data[0] - 1        
            y_index_adjusted = index_data[1] - 1
            # print("X_index and Y_index", x_index_adjusted, y_index_adjusted)

            current_heated_node_temp = current_temps[x_index_adjusted][y_index_adjusted]
            # print("Current Heated Node Temp", current_heated_node_temp)

            if state_machine.buffer_temp<= current_heated_node_temp < state_machine.switch_temp and temps_buffer_to_switch and not temps_target_to_buffer:
                state_machine.to_S11()
            elif current_heated_node_temp >= state_machine.switch_temp and temps_buffer_to_switch and not temps_target_to_buffer:
                state_machine.to_S12()

            # state_machine.print_state()
            
            # Condition to break the inner loop if state reaches S8
            if state_machine.state == 'S12':
                print("Reached state S12, breaking the inner loop.")

                node_index_subscriber = EE_nodeSubscriber()
                timeout = rospy.Time.now() + rospy.Duration(5)  # 5 seconds timeout
                while node_index_subscriber.get_EE_node_data() is None:
                    rospy.sleep(0.1)  # Sleep for a short duration
                    if rospy.Time.now() > timeout:
                        print("Timeout waiting for /EE_node data.")
                        return

                index_data = node_index_subscriber.get_EE_node_data()
                print("Index data", index_data[:2])
                
                state_machine.to_S10()
            

            data = temp_array_subscriber.get_temperature_data()
                                                                 
            # print(data)
            node_num_x = state_machine.parameters['node_num_x']
            node_num_y = state_machine.parameters['node_num_y']
            data_numpy = np.array(data)
            reshaped_data = data_numpy.reshape(node_num_x+2, node_num_y+2)
            heated_nodes_data = reshaped_data[1:-1, 1:-1] # get only the heated nodes 
            current_temps = heated_nodes_data
            flat_current_temps = heated_nodes_data.flatten()
            
            # Categorize temperatures into three sets
            temps_target_to_buffer = {temp for temp in flat_current_temps if state_machine.target_temp <= temp < state_machine.buffer_temp}



        # Keep the program alive
        rate.sleep()  # Sleep to maintain the desired loop rate and allow ROS callbacks

if __name__ == '__main__':
    main()






    










            
    # def on_temp_msg_received(self, msg):
    #     # Evaluate current temperatures and transition to the appropriate state
    #     data = msg.data
    #     node_num = 5
    #     data_numpy = np.array(data)
    #     reshaped_data = data_numpy.reshape(node_num+2, node_num+2)
    #     heated_nodes_data = reshaped_data[1:-1, 1:-1] # get only the heated nodes 
    #     self.current_temps = heated_nodes_data
    #     self.flat_current_temps = heated_nodes_data.flatten()
    #     print("Temp Array", self.current_temps) 

    #     # Categorize temperatures into three sets
    #     self.temps_0_to_target = {temp for temp in self.flat_current_temps if 0 < temp < self.target_temp}
    #     print("0 to target: ", self.temps_0_to_target)
    #     self.temps_target_to_buffer = {temp for temp in self.flat_current_temps if self.target_temp <= temp < self.buffer_temp}
    #     self.temps_buffer_to_switch = {temp for temp in self.flat_current_temps if self.buffer_temp <= temp < self.switch_temp}

    
    # def determine_states(self, msg):

    #     x_index_adjusted = msg.data[0] - 1
    #     y_index_adjusted = msg.data[1] - 1
    #     print("X_index and Y_index", x_index_adjusted, y_index_adjusted)

    #     current_heated_node_temp = self.current_temps[x_index_adjusted][y_index_adjusted]
    #     print("Current Heated Node Temp", current_heated_node_temp)


    #     if all(temp < self.target_temp for temp in self.flat_current_temps):
    #         self.to_S1()
    #     elif current_heated_node_temp < self.target_temp and self.temps_0_to_target:
    #         self.to_S2()
    #     elif self.buffer_temp <= current_heated_node_temp < self.switch_temp and self.temps_0_to_target:
    #         self.to_S3()
    #     elif current_heated_node_temp >= self.switch_temp and self.temps_0_to_target:
    #         self.to_S4()
    #     elif self.buffer_temp < current_heated_node_temp <= self.switch_temp and self.temps_target_to_buffer and not self.temps_buffer_to_switch:
    #         self.to_S5()
    #     # elif all(self.target_temp <= temp < self.buffer_temp for temp in current_temps) and any(temp >= self.switch_temp for temp in current_temps):
    #     #     self.to_S6()
    #     # elif any(temp >= self.switch_temp for temp in current_temps):  # Continuation of S6 logic for sustained heating beyond switch temperature
    #     #     self.to_S7()
    #     # elif all(temp >= self.target_temp for temp in current_temps) and not any(temp < self.target_temp for temp in current_temps) and not any(temp >= self.buffer_temp for temp in current_temps):
    #     #     self.to_S8()

    