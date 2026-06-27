#!/usr/bin/env python
import rospy
import yaml
import numpy as np
from std_msgs.msg import String, Float32MultiArray, Int32MultiArray
from one_step_lookahead import OneStepLookahead
from heat_planning.srv import HeatNode
import os
import threading
import time
from finite_automata_bnb import BnBSearch, StateMachine

class policy:
    def __init__(self):
        self.heat_command_pub = rospy.Publisher('heat_command', Int32MultiArray, queue_size=1)
        self.zigzag_pub = rospy.Publisher('zigzag', Int32MultiArray, queue_size=1)
        self.temp_array_sub = rospy.Subscriber('temp_array_processed', Float32MultiArray, self.temp_array_callback,queue_size=1)

        self.best_node = None
        self.action_traj = []
        self.surface_index = -1
        self.steel_switch_temperature = 137
        self.air_switch_temperature = 122
        self.wood_switch_temperature = 122
        
        file = open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/param.yaml', 'r')
        self.parameters = yaml.safe_load(file)
        self.node_num_x = self.parameters['node_num_x']
        self.node_num_y = self.parameters['node_num_y']
        self.switch_temp = self.parameters['switch_temp']
        self.max_temp = self.parameters['max_temp']
        self.temp_data = np.array([])
        self.policy_type = self.parameters['policy_type']
        self.state_machine = StateMachine(120, 125, 130,140)
        self.one_step_lookahead = None
        self.heat_node_client = rospy.ServiceProxy('heat_node', HeatNode)
        if(self.policy_type == "greedy"):
            self.policy_method = self.greedy_method
        elif(self.policy_type == "one_step"):
            self.policy_method = self.one_step_method
            self.thread = threading.Thread(target=self.run_one_step_lookahead)
            self.thread.start()
            self.one_step_lookahead = OneStepLookahead(self.switch_temp)     
        elif(self.policy_type == "zigzag"):
            self.policy_method = self.zigzag
        elif(self.policy_type == "heat_cen"):
            self.policy_method = self.heat_center_method
        elif(self.policy_type == "bnb"):
    
            self.policy_method = self.bnb_method
            self.bnb_search = BnBSearch(110, 115, 130,140)
            self.thread = threading.Thread(target=self.run_bnb_lookahead)
            self.thread.start()

        self.curr_x_index = -1
        self.curr_y_index = -1
        # Start a separate thread that runs OneStepLookahead in a loop
        
        self.min_temps = None
        self.max_temps = None

        self.current_state = None

    def update_state_temp(self):
        temp_array =  self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
        if self.current_state is None:
            raise ValueError("Current state is None")
        self.current_state.update_temp_array(temp_array)

    def heat_one_node_callback(self, data):
        row = 0
        col = 2

        if(len(self.temp_data>0)):
            reshaped_data = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
            temp_data_formatted = reshaped_data[1:-1, 1:-1]
            msg = Int32MultiArray()
            
            if(data.data[0] == -1 and data.data[1] == -1):
                msg.data = [row, col]
                self.heat_command_pub.publish(msg)
            
            elif(temp_data_formatted[data.data[0], data.data[1]] > self.switch_temp):
                msg.data = [row, col, 1]
                self.heat_command_pub.publish(msg)  
            
            elif(temp_data_formatted[data.data[0], data.data[1]] < self.switch_temp):
                msg.data = [row, col, 2]
                self.heat_command_pub.publish(msg)  
    
    def one_step_callback(self):
        if(len(self.temp_data)>0):
            # DON'T CHANGE THE PART BELOW
            reshaped_data = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
            temp_data_formatted = reshaped_data[1:-1, 1:-1]
            # DON'T CHANGE THE PART ABOVE

            # node_action = self.one_step_lookahead.get_node_to_heat(reshaped_data,(row_heated_curr_index, col_heated_curr_index))
            # self.best_node = node_action
            
            msg = Int32MultiArray()

            # msg.data = node_action
            # self.heat_command_pub.publish(msg)

            row_heated_curr_index = self.curr_x_index
            col_heated_curr_index = self.curr_y_index
            node_action, action_traj = self.one_step_lookahead.get_node_to_heat(reshaped_data,(row_heated_curr_index, col_heated_curr_index))
            self.best_node = node_action
            self.action_traj = action_traj
            return [node_action[0], node_action[1]]

            if(row_heated_curr_index == -1 and col_heated_curr_index == -1):
                node_action = self.one_step_lookahead.get_node_to_heat(reshaped_data,(row_heated_curr_index, col_heated_curr_index))
                # self.best_node = node_action
                msg.data = [node_action[0], node_action[1]]
            elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] > self.switch_temp):
                node_action = self.one_step_lookahead.get_node_to_heat(reshaped_data,(row_heated_curr_index, col_heated_curr_index))
                self.best_node = node_action
                msg.data = [node_action[0], node_action[1]]
            elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] < self.switch_temp):
                msg.data = [row_heated_curr_index, col_heated_curr_index]
            
            return msg.data
        
        else:
            return [-1, -1]
        
    def bnb_callback(self):
        if(len(self.temp_data)>0):
            try:
                # DON'T CHANGE THE PART BELOW
                temp_array = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
                temp_data_formatted = temp_array[1:-1, 1:-1]
                # DON'T CHANGE THE PART ABOVE

                # node_action = self.one_step_lookahead.get_node_to_heat(reshaped_data,(row_heated_curr_index, col_heated_curr_index))
                # self.best_node = node_action
                
                msg = Int32MultiArray()

                # msg.data = node_action
                # self.heat_command_pub.publish(msg)

                row_heated_curr_index = self.curr_x_index
                col_heated_curr_index = self.curr_y_index
                action_traj = self.bnb_search.get_node_to_heat_bnb(temp_array,(row_heated_curr_index, col_heated_curr_index))
                # self.best_node = node_action
                # self.action_traj = action_traj
                #add action_traj list to self.action_traj list
                self.action_traj =action_traj
                print("action_traj",self.action_traj)
                a = self.action_traj[0].input
                return [a[0], a[1]] 

                if(row_heated_curr_index == -1 and col_heated_curr_index == -1):
                    node_action = self.one_step_lookahead.get_node_to_heat(reshaped_data,(row_heated_curr_index, col_heated_curr_index))
                    # self.best_node = node_action
                    msg.data = [node_action[0], node_action[1]]
                elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] > self.switch_temp):
                    node_action = self.one_step_lookahead.get_node_to_heat(reshaped_data,(row_heated_curr_index, col_heated_curr_index))
                    self.best_node = node_action
                    msg.data = [node_action[0], node_action[1]]
                elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] < self.switch_temp):
                    msg.data = [row_heated_curr_index, col_heated_curr_index]
                
                return msg.data
            
            except Exception as e:
                print(e)
                # os._exit(1)
            
        else:
            return [-1, -1]
        
    

    def one_step_method(self):
        if(len(self.temp_data)<=0):
            return [-1, -1]
        reshaped_data = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
        temp_data_formatted = reshaped_data[1:-1, 1:-1]
        
        row_heated_curr_index = self.curr_x_index
        col_heated_curr_index = self.curr_y_index
        
        # print(temp_data_formatted[row_heated_curr_index, col_heated_curr_index])

        if(row_heated_curr_index == -1 and col_heated_curr_index == -1):
            compute_flag = True
        elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] > self.switch_temp):
            # self.heat_node_client(-2,-2)
            compute_flag = True
        elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] < self.switch_temp):
            compute_flag = False
        
        if compute_flag:
            if len(self.action_traj) != 0:
                # action = self.best_node
                # self.best_node = None
                #pop first element from action_traj
                action = self.action_traj.pop(0)
                return [action[0], action[1]]
            else:
                print("computing best node")
                min_action = self.one_step_callback()
                return min_action
        else:
            return [row_heated_curr_index, col_heated_curr_index]
        
    def bnb_method(self):
        if(len(self.temp_data)<=0):
            return [-1, -1]
        reshaped_data = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
        temp_data_formatted = reshaped_data[1:-1, 1:-1]
        
        row_heated_curr_index = self.curr_x_index
        col_heated_curr_index = self.curr_y_index
        
        # print(temp_data_formatted[row_heated_curr_index, col_heated_curr_index])

        

        # if(row_heated_curr_index == -1 and col_heated_curr_index == -1):
        #     compute_flag = True
        # elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] > self.switch_temp):
        #     # self.heat_node_client(-2,-2)
        #     compute_flag = True
        # elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] < self.switch_temp):
        #     compute_flag = False

        if(row_heated_curr_index == -1 and col_heated_curr_index == -1):
            compute_flag = True 
            print("cooling")
        elif(not self.current_state.is_state_valid()):
            # self.heat_node_client(-2,-2)
            print("invalid state")
            print("current_state",self.current_state.id)
            compute_flag = True
        elif(self.current_state.is_state_valid()):
            print("valid state")
            compute_flag = False

        if compute_flag:
            if len(self.action_traj) != 0:
            # if False:
                # action = self.best_node
                # self.best_node = None
                #pop first element from action_traj
                action = self.action_traj.pop(0)
                cmd = action.input
                print('executing action',cmd)
                # self.current_state = self.state_machine.get_next_state(self.current_state,action)
                self.current_state = self.state_machine.update_state(self.current_state.temp_array, cmd,cmd,0)
                return [cmd[0], cmd[1]]
            else:
                print("going greedy")
                min_action = self.greedy_method()
                self.current_state = self.state_machine.update_state(self.current_state.temp_array, min_action,min_action,0)
                return min_action
        else:
            return [row_heated_curr_index, col_heated_curr_index]

    def greedy_method(self,exploratory = False):
        if(len(self.temp_data)>0):
            # DON'T CHANGE THE PART BELOW
            reshaped_data = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
            temp_data_formatted = reshaped_data[1:-1, 1:-1]
            # DON'T CHANGE THE PART ABOVE            

            if self.min_temps is None:
                self.min_temps = temp_data_formatted
            else:
                self.min_temps = np.minimum(self.min_temps, temp_data_formatted)
            if self.max_temps is None:
                self.max_temps = temp_data_formatted
            else:
                self.max_temps = np.maximum(self.max_temps, temp_data_formatted)

            temp_range = abs(self.max_temps - self.min_temps)
            




            if exploratory:
                min_index = np.argmin(temp_range)
                min_index_2d = np.unravel_index(min_index, temp_range.shape)
            else:
                min_index = np.argmin(temp_data_formatted)
                min_index_2d = np.unravel_index(min_index, temp_data_formatted.shape)

            # print('current_x_y',self.curr_x_index, self.curr_y_index)

            msg = Int32MultiArray()

            row_heated_curr_index = self.curr_x_index
            col_heated_curr_index = self.curr_y_index
            
            # if(row_heated_curr_index == -1 and col_heated_curr_index == -1):
            #     msg.data = [min_index_2d[0], min_index_2d[1]]
            # elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] > self.switch_temp):
            #     msg.data = [min_index_2d[0], min_index_2d[1]]
            # elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] < self.switch_temp):
            #     msg.data = [row_heated_curr_index, col_heated_curr_index]

            if temp_data_formatted[min_index_2d[0], min_index_2d[1]] > self.max_temp:
                msg.data = [-1, -1]
            else:
                msg.data = [min_index_2d[0], min_index_2d[1]]

            # print('policy',msg.data)
            return msg.data
        
        else:
            return [-1, -1]

    def heat_center_method(self):
        if(len(self.temp_data)>0):
            reshaped_data = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
            temp_data_formatted = reshaped_data[1:-1, 1:-1]
            msg = Int32MultiArray()
            if(self.surface_index == -1):
                self.surface_index = 0
                msg.data = [0, 4]

            elif(self.surface_index == 0 and temp_data_formatted[0][4] > self.steel_switch_temperature):
                self.surface_index = 1
                msg.data = [5, 4]
    
            elif(self.surface_index == 1 and temp_data_formatted[5][4] > self.air_switch_temperature):
                self.surface_index = 2
                msg.data = [11, 4]

            elif(self.surface_index == 2 and temp_data_formatted[11][4] > self.wood_switch_temperature):
                self.surface_index = 0
                msg.data = [0, 4]
            return msg.data
    
    # def greedy_method_callback(self):
    #     if(len(self.temp_data)>0):
    #         # DON'T CHANGE THE PART BELOW
    #         reshaped_data = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
    #         temp_data_formatted = reshaped_data[1:-1, 1:-1]
    #         # DON'T CHANGE THE PART ABOVE            

    #         min_index = np.argmin(temp_data_formatted)
    #         min_index_2d = np.unravel_index(min_index, temp_data_formatted.shape)

    #         self.best_node = min_index_2d

    #     #     print('current_x_y',self.curr_x_index, self.curr_y_index)

    #     #     msg = Int32MultiArray()

    #     #     row_heated_curr_index = self.curr_x_index
    #     #     col_heated_curr_index = self.curr_y_index
            
    #     #     if(row_heated_curr_index == -1 and col_heated_curr_index == -1):
    #     #         msg.data = [min_index_2d[0], min_index_2d[1]]
    #     #     elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] > self.switch_temp):
    #     #         msg.data = [min_index_2d[0], min_index_2d[1]]
    #     #     elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] < self.switch_temp):
    #     #         msg.data = [row_heated_curr_index, col_heated_curr_index]

    #     #     print('policy',msg.data)
    #     #     return msg.data
        
    #     # else:
    #     #     return [-1, -1]


    def zigzag(self):
        return [-1, -1]
    
    def temp_array_callback(self, data):
        self.temp_data = np.array(data.data)

    def main_loop(self):
        rospy.spin()
    
    def run_one_step_lookahead(self):
        while not rospy.is_shutdown():
            self.best_node = self.one_step_callback()
    
    def run_bnb_lookahead(self):
        while not rospy.is_shutdown():
            self.best_node = self.bnb_callback()

if __name__ == '__main__':
    rospy.init_node('policy')
    policy_obj = policy()
    rate = rospy.Rate(10) # 10hz
    # send -2, -2 to heat_node service so that heat gun can be warmed up and then click enter to begin policy
    policy_obj.heat_node_client(-2, -2)
    temp_array =  policy_obj.temp_data.reshape(policy_obj.node_num_x+2, policy_obj.node_num_y+2)
    policy_obj.current_state = policy_obj.state_machine.update_state(temp_array, [-1,-1], [-1,-1], 0,init_state=True)
    input("Press Enter to begin policy")
    while not rospy.is_shutdown():
        if(policy_obj.policy_type != "zigzag"):
            policy_obj.update_state_temp()
            indices = policy_obj.policy_method()
            if(policy_obj.policy_type == "heat_cen"):
                if(len(indices) == 2):
                    print("here")
                    result = policy_obj.heat_node_client(indices[0], indices[1])

            elif not (indices[0] == policy_obj.curr_x_index and indices[1] == policy_obj.curr_y_index):
                st = time.time()
                result = policy_obj.heat_node_client(indices[0], indices[1])
                print("computed time",time.time()-st)
                policy_obj.curr_x_index = indices[0]
                policy_obj.curr_y_index = indices[1]
                print("sent")
            else:
                print(indices[0])
                print(indices[1])
                print(policy_obj.curr_x_index)
                print(policy_obj.curr_y_index)
                print("not sent")
                pass

        elif(policy_obj.policy_type == "zigzag"):
            result = policy_obj.heat_node_client(-1, -1)

        
        rate.sleep()
    if rospy.is_shutdown():
        policy_obj.heat_node_client(-2, -2)
        print("returning to cool position")
