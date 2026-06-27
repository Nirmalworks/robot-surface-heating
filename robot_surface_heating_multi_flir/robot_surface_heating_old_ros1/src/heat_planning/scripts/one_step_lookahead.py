#!/usr/bin/env python
import rospy
from k_val_est_np import FEAProblem
from k_val_est_np import ThermModel
import numpy as np
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import yaml
import os
import copy

# State Class

# Action Class

# Import model parameters

# Create thermal simulator

# input should be current temperatures and current heated node

# output should be action to take

class State():
    def __init__(self, switch_temp):

        self.switch_temp = switch_temp

        # set of nodes that are below the switch temp
        self.nodes_below_switch_temp = set()
        # set of nodes that are above the switch temp
        self.nodes_above_switch_temp = set()
        
        self.current_heated_node = None

    def update_state(self,temp_array,current_heated_node):
        buffer_temp = 120
        temp_data_formatted = temp_array[1:-1, 1:-1]
        self.current_heated_node = current_heated_node

        below_indices = np.where(temp_data_formatted < self.switch_temp)
        above_indices = np.where(temp_data_formatted >= self.switch_temp)

        self.nodes_below_switch_temp = set(zip(below_indices[0], below_indices[1]))
        self.nodes_above_switch_temp = set(zip(above_indices[0], above_indices[1]))

        # buffer temperature is 120 degrees # get set of nodes above and below buffer temp
        buffer_below_indices = np.where(temp_data_formatted < buffer_temp)
        buffer_above_indices = np.where(temp_data_formatted >= buffer_temp)

        self.nodes_below_buffer_temp = set(zip(buffer_below_indices[0], buffer_below_indices[1]))
        self.nodes_above_buffer_temp = set(zip(buffer_above_indices[0], buffer_above_indices[1]))


class OneStepLookahead():
    def __init__(self, switch_temp):
        self.switch_temp = switch_temp
        current_directory = os.getcwd()
        
        with open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/heat_planning/scripts/model_sheet_params.yaml', 'r') as file:
            self.model_sheet_params = yaml.load(file, Loader=yaml.FullLoader)

        fea = FEAProblem(model_sheet_params=self.model_sheet_params)

        self.therm_model =ThermModel(fea, model_sheet_params=self.model_sheet_params)

        self.state = State(self.switch_temp)
        self.dt = 0.1

    def get_node_to_heat(self,temp_array,current_heated_node):
        # temp_data_formatted = temp_array[1:-1, 1:-1]
        # DON'T CHANGE THE PART ABOVE            


        self.state.update_state(temp_array, current_heated_node)
        
        # print(self.get_allowable_actions(self.state))

        row_heated_curr_index = current_heated_node[0]
        col_heated_curr_index = current_heated_node[1]
        
        # if(row_heated_curr_index != -1 and col_heated_curr_index != -1):
        #     cur_heated_node_ind = self.therm_model.graph_to_data_mapping[(8, 1,1)] -1 # NEED TO CHECK THIS
        #     # print(self.therm_model.graph_to_data_mapping)
        #     # print(current_heated_node)
        #     # print(cur_heated_node_ind)
        #     # simulate current heated node till it is past the switch temp
        #     temp = temp_array.flatten()
        #     temps = []
        #     temps.append(temp)
        #     k_mat = self.therm_model.compute_k_matrix(self.therm_model.model_parameters, cur_heated_node_ind)
        #     # print(k_mat)
        #     # print(temp)
        #     # input()
        #     while(temp[cur_heated_node_ind] < self.switch_temp):
        #         temp = self.therm_model.simulate_model_dt(temp,cur_heated_node_ind,self.therm_model.model_parameters,self.dt,k_mat)
        #         temps.append(temp)

        #     # print("done simulating")
        #     # print(temp)
            
        #     self.state.update_state(temp.reshape(self.therm_model.row,self.therm_model.col), current_heated_node)
        # else:
        #     temp = temp_array.flatten()
        #     temps = []
        #     temps.append(temp)
        actions = self.get_allowable_actions(self.state)
        # print(actions)
        # input()
        # loop through all possible actions and simulate each action to switch temp
        # dictionary ofpossible actions  to take and the mean squared error of temp
        action_error = {}
        # init_next_temp = copy.deepcopy(temp)
        init_next_temp = temp_array.flatten()
        lookahead_time = 15
        action_trajectories = {}
        for action in actions:
            action_trajectory = []
            action_trajectory.append(action)
            temp = copy.deepcopy(init_next_temp)
            temps = []
            temps.append(temp)
            action_ind = self.therm_model.graph_to_data_mapping[(action[0], action[1],1)] -1
            k_mat = self.therm_model.compute_k_matrix(self.therm_model.model_parameters, action_ind)
            time = 0
            # check if current action is action, if so then skip
            if(row_heated_curr_index == action[0] and col_heated_curr_index == action[1]):
                continue
            # while(temp[action_ind] < self.switch_temp):
            #     temp = self.therm_model.simulate_model_dt(temp,action_ind,self.therm_model.model_parameters,self.dt,k_mat)
            #     temps.append(temp)
            #     time += self.dt
            #     if time > lookahead_time:
            #         break

            while(time < lookahead_time):
                if temp[action_ind] > self.switch_temp:
                    # determine coldest node
                    action_ind = np.argmin(temp)
                    action_tup = self.therm_model.data_to_graph_mapping[action_ind+1]
                    action_trajectory.append((action_tup[0], action_tup[1]))
                    k_mat = self.therm_model.compute_k_matrix(self.therm_model.model_parameters, action_ind)
                temp = self.therm_model.simulate_model_dt(temp,action_ind,self.therm_model.model_parameters,self.dt,k_mat)
                temps.append(temp)
                time += self.dt
                    



            action_error[action] = np.mean(np.square(np.array(temps) - self.switch_temp))
            action_trajectories[action] = action_trajectory
        # get the action with the lowest mean squared error
        min_action = min(action_error, key=action_error.get)
        print(min_action)
        print()
        print(action_trajectories[min_action])
        # print(min_action)
        # print the min action current temperature
        # print(temp_array[min_action[0]+1, min_action[1]+1])
        # print(action_error)
        # input()
        return [min_action[0], min_action[1]] , action_trajectories[min_action]

        min_index = np.argmin(temp_data_formatted)
        min_index_2d = np.unravel_index(min_index, temp_data_formatted.shape)

        
        
        if(row_heated_curr_index == -1 and col_heated_curr_index == -1):
            return [min_index_2d[0], min_index_2d[1]]
        elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] > self.switch_temp):
            return [min_index_2d[0], min_index_2d[1]]
        elif(temp_data_formatted[row_heated_curr_index, col_heated_curr_index] < self.switch_temp):
            return [row_heated_curr_index, col_heated_curr_index]

  

    def get_allowable_actions(self,state):
        return state.nodes_below_buffer_temp
    
    