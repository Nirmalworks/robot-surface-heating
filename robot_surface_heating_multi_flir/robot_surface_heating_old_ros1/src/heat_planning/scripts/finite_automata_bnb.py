#!/usr/bin/env python
import rospy
from k_val_est_np_complex_h import FEAProblem
from k_val_est_np_complex_h import ThermModel
import numpy as np
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import yaml
import os
import copy
import math
import time
# State Class

# Action Class

# Import model parameters

# Create thermal simulator

# input should be current temperatures and current heated node

# output should be action to take

class State():
    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):
        
        self.min_temp = min_temp
        self.buff_temp = buff_temp
        self.switch_temp = switch_temp
        self.max_temp = max_temp
        # set of nodes that are below the switch temp
        self.nodes_below_min_temp = set()
        # set of nodes that are above the switch temp
        # self.nodes_above_switch_temp = set()
        
        # self.current_heated_node = None
        self.node_temps = None
        self.temp_array = None
        self.cmd_a = None # Current input
        self.a = None # Current Heating state
        self.t = None # Current time

    def initialize_state(self, temp_array, cmd_a, a, t_0 ):
        self.temp_array = temp_array
        self.node_temps = temp_array[1:-1, 1:-1] # Remove boundary nodes
        self.t_0 = t_0
        if cmd_a[0] == -1 and cmd_a[1] == -1:
            self.cmd_a = 'c'
        else:
            self.cmd_a = cmd_a
        if a[0] == -1 and a[1] == -1:
            self.a = 'c'
        else:
            self.a = a
        # self.temp_array = temp_array  

    def is_state_valid(self):
        return NotImplementedError
    
    def get_possible_actions(self):
        return NotImplementedError

    def update_temp_array(self, temp_array):
        self.temp_array = temp_array
        self.node_temps = temp_array[1:-1, 1:-1]

class S_0(State):
    """The starting state, where there are nodes below $T_{min}$ and no nodes are being heated."""
    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):
        
        super().__init__(min_temp, buff_temp, switch_temp,max_temp)

        self.id = "S_0"

        self.init_state = False
    
    def is_state_valid(self,skip_node_t_check=False):
        # Format temp array , current assumption is there is a boundary nodes
        node_temps = self.node_temps

        # Create set of nodes that are below the min temp
        below_min_indices = np.where(node_temps < self.min_temp)
        self.nodes_below_min_temp = set(zip(below_min_indices[0], below_min_indices[1]))

        # state condition is cmd_a = c, a = c, and there are nodes below the min temp
        if self.cmd_a == 'c' and self.a == 'c' and len(self.nodes_below_min_temp) == 0 and self.init_state == False:
            return True
        elif self.cmd_a == 'c' and self.a == 'c' and len(self.nodes_below_min_temp) != 0 and self.init_state == True:
            return True
        else:
            return False
    
    def explain_invalidity(self):
        if not self.cmd_a == 'c':
            return "node is being heated"
        if not self.a == 'c':
            return "node is being heated"
        if not len(self.nodes_below_min_temp) == 0:
            return "There are nodes below the min temp"
        return "State is valid"
        

class S_1(State):
    """ State where a node is being heated, and there are nodes below $T_{min}$. and the node being heated is below the buffer temp"""
    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):

        super().__init__(min_temp, buff_temp, switch_temp,max_temp)

        self.id = "S_1"

    def is_state_valid(self, skip_node_t_check=False):
        # Format temp array , current assumption is there is a boundary nodes
        node_temps = self.node_temps

        # Create set of nodes that are below the min temp
        below_min_indices = np.where(node_temps < self.min_temp)
        self.nodes_below_min_temp = set(zip(below_min_indices[0], below_min_indices[1]))

        

        if self.a == 'c' or self.a == 'tr':
            return False
        
        heated_node_temp = node_temps[self.a[0], self.a[1]] # Temperature of node being heated
        
        if not len(self.nodes_below_min_temp) != 0: # If no nodes below min temp
            return False
        

        if not (heated_node_temp < self.buff_temp) and not skip_node_t_check:
            return False
        
        return True
    
    def explain_invalidity(self):
        if self.a == 'c' or self.a == 'tr':
            return "Node is not being heated"
        
        heated_node_temp = self.node_temps[self.a[0], self.a[1]]

        if not (len(self.nodes_below_min_temp) != 0):
            return "There are no nodes below the min temp"
        if not (heated_node_temp < self.buff_temp):
            print("heated node temp",heated_node_temp)
            return "Node being heated is not below the buffer temperature"
        return "State is valid"

class S_2(State):
    """ State where a node is being heated, 
    and there are nodes below $T_{min}$, 
    and the node being heated is between the buffer and switch temperatures."""
    
    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):
        super().__init__(min_temp, buff_temp, switch_temp,max_temp)
        self.id = "S_2"

    def is_state_valid(self,skip_node_t_check=False):
        # Format temp array , current assumption is there is a boundary nodes
        node_temps = self.node_temps

        # Create set of nodes that are below the min temp
        below_min_indices = np.where(node_temps < self.min_temp)
        self.nodes_below_min_temp = set(zip(below_min_indices[0], below_min_indices[1]))

        

        if self.a == 'c' or self.a == 'tr':
            return False
        

        heated_node_temp = node_temps[self.a[0], self.a[1]]

        if not (len(self.nodes_below_min_temp) != 0):
            return False
        if not (heated_node_temp >= self.buff_temp and heated_node_temp < self.switch_temp) and not skip_node_t_check:
            return False
        
        return True

    def explain_invalidity(self):
        if self.a == 'c' or self.a == 'tr':
            return "Node is not being heated"
        
        heated_node_temp = self.node_temps[self.a[0], self.a[1]]

        if not (len(self.nodes_below_min_temp) != 0):
            return "There are no nodes below the min temp"
        if not (heated_node_temp >= self.buff_temp and heated_node_temp < self.switch_temp):
            print("heatednode temp",heated_node_temp)
            return "Node being heated is not between the buffer and switch temperature"
        return "State is valid"

class S_3(State):
    """ State where there are no nodes below the target temperature, 
    there are nodes between the target temperature and switch temperature, 
    and there is a node being heated that is below the buffer temperature."""

    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):
        super().__init__(min_temp, buff_temp, switch_temp,max_temp) 
        self.id = "S_3"
    
    def is_state_valid(self,skip_node_t_check=False):
        # Format temp array , current assumption is there is a boundary nodes
        node_temps = self.node_temps

        # Create set of nodes that are below the min temp
        below_min_indices = np.where(node_temps < self.min_temp)
        self.nodes_below_min_temp = set(zip(below_min_indices[0], below_min_indices[1]))

        

        if self.a == 'c' or self.a == 'tr':
            return False
        
        heated_node_temp = node_temps[self.a[0], self.a[1]]

        if not (len(self.nodes_below_min_temp) == 0) and not skip_node_t_check:
            return False
        if not (heated_node_temp < self.buff_temp) and not skip_node_t_check:
            return False
        
        return True

    def explain_invalidity(self):
        if self.a == 'c' or self.a == 'tr':
            return "Node is not being heated"
        
        heated_node_temp = self.node_temps[self.a[0], self.a[1]]

        if not (len(self.nodes_below_min_temp) == 0):
            return "There are nodes below the min temp"
        if not (heated_node_temp < self.buff_temp):
            print("heated node temp",heated_node_temp)
            return "Node being heated is not below the buffer temperature"
        return "State is valid"

class S_4(State):
    """State where there are no nodes below the target temperature, 
    there are nodes between the target temperature and switch temperature, 
    there is a node being heated, 
    and the node is between the buffer temperature and the switch temperature."""

    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):
        super().__init__(min_temp, buff_temp, switch_temp,max_temp)
        self.id = "S_4"
    
    def is_state_valid(self,skip_node_t_check=False):
        # Format temp array , current assumption is there is a boundary nodes
        node_temps = self.node_temps

        # Create set of nodes that are below the min temp
        below_min_indices = np.where(node_temps < self.min_temp)
        self.nodes_below_min_temp = set(zip(below_min_indices[0], below_min_indices[1]))

        

        if self.a == 'c' or self.a == 'tr':
            return False
        
        heated_node_temp = node_temps[self.a[0], self.a[1]]


        if not (len(self.nodes_below_min_temp) == 0):
            return False
        if not (heated_node_temp >= self.buff_temp and heated_node_temp < self.switch_temp) and not skip_node_t_check:
            return False
        
        return True
    
    def explain_invalidity(self):
        if self.a == 'c' or self.a == 'tr':
            return "Node is not being heated"
        
        heated_node_temp = self.node_temps[self.a[0], self.a[1]]

        if not (len(self.nodes_below_min_temp) == 0):
            return "There are nodes below the min temp"
        if not (heated_node_temp >= self.buff_temp and heated_node_temp < self.switch_temp):
            print("heated node temp",heated_node_temp)
            return "Node being heated is not between the buffer and switch temperature"
        return "State is valid"

class S_5(State):
    """The robot is transitioning from node $i$ to node $j$.
      $t_0$(the time the robot starts to transitioning)"""

    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):
        super().__init__(min_temp, buff_temp, switch_temp,max_temp)
        self.id = "S_5"
        self.travelled = False
    
    def travel_time(self,t_0,current_time):
        if self.travelled == False:
            self.travelled = True
            return False
        else:
            return True

    def is_state_valid(self,skip_node_t_check=False):
        
        if self.a == 'tr' and not self.travel_time(self.t_0, self.t):
            return True
        else:  
            return False
    
    def explain_invalidity(self):
        if not self.a == 'tr':
            return "Robot is not transitioning"
        if self.travel_time(self.t_0, self.t):
            return "Robot has finished transitioning"
        return "State is valid"



class Action():
    def __init__(self):
        pass

    def is_preconditions_valid(self):
        return NotImplementedError
    
class A_0(Action):
    """The starts heating a node"""
    def __init__(self):
        self.input = None

    def __str__(self):
        return f"A_0({self.input})"
    
    def __repr__(self):
        return self.__str__()

    def is_action_class_valid(self, state: State):
        if not state.a == 'c':
            return False
        else:
            return True

    def is_action_valid(self, state: State, cmd_a=None):

        if cmd_a is None and self.input is None:
            raise ValueError("Action input is not initialized")
        if cmd_a is None:
            cmd_a = self.input

        if not self.is_action_class_valid(state) :
            return False
        if len(state.nodes_below_min_temp) != 0 and cmd_a in state.nodes_below_min_temp:
            return True
        
        cmd_node_temp = state.node_temps[cmd_a[0], cmd_a[1]]

        if len(state.nodes_below_min_temp) == 0 and cmd_node_temp < state.switch_temp:
            return True
        return False
    
    def initialize_action(self, cmd_a):
        self.input = cmd_a
    
    def get_next_state_params(self,state: State):
        if self.input is None:
            raise ValueError("Action input is not initialized")
        cmd_a = self.input
        a = 'tr'
        return (cmd_a,a,0)

class A_1(Action):
    """The robot switches from node $i$ to node $j$"""
    def __init__(self):
        self.input = None

    def __str__(self):
        return f"A_1({self.input})"

    def __repr__(self):
        return self.__str__()
    
    def is_action_class_valid(self, state: State):
        if state.is_state_valid():
            return False
        else:
            return True
    
    def is_action_valid(self, state: State, cmd_a = None):

        if cmd_a is None and self.input is None:
            raise ValueError("Action input is not initialized")
        if cmd_a is None:
            cmd_a = self.input

        if not self.is_action_class_valid(state):
            return False
        if len(state.nodes_below_min_temp) != 0 and cmd_a in state.nodes_below_min_temp:
            return True
        cmd_node_temp = state.node_temps[cmd_a[0], cmd_a[1]]
        if len(state.nodes_below_min_temp) == 0 and cmd_node_temp < state.switch_temp:
            return True
        return False

    def initialize_action(self, cmd_a):
        self.input = cmd_a
    
    def get_next_state_params(self,state: State):
        if self.input is None:
            raise ValueError("Action input is not initialized")
        cmd_a = self.input
        a = 'tr'
        return (cmd_a,a,0)

class A_2(Action):
    """The robot stops heating the node"""
    def __init__(self):
        pass
    
    def __str__(self):
        return f"A_2()"
    
    def __repr__(self):
        return self.__str__()

    def is_action_class_valid(self, state: State):
        if state.id == "S_4" and state.is_state_valid():
            return True
        else:
            return False
    
    def is_action_valid(self, state: State, cmd_a='c'):
        if not self.is_action_class_valid(state):
            return False
        if cmd_a != 'c':
            return False
        return True
    
    def get_next_state_params(self,state: State):
        cmd_a = 'c'
        a = 'tr'
        return (cmd_a,a,0)

class Tr(Action):
    """The robot has finished transitioning from node $i$ to node $j$"""
    def __init__(self):
        pass

    def __str__(self):
        return f"Tr()"
    
    def __repr__(self):
        return self.__str__()

    def is_action_class_valid(self, state: State):
        if state.id == "S_5" and not state.is_state_valid():
            return True
        else:
            return False

    def is_action_valid(self, state: State):
        if not self.is_action_class_valid(state):
            return False
        return True
    
    def get_next_state_params(self,state: State):
        cmd_a = state.cmd_a
        a = cmd_a
        return (cmd_a,a,0)
        
        


class StateMachine():
    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):
        self.min_temp = min_temp
        self.buff_temp = buff_temp
        self.switch_temp = switch_temp
        self.max_temp = max_temp
        self.states = []
        self.actions = []
        self.current_state = None
        self.current_action = None
        self.current_time = None

    def intialize_current_state(self,temp_array, cmd_a, a, t_0):
        S_0_state = S_0(self.min_temp, self.buff_temp, self.switch_temp,self.max_temp)
        S_0_state.intialize_state(temp_array, cmd_a, a, t_0)
        if S_0_state.is_state_valid():
            self.current_state = S_0_state
            return self.current_state
        
        S_1_state = S_1(self.min_temp, self.buff_temp, self.switch_temp,self.max_temp)
        S_1_state.intialize_state(temp_array, cmd_a, a, t_0)
        if S_1_state.is_state_valid():
            self.current_state = S_1_state
            return self.current_state
        
        S_2_state = S_2(self.min_temp, self.buff_temp, self.switch_temp,self.max_temp)
        S_2_state.intialize_state(temp_array, cmd_a, a, t_0)
        if S_2_state.is_state_valid():
            self.current_state = S_2_state
            return self.current_state
        
        S_3_state = S_3(self.min_temp, self.buff_temp, self.switch_temp,self.max_temp)
        S_3_state.intialize_state(temp_array, cmd_a, a, t_0)
        if S_3_state.is_state_valid():
            self.current_state = S_3_state
            return self.current_state
        
        S_4_state = S_4(self.min_temp, self.buff_temp, self.switch_temp,self.max_temp)
        S_4_state.intialize_state(temp_array, cmd_a, a, t_0)
        if S_4_state.is_state_valid():
            self.current_state = S_4_state
            return self.current_state
        
        S_5_state = S_5(self.min_temp, self.buff_temp, self.switch_temp,self.max_temp, t_0)
        S_5_state.intialize_state(temp_array, cmd_a, a, t_0)
        if S_5_state.is_state_valid():
            self.current_state = S_5_state
            return self.current_state
    
    def update_state(self,temp_array, cmd_a, a, t_0, init_state = False,skip_node_t_check = False):
        state_classes = [S_0, S_1, S_2, S_3, S_4, S_5]
        for state_class in state_classes:
            state_instance = state_class(self.min_temp, self.buff_temp, self.switch_temp, self.max_temp)
            state_instance.initialize_state(temp_array, cmd_a, a, t_0)
            if state_instance.id == "S_0":
                state_instance.init_state = init_state
            if state_instance.is_state_valid(skip_node_t_check=skip_node_t_check):
                # self.current_state = state_instance
                return state_instance
        print("cmd",cmd_a)
        heated_node_temp = temp_array[cmd_a[0]+1, cmd_a[1]+1]
        print("heated node temp",heated_node_temp)
        raise ValueError("State is not valid")

    def get_possible_actions(self, state: State,ordering='coldest'):

    

        # if nodes below min temp, then get those nodes
        heatable_nodes = []
        possible_actions = []
        temp_data_formatted = state.node_temps

        if len(state.nodes_below_min_temp) != 0:
            heatable_nodes = state.nodes_below_min_temp
        else:            
            below_indices = np.where(temp_data_formatted < self.switch_temp)
            heatable_nodes = set(zip(below_indices[0], below_indices[1]))

        if ordering == 'coldest':
            heatable_nodes = sorted(heatable_nodes, key=lambda x: temp_data_formatted[x[0], x[1]])
        
        # if state is S_5, then return the node that the robot is transitioning to
        if state.id == "S_5":
            tr = Tr()
            if tr.is_action_class_valid(state):
                return [tr]
            else:
                raise ValueError("Transition action is not valid")
        
        # if state is S_0 then only action class is to start heating
        if state.id == "S_0":
            a_0 = A_0()
            if a_0.is_action_class_valid(state):
                for node in heatable_nodes:
                    if a_0.is_action_valid(state, node):
                        a_0.initialize_action(node)
                        possible_actions.append(copy.deepcopy(a_0))
                return possible_actions
            else:
                raise ValueError("Action is not valid")

        a_1 = A_1()

        if a_1.is_action_class_valid(state):
            for node in heatable_nodes:
                if a_1.is_action_valid(state, node):
                    a_1.initialize_action(node)
                    possible_actions.append(copy.deepcopy(a_1))
        
        a_2 = A_2()

        if a_2.is_action_class_valid(state):
            if a_2.is_action_valid(state, 'c'):
                possible_actions.append(copy.deepcopy(a_2))

        return possible_actions

    def get_next_state(self,state: State,action: Action):
        # check if action is valid
        if action.is_action_valid(state):
            state_params = action.get_next_state_params(state)
            new_state = self.update_state(state.temp_array,state_params[0], state_params[1], state_params[2])
            new_state.t = state.t
            return new_state
        else:
            # print(state.id)
            # print(action)
            raise ValueError("Action is not valid")
        

class BnBSearch():
    def __init__(self, min_temp, buff_temp, switch_temp,max_temp):

        self.min_temp = min_temp
        self.buff_temp = buff_temp
        self.switch_temp = switch_temp
        self.max_temp = max_temp

        # Load model parameters and get the thermal model
        current_directory = os.getcwd()
        with open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/heat_planning/scripts/model_sheet_params.yaml', 'r') as file:
            self.model_sheet_params = yaml.load(file, Loader=yaml.FullLoader)

        fea = FEAProblem(model_sheet_params=self.model_sheet_params)
        self.therm_model =ThermModel(fea, model_sheet_params=self.model_sheet_params)

        # self.state = State(self.switch_temp)
        self.dt = 0.1

        # initialize state machine
        self.state_machine = StateMachine(self.min_temp, self.buff_temp, self.switch_temp,self.max_temp)

        self.row_len = self.therm_model.row_len
        self.col_len = self.therm_model.col_len

    def get_cost(self, state: State):
        node_temps = state.node_temps

        # set cost to zero for each node temp if temp inbetween t min and t max else take mse
        cost = 0
        for i in range(node_temps.shape[0]):
            for j in range(node_temps.shape[1]):
                if node_temps[i,j] < self.min_temp:
                    cost += np.square(node_temps[i,j] - self.min_temp)
                elif node_temps[i,j] > self.max_temp:
                    cost += np.square(node_temps[i,j] - self.max_temp)
        cost = cost/(node_temps.shape[0]*node_temps.shape[1])
        return cost



    def get_node_to_heat(self,temp_array,current_heated_node):
        # temp_data_formatted = temp_array[1:-1, 1:-1]
        # DON'T CHANGE THE PART ABOVE            

        self.state = self.state_machine.update_state(temp_array, current_heated_node, current_heated_node, 0)

        # print(self.state.id)
        
        # # print(self.get_allowable_actions(self.state))

        row_heated_curr_index = current_heated_node[0]
        col_heated_curr_index = current_heated_node[1]
        
        # if(row_heated_curr_index != -1 and col_heated_curr_index != -1):
        #     cur_heated_node_ind = self.therm_model.graph_to_data_mapping[(8, 1,1)] -1 # NEED TO CHECK THIS
        #     # # print(self.therm_model.graph_to_data_mapping)
        #     # # print(current_heated_node)
        #     # # print(cur_heated_node_ind)
        #     # simulate current heated node till it is past the switch temp
        #     temp = temp_array.flatten()
        #     temps = []
        #     temps.append(temp)
        #     k_mat = self.therm_model.compute_k_matrix(self.therm_model.model_parameters, cur_heated_node_ind)
        #     # # print(k_mat)
        #     # # print(temp)
        #     # input()
        #     while(temp[cur_heated_node_ind] < self.switch_temp):
        #         temp = self.therm_model.simulate_model_dt(temp,cur_heated_node_ind,self.therm_model.model_parameters,self.dt,k_mat)
        #         temps.append(temp)

        #     # # print("done simulating")
        #     # # print(temp)
            
        #     self.state.update_state(temp.reshape(self.therm_model.row,self.therm_model.col), current_heated_node)
        # else:
        #     temp = temp_array.flatten()
        #     temps = []
        #     temps.append(temp)
        actions = self.get_allowable_actions(self.state)
        # print(len(actions))
        input()
        # loop through all possible actions and simulate each action to switch temp
        # dictionary ofpossible actions  to take and the mean squared error of temp
        action_error = {}
        # init_next_temp = copy.deepcopy(temp)
        init_next_temp = temp_array.flatten()
        lookahead_time = 5
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
        # print(min_action)
        # print()
        # print(action_trajectories[min_action])
        # # print(min_action)
        # # print the min action current temperature
        # # print(temp_array[min_action[0]+1, min_action[1]+1])
        # # print(action_error)
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


    def get_node_to_heat_bnb(self,temp_array,current_heated_node,action_traj=[],compute_time=5):

        # Initial Parameters
        t_h = 10 # Time horizon
        c_h = 5
        self.compute_time = compute_time    

        upper_bound = math.inf # solution upper bound

        a_seq = [] # Action sequence
        cost = 0
        best_a_seq = None

        self.best_a_seq = None
        self.upper_bound = math.inf
        
        self.start_time = time.time()
        if current_heated_node[0] == -1 and current_heated_node[1] == -1:
            state = self.state_machine.update_state(temp_array, current_heated_node, current_heated_node, 0, init_state=True)
            state.t = 0
        else:
            state = self.state_machine.update_state(temp_array, current_heated_node, current_heated_node, 0,skip_node_t_check=True)
            state.t = 0
            state,cost = self.simulate(state, 0, 100)
        # input("1")
        
        
        

        for action in action_traj:
            state,cost = self.simulate(state, 0, 100, action=action)
            state.t = 0
        state.t = 0
        # self.state = state
    
        upper_bound, best_a_seq = self.DFBNB(state, a_seq, best_a_seq, cost, upper_bound, t_h)
        return best_a_seq
        
    def DFBNB(self,state, a_seq, best_a_seq, cost, upper_bound, t_h):
        
        if state.t > t_h:
            if upper_bound > cost:
                upper_bound = cost
                best_a_seq = a_seq
            return upper_bound, best_a_seq
        if time.time() - self.start_time > self.compute_time:
            return upper_bound, best_a_seq
        # simulate state until invalid
        # state, cost = self.simulate(state,cost)

        # Get possible actions for invalid state
        actions = self.state_machine.get_possible_actions(state,ordering='coldest')
        # print('actions',actions)
        # input('testing actions')

        action = actions.pop(0)
        return 0 , [action]

        while len(actions) > 0:
            action = actions.pop(0)
            # print("actions",action)
            new_a_seq = copy.deepcopy(a_seq)
            new_a_seq = new_a_seq + [action] # Need to make a deep copy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            next_state, cost = self.simulate(state,cost,t_h,action=action)

            lower_bound = cost/t_h
            if lower_bound < upper_bound:
                upper_bound, best_a_seq = self.DFBNB(next_state, new_a_seq, best_a_seq, cost, upper_bound, t_h)
            else:
                # print('pruned')
                continue
        return upper_bound, best_a_seq

    def simulate(self,state,cost,t_h, action=None):
        if action is not None:
            state = self.state_machine.get_next_state(state,action)
        if state.a == 'tr':
            # get possible actions
            actions = self.state_machine.get_possible_actions(state,ordering='coldest')
            state = self.state_machine.get_next_state(state,actions[0])
            
        a = state.a
        if a == 'tr' or a == 'c':
            a = [-1,-1]

        action_ind = self.therm_model.graph_to_data_mapping[(a[0]+1, a[1]+1,1)] -1    
        k_mat = self.therm_model.compute_k_matrix(self.therm_model.model_parameters, action_ind)

        while state.is_state_valid() and state.t < t_h:
            
            
            temp_array_flatten = state.temp_array.flatten()
            temp_array_next_flatten = self.therm_model.simulate_model_dt(temp_array_flatten, action_ind, self.therm_model.model_parameters, self.dt,k_mat)
            temp_array_next = temp_array_next_flatten.reshape(state.temp_array.shape)
            state.update_temp_array(temp_array_next)
            state.t += self.dt
            cost += self.get_cost(state)
            # # print(state.id)
            # # print(temp_array_next[a[0]+1, a[1]+1])
            # # print(temp_array_next_flatten[action_ind])
            # # print()
            # # print()
        
        return state, cost
    
    


def main():
    bnb = BnBSearch(120, 125, 130,140)

    # print(bnb.row_len)
    # print(bnb.col_len)

    # create fake temp array of 75 deg
    temp_array = np.ones((bnb.row_len, bnb.col_len)) * 75

    # temp_array = np.ones((bnb.row_len, bnb.col_len)) * 130

    # change pne node to 100 deg
    # temp_array[5,5] = 110

    best_action_seq = bnb.get_node_to_heat_bnb(temp_array,[-1, -1])
    # print(best_action_seq)


if __name__ == "__main__":
    main()
