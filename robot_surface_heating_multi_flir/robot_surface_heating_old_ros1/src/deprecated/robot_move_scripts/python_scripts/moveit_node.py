#!/usr/bin/python

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

## BEGIN_SUB_TUTORIAL imports
##
## To use the Python MoveIt interfaces, we will import the `moveit_commander`_ namespace.
## This namespace provides us with a `MoveGroupCommander`_ class, a `PlanningSceneInterface`_ class,
## and a `RobotCommander`_ class. More on these below. We also import `rospy`_ and some messages that we will use:
##

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import math
import random
import time
import csv
from std_msgs.msg import Float32MultiArray, Float32
import yaml



__REQUIRED_API_VERSION__ = "1"


try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
import array

from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import Pose

## END_SUB_TUTORIAL

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True

class EE_nodeSubscriber():

    Ox= 0
    Oy= 0
    Oz= 0
    Ow= -1
    z_pos_1= 0.3
    z_pos_2 = 0.5
    def __init__(self):
        # Create a subscriber for the "/Temp_Array" topic
        self.controller_state_pub = rospy.Publisher('controller_state', Int32MultiArray, queue_size=10)
        self.EE_node_data = None
        group_name = "yaskawa_arm"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.curr_state = None
        self.param_file = open('/home/cam/ND_ws/robot_surface_heating_ws/src/param.yaml', 'r')
        parameters = yaml.safe_load(self.param_file)
        node_num_x = parameters['node_num_x']
        node_num_y = parameters['node_num_y']

        x_start = parameters['x_start_mp']
        x_end = parameters['x_end_mp']
        y_start = parameters['y_start_mp']
        y_end = parameters['y_end_mp']
        x_spacing = (x_end-x_start)/(node_num_x+1)  # Adjust this to change the spacing between points
        y_spacing = (y_end-y_start)/(node_num_y+1)  # Adjust this to change the spacing between points

        x_values = np.arange(x_start, x_end, x_spacing)
        y_values = np.arange(y_end, y_start, -y_spacing)
        self.x_values = np.delete(x_values, 0)
        self.y_values = np.delete(y_values, 0)
        self.tcp_z_ehgith_curr = self.move_group.get_current_pose().pose.position.z
        self.scene = moveit_commander.PlanningSceneInterface()
        self.add_collision_table()
        self.policy_type = parameters['policy_type']
        if(self.policy_type=="greedy"):
            self.EE_node_subscriber = rospy.Subscriber("heat_command_to_control", Int32MultiArray, self.EE_node_callback)
        elif(self.policy_type=="zigzag"):
            self.EE_node_subscriber = rospy.Subscriber("heat_command_to_control", Int32MultiArray, self.zigzag_callback)
        self.zigzag_state = False

    def add_collision_table(self):
 
        table_id = "table"  # Unique identifier for the table in the planning scene.
        table_size = [1.5, 1, 0.05]  # Dimensions of the table (x, y, z)
        table_pose = geometry_msgs.msg.PoseStamped()
        table_pose.header.frame_id = "base_link"
        table_pose.pose.orientation.w = 1.0
        table_pose.pose.position.x = -0.1  # Position of the table in the x-direction
        table_pose.pose.position.y = -0.8 # Position of the table in the y-direction
        table_pose.pose.position.z = 0.2  # Position the table such that its base is on the ground.

        # Add the table as a collision object in the scene.
        self.scene.add_box(name=table_id, pose=table_pose, size=table_size)

    def EE_node_callback(self, data):
        # This function will be called every time a message is received on the "/Temp_Array" topic
        # Access the data from the received message
        if(data.data[0] != -1 and data.data[1] != -1):
            x_pos = self.x_values[data.data[0]]
            y_pos = self.y_values[data.data[1]]
            pose_curr = self.move_group.get_current_pose().pose
            if(self.curr_state != data.data):
                if(self.curr_state == None):
                    self.go_to_arm_goal(pose_curr.position.x, pose_curr.position.y, 0.5, EE_nodeSubscriber.Ox, EE_nodeSubscriber.Oy, EE_nodeSubscriber.Oz, EE_nodeSubscriber.Ow)
                    self.curr_state = data.data
                    self.go_to_arm_goal(x_pos, y_pos, EE_nodeSubscriber.z_pos_1, EE_nodeSubscriber.Ox, EE_nodeSubscriber.Oy, EE_nodeSubscriber.Oz, EE_nodeSubscriber.Ow)
                elif(self.curr_state != None):
                    euc_distance = math.sqrt((data.data[0]-self.curr_state[0])**2 + (data.data[1]-self.curr_state[1])**2)
                    if(euc_distance <=math.sqrt(3)):
                        self.go_to_arm_goal(x_pos, y_pos, EE_nodeSubscriber.z_pos_1, EE_nodeSubscriber.Ox, EE_nodeSubscriber.Oy, EE_nodeSubscriber.Oz, EE_nodeSubscriber.Ow)
                        self.curr_state = data.data
                    else:
                        self.go_to_arm_goal(pose_curr.position.x, pose_curr.position.y, 0.5, EE_nodeSubscriber.Ox, EE_nodeSubscriber.Oy, EE_nodeSubscriber.Oz, EE_nodeSubscriber.Ow)
                        self.curr_state = data.data
                        self.go_to_arm_goal(x_pos, y_pos, EE_nodeSubscriber.z_pos_1, EE_nodeSubscriber.Ox, EE_nodeSubscriber.Oy, EE_nodeSubscriber.Oz, EE_nodeSubscriber.Ow) 
    
    def zigzag_callback(self, data):
        if(self.zigzag_state == False):
            self.zigzag_state = True
            self.zigzag()
            self.zigzag_state = False
    
    def go_to_arm_goal(self,x,y,z,Ox,Oy,Oz,Ow):
        move_group = self.move_group
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('PTP')
        move_group.set_planning_time(10.0)
        move_group.set_max_velocity_scaling_factor(0.7)
        move_group.set_max_acceleration_scaling_factor(0.1)
        move_group.set_num_planning_attempts(10)
        move_group.set_goal_tolerance(0.01)

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z
        pose_goal.orientation.x = Ox
        pose_goal.orientation.y = Oy
        pose_goal.orientation.z = Oz
        pose_goal.orientation.w = Ow

        move_group.set_pose_target(pose_goal)
        move_group.plan()
        move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)
    
    def zigzag(self):
        x_last_index = len(self.x_values)-1
        y_last_index = len(self.y_values)-1
        waypoints = []
        z = 0.3
        y_reverse = True

        while(x_last_index >=0):
            if(y_reverse == True):
                waypoints.append((self.x_values[x_last_index], self.y_values[0], z))
                waypoints.append((self.x_values[x_last_index], self.y_values[-1], z))
                y_reverse = False
            elif(y_reverse == False):
                waypoints.append((self.x_values[x_last_index], self.y_values[-1], z))
                waypoints.append((self.x_values[x_last_index], self.y_values[0], z))
                y_reverse = True

            x_last_index-=1

        self.go_to_cartesian_waypoints(waypoints)
            
    def go_to_cartesian_waypoints(self, waypoints):
        move_group = self.move_group

        # Configure the motion planner
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')
        move_group.set_planning_time(10.0)
        move_group.set_max_velocity_scaling_factor(0.01)
        move_group.set_max_acceleration_scaling_factor(0.05)
        move_group.set_num_planning_attempts(10)
        move_group.set_goal_tolerance(0.01)

        # Initialize the waypoints list
        cartesian_waypoints = []

        # Iterate through each waypoint
        for waypoint in waypoints:
            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.position.x = waypoint[0]
            pose_goal.position.y = waypoint[1]
            pose_goal.position.z = waypoint[2]
            cartesian_waypoints.append(copy.deepcopy(pose_goal))

        last_waypoint = copy.deepcopy(cartesian_waypoints[-1])
        last_waypoint.position.z = 0.8
        cartesian_waypoints.append(last_waypoint)
        
        (plan, fraction) = move_group.compute_cartesian_path(cartesian_waypoints, 0.01, 0.0)
        curr_state = move_group.get_current_state()
        plan = move_group.retime_trajectory(curr_state,plan,velocity_scaling_factor= 0.1)

        if fraction == 1.0:
            rospy.loginfo("Executing the planned Cartesian path")
            move_group.execute(plan, wait=True)

        # Stop the movement
        move_group.stop()
        # Clear the targets
        move_group.clear_pose_targets()

    def main_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if(self.policy_type == "greedy"):
                if(self.curr_state == None):
                    print("none")
                    message = Int32MultiArray()
                    message.data = [-1,-1]
                    self.controller_state_pub.publish(message)
                elif(self.tcp_z_ehgith_curr >= 0.29 and self.tcp_z_ehgith_curr <= 0.31):
                    print("heating")
                    message = Int32MultiArray()
                    message.data = self.curr_state
                    self.controller_state_pub.publish(message)
                else:
                    print("flight")
                    message = Int32MultiArray()
                    message.data = [-1,-1]
                    self.controller_state_pub.publish(message)
                rate.sleep()
            elif(self.policy_type == "zigzag"):
                message = Int32MultiArray()
                message.data = [-5,-5]
                self.controller_state_pub.publish(message)

def main():
    rospy.init_node("controller", anonymous=True)
    Node_subscriber = EE_nodeSubscriber()
    Node_subscriber.main_loop()

if __name__ == "__main__":
    main()
