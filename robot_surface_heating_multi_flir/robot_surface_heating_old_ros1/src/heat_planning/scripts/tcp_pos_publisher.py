#!/usr/bin/python
import sys
import copy
import rospy
import moveit_commander

from std_msgs.msg import Float32
from geometry_msgs.msg import Point

if __name__ == "__main__":
    rospy.init_node("tcp_pos_publisher", anonymous=True)
    group_name = "yaskawa_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    z_publisher = rospy.Publisher("tcp_pos", Point, queue_size=10)

    while(rospy.is_shutdown() == False):
        tcp_pose = move_group.get_current_pose().pose.position
        z_publisher.publish(tcp_pose)

