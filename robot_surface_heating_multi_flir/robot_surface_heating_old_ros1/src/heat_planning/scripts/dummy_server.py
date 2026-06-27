#!/usr/bin/env python
# license removed for brevity
import rospy
from heat_planning.srv import HeatNode, HeatNodeResponse

def handle_add_two_ints(req):

    print(req.node_x_index)
    print(req.node_y_index)

    return HeatNodeResponse("done")

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('heat_node', HeatNode, handle_add_two_ints)
    print("Ready to add two ints.")
    rospy.spin() 
if __name__ == "__main__":
    while not rospy.is_shutdown():
        add_two_ints_server()