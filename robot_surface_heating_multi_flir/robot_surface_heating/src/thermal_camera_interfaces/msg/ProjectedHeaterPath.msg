std_msgs/Header header

float32 x_min
float32 y_min
float32 cell_size_m
uint32 nx
uint32 ny

geometry_msgs/Point center
geometry_msgs/Vector3 axis_u
geometry_msgs/Vector3 axis_v
geometry_msgs/Vector3 axis_n

geometry_msgs/Point start_proj
geometry_msgs/Point target_proj

geometry_msgs/Point[] control_proj
geometry_msgs/Point[] path_proj
geometry_msgs/Point[] path_cells

float32 final_cost
uint32 best_branch_index