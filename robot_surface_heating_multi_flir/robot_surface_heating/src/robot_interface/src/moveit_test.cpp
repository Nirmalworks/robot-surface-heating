#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/wait_for_message.hpp"
#include "sensor_msgs/msg/detail/laser_scan__struct.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <moveit_msgs/srv/get_position_fk.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/msg/robot_state.hpp>

#include <chrono>
#include <memory>
#include <vector>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_demo");


class ur_robot{
  private:
    std::string robot_name;
    std::string frame_name;
    std::string tcp_name;
    std::string joint_state_topic_name;
    std::shared_ptr<rclcpp::Node> node;
    rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedPtr fk_client;

  public:
    ur_robot(std::string robot_name_, std::string frame_name_, std::string tcp_name_, std::string joint_state_topic_name_){
      robot_name = robot_name_;
      frame_name = frame_name_;
      tcp_name = tcp_name_;
      joint_state_topic_name = joint_state_topic_name_;
      node = std::make_shared<rclcpp::Node>(robot_name+"_planning_node");
      fk_client = node->create_client<moveit_msgs::srv::GetPositionFK>("compute_fk");
    }

    sensor_msgs::msg::JointState getJointStates(){
      sensor_msgs::msg::JointState joint_states;
      bool found = rclcpp::wait_for_message(joint_states, node, joint_state_topic_name, std::chrono::seconds(1));

      return joint_states;
    }

    void waitUnitlArrival(geometry_msgs::msg::Pose goal_pose){
      bool arrived = 0;

    }
    
    void planAndExecute(geometry_msgs::msg::Pose goal_pose){

    }


};

// extern "C" {
//   __declspec(dllexport) void __cdecl dual_ur_motion(int argc, char** argv, const geometry_msgs::msg::Pose& target_pose_0, 
//     const geometry_msgs::msg::Pose& target_pose_1, const char* move_group) {

//     // setup
//     rclcpp::init(argc, argv);
//     auto const node = std::make_shared<rclcpp::Node>("dual_motion_pose_goal");

//     using moveit::planning_interface::MoveGroupInterface;
//     auto move_group_interface = MoveGroupInterface(node, move_group);

//     rclcpp::executors::SingleThreadedExecutor executor;
//     executor.add_node(node);
//     std::thread([&executor]() { executor.spin(); }).detach();

//     auto message_0 = sensor_msgs::msg::JointState();
//     auto message_1 = sensor_msgs::msg::JointState();

//     rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedPtr client =
//       node->create_client<moveit_msgs::srv::GetPositionFK>("compute_fk");

//     // get ur5 current joint state
//     bool found = rclcpp::wait_for_message(message_0, node,"/arm_0/joint_states",std::chrono::seconds(2));
//     // if(!found) {
      
//     // }
//     // auto request = std::make_shared<moveit_msgs::srv::GetPositionFK::Request>();
//     // request->header = message_0.header;
//     // request->header.frame_id = "world";
//     // request->fk_link_names.push_back("arm_0_tool0");
//     // request->robot_state.joint_state = message;

//     // auto result = client->async_send_request(request);

//     // get ur10e current joint state
//     found = rclcpp::wait_for_message(message_1, node,"/arm_1/joint_states",std::chrono::seconds(2));
//     // auto request = std::make_shared<moveit_msgs::srv::GetPositionFK::Request>();
//     // request->header = message.header;
//     // request->header.frame_id = "world";
//     // request->fk_link_names.push_back("tool0");
//     // request->robot_state.joint_state = message;

//     // auto result = client->async_send_request(request);

//     // set start state
//     // auto pose = result.get()->pose_stamped[0].pose;
//     // pose.position.z-=0.1;
//     // pose.position.y+=0.1;
//     auto start_state = moveit_msgs::msg::RobotState();
//     start_state.joint_state = message_0;
//     start_state.joint_state.name.insert(start_state.joint_state.name.end(), message_1.name.begin(), message_1.name.end());
//     start_state.joint_state.position.insert(start_state.joint_state.position.end(), message_1.position.begin(), message_1.position.end());
//     start_state.joint_state.velocity.insert(start_state.joint_state.velocity.end(), message_1.velocity.begin(), message_1.velocity.end());
//     start_state.joint_state.effort.insert(start_state.joint_state.effort.end(), message_1.effort.begin(), message_1.effort.end());

//     move_group_interface.setStartState(start_state);

//     // create dual motion plan
//     move_group_interface.setPlanningPipelineId("ompl");
//     move_group_interface.setPlannerId("RRTkConfigDefault");
//     move_group_interface.setPoseTarget(target_pose_0, "arm_0_tool0");
//     move_group_interface.setPoseTarget(target_pose_1, "tool0");
//     move_group_interface.setPlanningTime(5.0);

//     // execute motion plan
//     moveit::planning_interface::MoveGroupInterface::Plan my_plan;
//     bool success = (move_group_interface.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//     if(success) {
//       move_group_interface.execute(my_plan);
//     }

//     // shutdown
//     rclcpp::shutdown();
//   }
// }

// int main(int argc, char * argv[])
// {
//   // Initialize ROS and create the Node
//   rclcpp::init(argc, argv);
//   auto const node = std::make_shared<rclcpp::Node>("pose_goal");

//   // Create a ROS logger
//   auto const logger = rclcpp::get_logger("pose_goal");

//   // Create the MoveIt Move Group Interface for panda arm
//   using moveit::planning_interface::MoveGroupInterface;
//   // auto move_group_interface_ur10e = MoveGroupInterface(node, "ur10e");
//   auto move_group_interface_ur5 = MoveGroupInterface(node, "ur5");

//   rclcpp::executors::SingleThreadedExecutor executor;
//   executor.add_node(node);
//   std::thread([&executor]() { executor.spin(); }).detach();

//   auto message = sensor_msgs::msg::JointState();

//   rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedPtr client =
//     node->create_client<moveit_msgs::srv::GetPositionFK>("compute_fk");
  
//   rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client =
//     node->create_client<moveit_msgs::srv::GetPositionIK>("compute_ik");


//   bool found = rclcpp::wait_for_message(message, node,"/arm_0/joint_states",std::chrono::seconds(2));
//   auto request = std::make_shared<moveit_msgs::srv::GetPositionFK::Request>();
//   auto ik_request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
//   request->header = message.header;
//   request->header.frame_id = "world";
//   request->fk_link_names.push_back("arm_0_tool0");
//   request->robot_state.joint_state = message;

//   auto result = client->async_send_request(request);

//   auto pose = result.get()->pose_stamped[0].pose;
//   pose.position.z-=0.1;
//   pose.position.y+=0.1;
//   auto start_state = moveit_msgs::msg::RobotState();
//   start_state.joint_state = message;
//   move_group_interface_ur5.setStartState(start_state);

//   // geometry_msgs::msg::PoseStamped pose_stamp = result.get()->pose_stamped[0];

//   // before picking an object
//   geometry_msgs::msg::Pose before_pick_pose;
//   before_pick_pose.position.x = -0.448;
//   before_pick_pose.position.y = -0.159;
//   before_pick_pose.position.z = 0.227;

//   before_pick_pose.orientation.x = 0.183;
//   before_pick_pose.orientation.y = 0.982;
//   before_pick_pose.orientation.z = -0.02;
//   before_pick_pose.orientation.w = 0.015;

//   move_group_interface_ur5.setPlanningPipelineId("ompl");
//   move_group_interface_ur5.setPlannerId("RRTkConfigDefault");
//   move_group_interface_ur5.setPoseTarget(before_pick_pose);
//   move_group_interface_ur5.setPlanningTime(5.0);

//   moveit::planning_interface::MoveGroupInterface::Plan my_plan;
//   bool success = (move_group_interface_ur5.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//   if(success) {
//     move_group_interface_ur5.execute(my_plan);
//   }



//   // going down to pick an object
//   found = rclcpp::wait_for_message(message, node,"/arm_0/joint_states",std::chrono::seconds(2));
//   start_state.joint_state = message;
//   move_group_interface_ur5.setStartState(start_state);

//   geometry_msgs::msg::Pose pick_pose;
//   pick_pose.position.x = -0.448;
//   pick_pose.position.y = -0.159;
//   pick_pose.position.z = 0.038;

//   pick_pose.orientation.x = 0.183;
//   pick_pose.orientation.y = 0.982;
//   pick_pose.orientation.z = -0.02;
//   pick_pose.orientation.w = 0.015;

//   move_group_interface_ur5.setPlanningPipelineId("pilz_industrial_motion_planner");
//   move_group_interface_ur5.setPlannerId("LIN");
//   move_group_interface_ur5.setPoseTarget(pick_pose);
//   move_group_interface_ur5.setPlanningTime(5.0);
//   // moveit::planning_interface::MoveGroupInterface::Plan my_plan;
//   success = (move_group_interface_ur5.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//   if(success) {
//     move_group_interface_ur5.execute(my_plan);
//   }

//   auto wait_time = rclcpp::sleep_for(std::chrono::seconds(6));

//   // going up after picking an object
//   found = rclcpp::wait_for_message(message, node,"/arm_0/joint_states",std::chrono::seconds(2));
//   start_state.joint_state = message;
//   move_group_interface_ur5.setStartState(start_state);

//   geometry_msgs::msg::Pose after_pick_pose;
//   after_pick_pose.position.x = -0.448;
//   after_pick_pose.position.y = -0.159;
//   after_pick_pose.position.z = 0.17;

//   after_pick_pose.orientation.x = 0.183;
//   after_pick_pose.orientation.y = 0.982;
//   after_pick_pose.orientation.z = -0.02;
//   after_pick_pose.orientation.w = 0.015;

//   move_group_interface_ur5.setPlanningPipelineId("pilz_industrial_motion_planner");
//   move_group_interface_ur5.setPlannerId("LIN");
//   move_group_interface_ur5.setPoseTarget(after_pick_pose);
//   move_group_interface_ur5.setPlanningTime(6.0);
  
//   success = (move_group_interface_ur5.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//   if(success) {
//     move_group_interface_ur5.execute(my_plan);
//   }

//   wait_time = rclcpp::sleep_for(std::chrono::seconds(6));

//   // putting the object in the box
//   found = rclcpp::wait_for_message(message, node,"/arm_0/joint_states",std::chrono::seconds(2));
//   start_state.joint_state = message;
//   move_group_interface_ur5.setStartState(start_state);

//   geometry_msgs::msg::Pose putting_pose;
//   putting_pose.position.x = -0.6674;
//   putting_pose.position.y = -0.0678;
//   putting_pose.position.z = 0.354;

//   putting_pose.orientation.x = 0.0;
//   putting_pose.orientation.y = 1.0;
//   putting_pose.orientation.z = 0.0;
//   putting_pose.orientation.w = 0.0;

//   move_group_interface_ur5.setPlanningPipelineId("ompl");
//   move_group_interface_ur5.setPlannerId("RRTkConfigDefault");
//   move_group_interface_ur5.setPoseTarget(putting_pose);
//   move_group_interface_ur5.setPlanningTime(5.0);
  
//   success = (move_group_interface_ur5.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//   if(success) {
//     move_group_interface_ur5.execute(my_plan);
//   }

//   // Shutdown
//   rclcpp::shutdown();
//   return 0;
// }

int main(int argc, char** argv) {
  rclcpp::init(0, nullptr);
  auto const node = std::make_shared<rclcpp::Node>("dual_motion_pose_goal");

  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, "ur5");
  // auto arm_0_mgi = MoveGroupInterface(node, arm_0_mg);
  // auto arm_1_mgi = MoveGroupInterface(node, arm_1_mg);

  // rclcpp::executors::SingleThreadedExecutor executor;
  // executor.add_node(node);
  // std::thread([&executor]() { executor.spin(); }).detach();

  auto message_0 = sensor_msgs::msg::JointState();
  auto message_1 = sensor_msgs::msg::JointState();

  rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedPtr client =
      node->create_client<moveit_msgs::srv::GetPositionFK>("compute_fk");

  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client =
      node->create_client<moveit_msgs::srv::GetPositionIK>("compute_ik");

  // get ur5 current joint state
  bool found = rclcpp::wait_for_message(message_0, node,"/arm_0/joint_states",std::chrono::seconds(2));
  if(!found) {
      RCLCPP_INFO(node->get_logger(), "arm 0 fk not found"); 
  }
  // auto request = std::make_shared<moveit_msgs::srv::GetPositionFK::Request>();
  // request->header = message_0.header;
  // request->header.frame_id = "world";
  // request->fk_link_names.push_back("arm_0_tool0");
  // request->robot_state.joint_state = message;

  // auto result = client->async_send_request(request);

  // get ur10e current joint state
  found = rclcpp::wait_for_message(message_1, node,"/joint_states",std::chrono::seconds(2));
  if(!found) {
      RCLCPP_INFO(node->get_logger(), "arm 1 fk not found"); 
  }
  // auto request = std::make_shared<moveit_msgs::srv::GetPositionFK::Request>();
  // request->header = message.header;
  // request->header.frame_id = "world";
  // request->fk_link_names.push_back("tool0");
  // request->robot_state.joint_state = message;

  // auto result = client->async_send_request(request);

  // set start state
  // auto pose = result.get()->pose_stamped[0].pose;
  // pose.position.z-=0.1;
  // pose.position.y+=0.1;
  // auto start_state = moveit_msgs::msg::RobotState();
  // start_state.joint_state = message_0;
  // // start_state.joint_state.name.insert(start_state.joint_state.name.end(), message_1.name.begin(), message_1.name.end());
  // // start_state.joint_state.position.insert(start_state.joint_state.position.end(), message_1.position.begin(), message_1.position.end());
  // // start_state.joint_state.velocity.insert(start_state.joint_state.velocity.end(), message_1.velocity.begin(), message_1.velocity.end());
  // // start_state.joint_state.effort.insert(start_state.joint_state.effort.end(), message_1.effort.begin(), message_1.effort.end());

  // // move_group_interface.setStartState(start_state);
  // RCLCPP_INFO(node->get_logger(), "start state set"); 

  // // convert data from python binding to target poses
  geometry_msgs::msg::Pose target_pose_0 = geometry_msgs::msg::Pose();
  geometry_msgs::msg::Pose target_pose_1 = geometry_msgs::msg::Pose();
  // // target_pose_0.position.x = target_pose_0_pos[0];
  // // target_pose_0.position.y = target_pose_0_pos[1];
  // // target_pose_0.position.z = target_pose_0_pos[2];
  // // target_pose_0.orientation.x = target_pose_0_ort[0];
  // // target_pose_0.orientation.y = target_pose_0_ort[1];
  // // target_pose_0.orientation.z = target_pose_0_ort[2];
  // // target_pose_0.orientation.w = target_pose_0_ort[3];
  // // target_pose_1.position.x = target_pose_1_pos[0];
  // // target_pose_1.position.y = target_pose_1_pos[1];
  // // target_pose_1.position.z = target_pose_1_pos[2];
  // // target_pose_1.orientation.x = target_pose_1_ort[0];
  // // target_pose_1.orientation.y = target_pose_1_ort[1];
  // // target_pose_1.orientation.z = target_pose_1_ort[2];
  // // target_pose_1.orientation.w = target_pose_1_ort[3];
  target_pose_0.position.x = -0.50;
  target_pose_0.position.y = -0.0970;
  target_pose_0.position.z = 0.3036;
  target_pose_0.orientation.x = -0.00828499;
  target_pose_0.orientation.y = 0.998705;
  target_pose_0.orientation.z = 0.039051;
  target_pose_0.orientation.w = -0.0315452;

  target_pose_1.position.x = -0.82801; 
  target_pose_1.position.y = -0.17159;
  target_pose_1.position.z = 0.63211;
  target_pose_1.orientation.x = 0.66945;
  target_pose_1.orientation.y = 0.74259;
  target_pose_1.orientation.z = 0.017181;
  target_pose_1.orientation.w = -0.010503;

  RCLCPP_INFO(node->get_logger(), "target poses defined"); 

  // sensor_msgs::msg::JointState target_js_0 = sensor_msgs::msg::JointState();
  // sensor_msgs::msg::JointState target_js_1 = sensor_msgs::msg::JointState();
  // target_js_0.name = std::vector<std::string>{"arm_0_shoulder_pan_joint", "arm_0_shoulder_lift_joint", "arm_0_elbow_joint",
  //   "arm_0_wrist_1_joint", "arm_0_wrist_2_joint", "arm_0_wrist_3_joint"};
  // target_js_0.position = std::vector<double, std::allocator<double>>{0.008688189089298248, -1.5683868567096155, 1.5817065238952637,
  //   -1.6461947599994105, -1.5805700461017054, -1.6490934530841272};
  // target_js_0.velocity = std::vector<double, std::allocator<double>>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // target_js_0.effort = std::vector<double, std::allocator<double>>{0.020176295191049576, -0.02914353646337986, -0.04035259038209915, 0.0,
  //   0.0030500823631882668, 0.0030500823631882668};

  // target_js_1.name = std::vector<std::string>{"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
  //   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
  // target_js_1.position = std::vector<double, std::allocator<double>>{-0.03098367154598236, -1.1893862944892426, -1.9644802808761597,
  //   -1.4895161849311371, 1.617809772491455, -3.2563043276416224};
  // target_js_1.velocity = std::vector<double, std::allocator<double>>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // target_js_1.effort = std::vector<double, std::allocator<double>>{-0.11827291548252106, 0.02996097505092621, -0.04702407494187355, 0.005464488174766302,
  //   -0.135195791721344, 0.014206105843186378};

  // move_group_interface.setJointValueTarget(target_js_0);
  // move_group_interface.setJointValueTarget(target_js_1);

    // auto start_state = moveit_msgs::msg::RobotState();
    // start_state.joint_state = message_0;
    // start_state.joint_state.name.insert(start_state.joint_state.name.end(), message_1.name.begin(), message_1.name.end());
    // start_state.joint_state.position.insert(start_state.joint_state.position.end(), message_1.position.begin(), message_1.position.end());
    // start_state.joint_state.velocity.insert(start_state.joint_state.velocity.end(), message_1.velocity.begin(), message_1.velocity.end());
    // start_state.joint_state.effort.insert(start_state.joint_state.effort.end(), message_1.effort.begin(), message_1.effort.end());

    // convert data from python binding to target poses
    // geometry_msgs::msg::PoseStamped target_pose_0 = geometry_msgs::msg::PoseStamped();
    // geometry_msgs::msg::PoseStamped target_pose_1 = geometry_msgs::msg::PoseStamped();

    // target_pose_0.header.frame_id = "world";
    // target_pose_0.header.stamp = node->get_clock()->now();
    // target_pose_0.pose.position.x = -0.50;
    // target_pose_0.pose.position.y = -0.0970;
    // target_pose_0.pose.position.z = 0.3036;
    // target_pose_0.pose.orientation.x = -0.00828499;
    // target_pose_0.pose.orientation.y = 0.998705;
    // target_pose_0.pose.orientation.z = 0.039051;
    // target_pose_0.pose.orientation.w = -0.0315452;





    // target_pose_1.header.frame_id = "world";
    // target_pose_1.header.stamp = node->get_clock()->now();
    // target_pose_1.pose.position.x = -0.82801; 
    // target_pose_1.pose.position.y = -0.17159;
    // target_pose_1.pose.position.z = 0.63211;
    // target_pose_1.pose.orientation.x = 0.66945;
    // target_pose_1.pose.orientation.y = 0.74259;
    // target_pose_1.pose.orientation.z = 0.017181;
    // target_pose_1.pose.orientation.w = -0.010503;

    // auto request_0 = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    // auto ik_request_0 = moveit_msgs::msg::PositionIKRequest();
    // ik_request_0.group_name = "ur5";
    // ik_request_0.robot_state = start_state;
    // ik_request_0.avoid_collisions = false;
    // // ik_request_0.ik_link_names = std::vector<std::string>{"arm_0_tool0", "tool0"};
    // ik_request_0.ik_link_names = std::vector<std::string>{"arm_0_tool0"};
    // // ik_request_0.pose_stamped_vector = std::vector<geometry_msgs::msg::PoseStamped>{target_pose_0, target_pose_1};
    // ik_request_0.pose_stamped_vector = std::vector<geometry_msgs::msg::PoseStamped>{target_pose_0};
    // // ik_request_0.timeout = rclcpp::Duration();
    // request_0->ik_request = ik_request_0;

    // auto result_0 = ik_client->async_send_request(request_0);
    // if (rclcpp::spin_until_future_complete(node, result_0) == rclcpp::FutureReturnCode::SUCCESS) {
    //     move_group_interface.setJointValueTarget(result_0.get()->solution.joint_state);
    // } else {
    //     RCLCPP_ERROR(node->get_logger(), "ik solution not found");
    //     return 1;
    // }

    // for(int i = 0; i < result_0.get()->solution.joint_state.name.size(); i++) {
    //   RCLCPP_INFO(node->get_logger(), "name %s pos %d vel %d eff %d", result_0.get()->solution.joint_state.name[i],
    //     result_0.get()->solution.joint_state.position[i], result_0.get()->solution.joint_state.velocity[i], result_0.get()->solution.joint_state.effort[i]);
    // }

  // moveit::core::RobotStatePtr kinematic_state = move_group_interface.getCurrentState();

  // move_group_interface.setJointValueTarget(arm_0_joint_names, arm_0_joint_values);
  // move_group_interface.setJointValueTarget(arm_1_joint_names, arm_1_joint_values);

  // find IK solution for each arm separately
  // auto message = sensor_msgs::msg::JointState();
  // auto ik_request_0 = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
  // // ik_request_0->header = message.header;
  // // ik_request_0->header.frame_id = "arm_0_base_link";
  // // ik_request_0->ik_link_names.push_back("arm_0_tool0");
  // // ik_request_0->robot_state.joint_state = message;

  // auto ik_result_0 = client->async_send_request(ik_request_0);


  // find IK solution for each arm separately
  // auto kinematic_model = arm_0_mgi.getRobotModel();
  // moveit::core::RobotStatePtr kinematic_state = move_group_interface.getCurrentState();
  // const moveit::core::JointModelGroup* arm_0_joint_model_group = kinematic_model->getJointModelGroup(arm_0_mg);
  // const moveit::core::JointModelGroup* arm_1_joint_model_group = kinematic_model->getJointModelGroup(arm_1_mg);

  // std::vector<double> arm_0_joint_values, arm_1_joint_values;
  // const std::vector<std::string>& arm_0_joint_names = arm_0_joint_model_group->getVariableNames();
  // const std::vector<std::string>& arm_1_joint_names = arm_1_joint_model_group->getVariableNames();

  // double timeout = 0.1;
  // bool arm_0_found_ik = kinematic_state->setFromIK(arm_0_joint_model_group, target_pose_0, timeout);
  // bool arm_1_found_ik = kinematic_state->setFromIK(arm_1_joint_model_group, target_pose_1, timeout);

  // if (arm_0_found_ik && arm_1_found_ik)
  // {
  //   kinematic_state->copyJointGroupPositions(arm_0_joint_model_group, arm_0_joint_values);
  //   kinematic_state->copyJointGroupPositions(arm_1_joint_model_group, arm_1_joint_values);

  //   for (std::size_t i = 0; i < arm_0_joint_names.size(); ++i)
  //   {
  //     RCLCPP_INFO(node->get_logger(), "Joint %s: %f", arm_0_joint_names[i].c_str(), arm_0_joint_values[i]);
  //     RCLCPP_INFO(node->get_logger(), "Joint %s: %f", arm_1_joint_names[i].c_str(), arm_1_joint_values[i]);
  //   }
  // }
  // else
  // {
  //   RCLCPP_INFO(node->get_logger(), "Did not find IK solution");
  // }
  // move_group_interface.setJointValueTarget(arm_0_joint_names, arm_0_joint_values);
  // move_group_interface.setJointValueTarget(arm_1_joint_names, arm_1_joint_values);

  // create dual motion plan
  move_group_interface.setPlanningPipelineId("ompl");
  move_group_interface.setPlannerId("RRTkConfigDefault");
  move_group_interface.setPoseTarget(target_pose_0, "arm_0_tool0");
  // move_group_interface.setPoseTarget(target_pose_1, "tool0");
  move_group_interface.setPlanningTime(60.0);
  // move_group_interface.setStartStateToCurrentState();

  RCLCPP_INFO(node->get_logger(), "motion planning configured"); 


  // execute motion plan
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  auto result_code = move_group_interface.plan(my_plan);
  // bool success = (move_group_interface.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if(result_code == moveit::core::MoveItErrorCode::SUCCESS) {
    move_group_interface.execute(my_plan);
  } else {
      RCLCPP_INFO(node->get_logger(), "planning failed: %d", result_code.val); 
  }

  // shutdown
  rclcpp::shutdown();
}

// int main(int argc, char** argv) {
//     // Initialize ROS and create the Node
//     rclcpp::init(argc, argv);
//     auto const node = std::make_shared<rclcpp::Node>(
//       "hello_moveit",
//       rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
//     );

//     // Create a ROS logger
//     auto const logger = rclcpp::get_logger("hello_moveit");
//     RCLCPP_INFO(logger, "hello");

//     // Next step goes here
//     // Create the MoveIt MoveGroup Interface
//     using moveit::planning_interface::MoveGroupInterface;
//     auto move_group_interface = MoveGroupInterface(node, "ur5");

//     // Set a target Pose
//     auto const target_pose = []{
//       geometry_msgs::msg::Pose msg;
//       // msg.orientation.w = 1.0;
//       msg.orientation.x = -0.00828499;
//       msg.orientation.y = 0.998705;
//       msg.orientation.z = 0.039051;
//       msg.orientation.w = -0.0315452;
//       msg.position.x = -0.50;
//       msg.position.y = -0.0970;
//       msg.position.z = 0.3036;
//       return msg;
//     }();
//     move_group_interface.setPoseTarget(target_pose);

//     // Create a plan to that target pose
//     auto const [success, plan] = [&move_group_interface]{
//       moveit::planning_interface::MoveGroupInterface::Plan msg;
//       auto const ok = static_cast<bool>(move_group_interface.plan(msg));
//       return std::make_pair(ok, msg);
//     }();

//     // Execute the plan
//     if(success) {
//       move_group_interface.execute(plan);
//     } else {
//       RCLCPP_ERROR(logger, "Planning failed!");
//     }

//     // Shutdown ROS
//     rclcpp::shutdown();
//     return 0;
// }





// #include "geometry_msgs/msg/twist.hpp"
// #include "rclcpp/rclcpp.hpp"
// #include "rclcpp/wait_for_message.hpp"
// #include "sensor_msgs/msg/detail/laser_scan__struct.hpp"
// #include "sensor_msgs/msg/laser_scan.hpp"
// #include "wall_following_pkg/srv/find_wall.hpp"

// #include <chrono>
// #include <memory>

// using FindWall = wall_following_pkg::srv::FindWall;
// using std::placeholders::_1;
// using std::placeholders::_2;
// using namespace std::chrono_literals;

// const bool SIMULATION = true;

// class ServerNode : public rclcpp::Node {
// public:
//   ServerNode() : Node("find_wall_server") {
//     RCLCPP_INFO(this->get_logger(), "Robot simulated = %d", SIMULATION);
//     srv_ = create_service<FindWall>( "find_wall", std::bind(&ServerNode::find_wall_callback, this, _1, _2));
//     this->laser_left = 0.0;
//     this->laser_right = 0.0;
//     this->laser_forward = 0.0;
//     this->laser_backward = 0.0;
//   }

// private:
//   rclcpp::Service<FindWall>::SharedPtr srv_;
//   geometry_msgs::msg::Twist twist;
//   rclcpp::TimerBase::SharedPtr timer_;
//   float laser_left;
//   float laser_right;
//   float laser_forward;
//   float laser_backward;

//   void find_wall_callback(const std::shared_ptr<FindWall::Request> request,
//                           const std::shared_ptr<FindWall::Response> response) {
//     typeid(request).name(); // non intrusive function to avoid unused value warning

    

//     auto message = sensor_msgs::msg::LaserScan();

//     RCLCPP_INFO(this->get_logger(), "starting to wait for message");
//     bool scan_found = false;
//     while (!scan_found) {
//       scan_found = rclcpp::wait_for_message(message, this->shared_from_this(),"/scan",std::chrono::seconds(1));
//       RCLCPP_INFO(this->get_logger(), "scan_found = %d", scan_found);
//       RCLCPP_INFO(this->get_logger(), "still waiting for message");
//     }

//       this->laser_left = message.ranges[90];
//       this->laser_right = message.ranges[270];
//       this->laser_forward = message.ranges[0];
//       this->laser_backward = message.ranges[180];
//       RCLCPP_INFO(this->get_logger(), "[LEFT] = '%f'", this->laser_left);
//       RCLCPP_INFO(this->get_logger(), "[RIGHT] = '%f'", this->laser_right);
//       RCLCPP_INFO(this->get_logger(), "[FORWARD] = '%f'", this->laser_forward);
//       RCLCPP_INFO(this->get_logger(), "[BACKWARD] = '%f'", this->laser_backward);

    
//     response->wallfound = true;
//   }
// };

// int main(int argc, char *argv[]) {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<ServerNode>());
//   rclcpp::shutdown();
//   return 0;
// }