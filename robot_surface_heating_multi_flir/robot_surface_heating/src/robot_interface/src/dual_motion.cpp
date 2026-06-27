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
#include <array>

#define MEMBER_COUNT 6

/* deprecated */
// void load_member(void *target, void *donor, auto const& node) {
//     using namespace std;
//     for(std::size_t i = 0; i < MEMBER_COUNT; i++) {
//         RCLCPP_INFO(node->get_logger(), "transferring... "); 
//         static_cast<char *>(target)[i] = static_cast<char *>(donor)[i];
//         RCLCPP_INFO(node->get_logger(), "transferred"); 
//     }
// }

/* load ctypes bindings into a JointState message object */
void load_joint_state(sensor_msgs::msg::JointState &target, const char **names, const float *pos, const float *vel, const float *eff, auto const& node) {
    using namespace std;
    for(size_t i = 0; i < MEMBER_COUNT; i++) {
        // RCLCPP_INFO(node->get_logger(), "transferring names %li %s", i, names[i]);
        target.name[i] = names[i];
        // RCLCPP_INFO(node->get_logger(), "transferring pos %lf", pos[i]);
        target.position[i] = pos[i];
        // RCLCPP_INFO(node->get_logger(), "transferring vel %lf", vel[i]);
        target.velocity[i] = vel[i];
        // RCLCPP_INFO(node->get_logger(), "transferring eff %lf", eff[i]);
        target.effort[i] = eff[i];
    }
}

// extern "C" {
//   __declspec(dllexport) void __cdecl dual_ur_motion(int argc, char** argv, const geometry_msgs::msg::Pose& target_pose_0, 
extern "C" {
    int dual_ur_motion(const char** joint_names_0, const float* joint_pos_0, const float* joint_vel_0, const float* joint_eff_0,
    const char** joint_names_1, const float* joint_pos_1, const float* joint_vel_1, const float* joint_eff_1, const char* both_mg, const char* planner) {

        // setup
        rclcpp::init(0, nullptr);
        auto const node = std::make_shared<rclcpp::Node>("dual_motion_pose_goal");

        using moveit::planning_interface::MoveGroupInterface;
        auto move_group_interface = MoveGroupInterface(node, both_mg);

        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        std::thread([&executor]() { executor.spin(); }).detach();

        // extract joint state data
        sensor_msgs::msg::JointState target_js_0 = sensor_msgs::msg::JointState();
        target_js_0.name = std::vector<std::string>(MEMBER_COUNT);
        target_js_0.position = std::vector<double>(MEMBER_COUNT);
        target_js_0.velocity = std::vector<double>(MEMBER_COUNT);
        target_js_0.effort = std::vector<double>(MEMBER_COUNT);
        sensor_msgs::msg::JointState target_js_1 = sensor_msgs::msg::JointState();
        target_js_1.name = std::vector<std::string>(MEMBER_COUNT);
        target_js_1.position = std::vector<double>(MEMBER_COUNT);
        target_js_1.velocity = std::vector<double>(MEMBER_COUNT);
        target_js_1.effort = std::vector<double>(MEMBER_COUNT);
        RCLCPP_INFO(node->get_logger(), "transferring 0 state");
        load_joint_state(target_js_0, joint_names_0, joint_pos_0, joint_vel_0, joint_eff_0, node);
        RCLCPP_INFO(node->get_logger(), "transferring 1 state");
        load_joint_state(target_js_1, joint_names_1, joint_pos_1, joint_vel_1, joint_eff_1, node);

        // create dual motion plan
        if(strcmp(planner, "PTP") == 0 || strcmp(planner, "LIN") == 0 || strcmp(planner, "CIRC") == 0) {
            // planner is from Pilz
            move_group_interface.setPlanningPipelineId("pilz_industrial_motion_planner");
            move_group_interface.setPlannerId(planner);

            // set joint goal constraints
            move_group_interface.setJointValueTarget(target_js_0);
            move_group_interface.setJointValueTarget(target_js_1);
        } else {
            // planner is from OMPL
            move_group_interface.setPlanningPipelineId("ompl");
            move_group_interface.setPlannerId(planner);

            // set joint goal targets
            move_group_interface.setJointValueTarget(target_js_0);
            move_group_interface.setJointValueTarget(target_js_1);
            RCLCPP_INFO(node->get_logger(), "joint targets set"); 
        }
        move_group_interface.setPlanningTime(30.0);

        RCLCPP_INFO(node->get_logger(), "motion planning configured"); 

        // for(const auto& elem : move_group_interface.getPlannerParams("RRTstarkConfigDefault", move_group_interface.getName())) {
        //     RCLCPP_INFO(node->get_logger(), "%s %s", elem.first.c_str(), elem.second.c_str());
        // }

        // execute motion plan
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        auto result_code = move_group_interface.plan(my_plan);
        // bool success = (move_group_interface.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if(result_code == moveit::core::MoveItErrorCode::SUCCESS) {
            // for(const auto& elem : move_group_interface.getPlannerParams("geometric::RRTConnect", move_group_interface.getName())) {
            //     RCLCPP_INFO(node->get_logger(), "%s %s", elem.first.c_str(), elem.second.c_str());
            // }
            move_group_interface.execute(my_plan);
        } else {
            // RCLCPP_INFO(node->get_logger(), "planning failed: %d", result_code.val); 
            // RCLCPP_ERROR(node->get_logger(), "planning failed after %f: %d", my_plan.planning_time_, result_code.val);
            // rclcpp::shutdown();
            // return result_code.val;
        }

        // shutdown
        rclcpp::shutdown();
        return result_code.val;
    }

    void dual_ur_motion_old(const float* target_pose_0_pos, const float* target_pose_0_ort,
        const float* target_pose_1_pos, const float* target_pose_1_ort, const char *arm_0_mg, const char *arm_1_mg,
        const char* both_mg) {

        // setup
        rclcpp::init(0, nullptr);
        auto const node = std::make_shared<rclcpp::Node>("dual_motion_pose_goal");

        using moveit::planning_interface::MoveGroupInterface;
        auto move_group_interface = MoveGroupInterface(node, both_mg);
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
            RCLCPP_ERROR(node->get_logger(), "arm 0 fk not found");
            return;
        }


        // request->header = message_0.header;
        // request->header.frame_id = "world";
        // request->fk_link_names.push_back("arm_0_tool0");
        // request->robot_state.joint_state = message;

        // auto result = client->async_send_request(request);

        // get ur10e current joint state
        while(message_1.name.size() != MEMBER_COUNT) {
            found = rclcpp::wait_for_message(message_1, node,"/arm_1/joint_states",std::chrono::seconds(2));
            if(!found) {
                RCLCPP_ERROR(node->get_logger(), "arm 1 fk not found");
                return;
            }
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
        auto start_state = moveit_msgs::msg::RobotState();
        start_state.joint_state = message_0;
        // RCLCPP_INFO(node->get_logger(), "pre >>>> length of state: %li", start_state.joint_state.name.size()); 
        // for(std::string joint_name : start_state.joint_state.name)
        //     RCLCPP_INFO(node->get_logger(), "%s", joint_name.c_str());
        start_state.joint_state.name.insert(start_state.joint_state.name.end(), message_1.name.begin(), message_1.name.end());
        start_state.joint_state.position.insert(start_state.joint_state.position.end(), message_1.position.begin(), message_1.position.end());
        start_state.joint_state.velocity.insert(start_state.joint_state.velocity.end(), message_1.velocity.begin(), message_1.velocity.end());
        start_state.joint_state.effort.insert(start_state.joint_state.effort.end(), message_1.effort.begin(), message_1.effort.end());

        // RCLCPP_INFO(node->get_logger(), "pose >>>> length of state: %li", start_state.joint_state.name.size()); 
        // for(std::string joint_name : start_state.joint_state.name)
        //     RCLCPP_INFO(node->get_logger(), "%s", joint_name.c_str());

        // // convert data from python binding to target poses
        // geometry_msgs::msg::PoseStamped target_pose_0 = geometry_msgs::msg::PoseStamped();
        // geometry_msgs::msg::PoseStamped target_pose_1 = geometry_msgs::msg::PoseStamped();

        // target_pose_0.header.frame_id = "world";
        // target_pose_0.header.stamp = node->get_clock()->now();
        // target_pose_0.pose.position.x = target_pose_0_pos[0];
        // target_pose_0.pose.position.y = target_pose_0_pos[1];
        // target_pose_0.pose.position.z = target_pose_0_pos[2];
        // target_pose_0.pose.orientation.x = target_pose_0_ort[0];
        // target_pose_0.pose.orientation.y = target_pose_0_ort[1];
        // target_pose_0.pose.orientation.z = target_pose_0_ort[2];
        // target_pose_0.pose.orientation.w = target_pose_0_ort[3];

        // target_pose_1.header.frame_id = "world";
        // target_pose_1.header.stamp = node->get_clock()->now();
        // target_pose_1.pose.position.x = target_pose_1_pos[0];
        // target_pose_1.pose.position.y = target_pose_1_pos[1];
        // target_pose_1.pose.position.z = target_pose_1_pos[2];
        // target_pose_1.pose.orientation.x = target_pose_1_ort[0];
        // target_pose_1.pose.orientation.y = target_pose_1_ort[1];
        // target_pose_1.pose.orientation.z = target_pose_1_ort[2];
        // target_pose_1.pose.orientation.w = target_pose_1_ort[3];

        // auto request_0 = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
        // auto ik_request_0 = moveit_msgs::msg::PositionIKRequest();
        // ik_request_0.group_name = arm_0_mg;
        // ik_request_0.robot_state = start_state;
        // ik_request_0.avoid_collisions = false;
        // ik_request_0.ik_link_names = std::vector<std::string>{"arm_0_tool0", "tool0"};
        // ik_request_0.pose_stamped_vector = std::vector<geometry_msgs::msg::PoseStamped>{target_pose_0, target_pose_1};
        // // ik_request_0.timeout = rclcpp::Duration();
        // request_0->ik_request = ik_request_0;

        // auto result_0 = ik_client->async_send_request(request_0);
        // if (rclcpp::spin_until_future_complete(node, result_0) == rclcpp::FutureReturnCode::SUCCESS) {
        //     move_group_interface.setJointValueTarget(result_0.get()->solution.joint_state);
        // } else {
        //     RCLCPP_ERROR(node->get_logger(), "ik solution not found");
        //     return;
        // }

        // move_group_interface.setStartState(start_state);
        RCLCPP_INFO(node->get_logger(), "start state set"); 

        // convert data from python binding to target poses
        geometry_msgs::msg::Pose target_pose_0 = geometry_msgs::msg::Pose();
        geometry_msgs::msg::Pose target_pose_1 = geometry_msgs::msg::Pose();
        target_pose_0.position.x = target_pose_0_pos[0];
        target_pose_0.position.y = target_pose_0_pos[1];
        target_pose_0.position.z = target_pose_0_pos[2];
        target_pose_0.orientation.x = target_pose_0_ort[0];
        target_pose_0.orientation.y = target_pose_0_ort[1];
        target_pose_0.orientation.z = target_pose_0_ort[2];
        target_pose_0.orientation.w = target_pose_0_ort[3];
        target_pose_1.position.x = target_pose_1_pos[0];
        target_pose_1.position.y = target_pose_1_pos[1];
        target_pose_1.position.z = target_pose_1_pos[2];
        target_pose_1.orientation.x = target_pose_1_ort[0];
        target_pose_1.orientation.y = target_pose_1_ort[1];
        target_pose_1.orientation.z = target_pose_1_ort[2];
        target_pose_1.orientation.w = target_pose_1_ort[3];

        RCLCPP_INFO(node->get_logger(), "target poses defined"); 


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
        // move_group_interface.setPlanningPipelineId("pilz_industrial_motion_planner");
        // move_group_interface.setPlannerId("LIN");
        move_group_interface.setPoseTarget(target_pose_0, "arm_0_tool0");
        move_group_interface.setPoseTarget(target_pose_1, "tool0");
        move_group_interface.setPlanningTime(60.0);

        RCLCPP_INFO(node->get_logger(), "motion planning configured"); 


        for(std::string joint_name : move_group_interface.getJointNames()) {
            RCLCPP_INFO(node->get_logger(), "%s", joint_name.c_str());
        }


        // execute motion plan
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;

        auto result_code = move_group_interface.plan(my_plan);
        // bool success = (move_group_interface.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if(result_code == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(node->get_logger(), "planning succeeded, duration taken: %f", my_plan.planning_time_);
            move_group_interface.execute(my_plan);
        } else {
            RCLCPP_ERROR(node->get_logger(), "planning failed after %f: %d", my_plan.planning_time_, result_code.val);
        }

        // shutdown
        rclcpp::shutdown();
    }
}