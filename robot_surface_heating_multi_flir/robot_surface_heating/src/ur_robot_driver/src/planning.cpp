#include <pluginlib/class_loader.hpp>

// MoveIt
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit_msgs/msg/planning_scene.h>
#include <moveit/move_group_interface/move_group_interface.h>


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const std::string PLANNING_GROUP = "ur10e";
    robot_model_loader::RobotModelLoader robot_model_loader(motion_planning_api_tutorial_node, "robot_description");
    const moveit::core::RobotModelPtr& robot_model = robot_model_loader.getModel();
    /* Create a RobotState and JointModelGroup to keep track of the current robot pose and planning group*/
    moveit::core::RobotStatePtr robot_state(new moveit::core::RobotState(robot_model));
    const moveit::core::JointModelGroup* joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);

    // Using the
    // :moveit_codedir:`RobotModel<moveit_core/robot_model/include/moveit/robot_model/robot_model.h>`,
    // we can construct a
    // :moveit_codedir:`PlanningScene<moveit_core/planning_scene/include/moveit/planning_scene/planning_scene.h>`
    // that maintains the state of the world (including the robot).
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    return 0;
}