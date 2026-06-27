#include <rclcpp/rclcpp.hpp>
#include <moveit/moveit_cpp/moveit_cpp.h>
#include <moveit/moveit_cpp/planning_component.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <iostream>
#include <string>

/*
Functions to create collision objects to contain robot motion planning and execution
within safe limits
*/

std::string spawnObject(moveit::planning_interface::PlanningSceneInterface& psi,
                 const moveit_msgs::msg::CollisionObject& object) {
	if (!psi.applyCollisionObject(object))
		throw std::runtime_error("Failed to spawn object: " + object.id);
	return object.id;
}

moveit_msgs::msg::CollisionObject createTable() {
	geometry_msgs::msg::Pose pose;
	moveit_msgs::msg::CollisionObject object;
	object.id = "base_collision_table";
	object.header.frame_id = "world";
	object.primitives.resize(1);
	object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
	object.primitives[0].dimensions = { 2.0, 1.0, 0.01 };
	pose.position.x = 0.75;
	pose.position.y -= 0.3;
	pose.position.z = 0.03;  // align surface with world
	object.primitive_poses.push_back(pose);
	return object;
}

moveit_msgs::msg::CollisionObject createCamWall() {
	geometry_msgs::msg::Pose pose;
	moveit_msgs::msg::CollisionObject object;
	object.id = "cam_1_2_collision_camera_wall";
	object.header.frame_id = "world";
	object.primitives.resize(1);
	object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
	object.primitives[0].dimensions = { 2.0, 0.01, 1.0 };
	pose.position.x = 1.25;
	pose.position.y = -0.0305;
	pose.position.z = 0.08;  // align surface with world
	object.primitive_poses.push_back(pose);
	return object;
}

moveit_msgs::msg::CollisionObject createWall() {
	geometry_msgs::msg::Pose pose;
	moveit_msgs::msg::CollisionObject object;
	object.id = "cam_0_3_collision_wall";
	object.header.frame_id = "world";
	object.primitives.resize(1);
	object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
	object.primitives[0].dimensions = { 2.0, 0.01, 1.0 };
	pose.position.x = 0.75;
	pose.position.y = -0.5805;
	pose.position.z = 0.08;  // align surface with world
	object.primitive_poses.push_back(pose);
	return object;
}

// moveit_msgs::msg::CollisionObject createUr5Back() {
// 	geometry_msgs::msg::Pose pose;
// 	moveit_msgs::msg::CollisionObject object;
// 	object.id = "ur5_back_wall";
// 	object.header.frame_id = "world";
// 	object.primitives.resize(1);
// 	object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
// 	object.primitives[0].dimensions = { 0.01, 1.0, 1.0 };
// 	pose.position.x = 0.40;
// 	pose.position.y = -0.35;
// 	pose.position.z = 0.40;  // align surface with world
// 	object.primitive_poses.push_back(pose);
// 	return object;
// }
// moveit_msgs::msg::CollisionObject createUr10eBack() {
// 	geometry_msgs::msg::Pose pose;
// 	moveit_msgs::msg::CollisionObject object;
// 	object.id = "ur10e_back_wall";
// 	object.header.frame_id = "world";
// 	object.primitives.resize(1);
// 	object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
// 	object.primitives[0].dimensions = { 0.01, 1.0, 1.0 };
// 	pose.position.x = -1.90;
// 	pose.position.y = -0.35;
// 	pose.position.z = 0.40;  // align surface with world
// 	object.primitive_poses.push_back(pose);
// 	return object;
// }

// moveit_msgs::msg::CollisionObject createCeiling() {
// 	geometry_msgs::msg::Pose pose;
// 	moveit_msgs::msg::CollisionObject object;
// 	object.id = "base_collision_ceiling";
// 	object.header.frame_id = "world";
// 	object.primitives.resize(1);
// 	object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
// 	object.primitives[0].dimensions = { 2.0, 1.0, 0.01 };
// 	pose.position.x -= 0.75;
// 	pose.position.y -= 0.3;
// 	pose.position.z = 1.0;  // align surface with world
// 	object.primitive_poses.push_back(pose);
// 	return object;
// }

// moveit_msgs::msg::CollisionObject createCenterBox() {
// 	geometry_msgs::msg::Pose pose;
// 	moveit_msgs::msg::CollisionObject object;
// 	object.id = "center_platform";
// 	object.header.frame_id = "world";
// 	object.primitives.resize(1);
// 	object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
// 	object.primitives[0].dimensions = { 0.4, 0.8, 0.18415 };
// 	pose.position.x = -0.65;
// 	pose.position.y = -0.1;
// 	pose.position.z = 0.07;  // align surface with world
// 	object.primitive_poses.push_back(pose);
// 	return object;
// }

void setupDemoScene() {
	// Add table and object to planning scene
	rclcpp::sleep_for(std::chrono::microseconds(100));  // Wait for ApplyPlanningScene service
	moveit::planning_interface::PlanningSceneInterface psi1;
    std::string table_id = spawnObject(psi1, createTable());
    std::string cam_wall_id = spawnObject(psi1, createCamWall());
    std::string wall_id = spawnObject(psi1, createWall());
    // std::string wall_5_id = spawnObject(psi1, createUr5Back());
    // std::string wall_10e_id = spawnObject(psi1, createUr10eBack());
    // std::string ceiling_id = spawnObject(psi1, createCeiling());
    // std::string platform_id = spawnObject(psi1, createCenterBox());
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
	
	// pick place task
	rclcpp::NodeOptions node_options;
	node_options.automatically_declare_parameters_from_overrides(true);
	auto node = rclcpp::Node::make_shared("base_collision_table", node_options);
	std::thread spinning_thread([node] { rclcpp::spin(node); });

	setupDemoScene();

	// Keep introspection alive
	spinning_thread.join();

	return 0;
}
