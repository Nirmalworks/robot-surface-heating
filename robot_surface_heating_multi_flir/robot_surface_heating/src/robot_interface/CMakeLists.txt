cmake_minimum_required(VERSION 3.8)
project(robot_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(moveit_common REQUIRED)
moveit_package()
find_package(moveit_core REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

find_package(rclcpp REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(thermal_camera_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PoseToPlan.srv"
)

set(THIS_PACKAGE_INCLUDE_DEPENDS
	moveit_core
  moveit_visual_tools
	moveit_ros_planning
	moveit_ros_planning_interface

	rclcpp
	tf2_eigen
)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(moveit_test.py
    src/moveit_test.cpp)
target_include_directories(moveit_test.py
    PUBLIC include)
ament_target_dependencies(moveit_test.py
    ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS moveit_test.py
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(demo_robot_interface.py
    src/moveit_test.cpp)
target_include_directories(demo_robot_interface.py
    PUBLIC include)
ament_target_dependencies(demo_robot_interface.py
    ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS demo_robot_interface.py
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(moveit_test_v2
    src/moveit_test.cpp)
target_include_directories(moveit_test_v2
    PUBLIC include)
ament_target_dependencies(moveit_test_v2
    ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS moveit_test_v2
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(base_collision_table
    src/collision_table.cpp)
target_include_directories(base_collision_table
    PUBLIC include)
ament_target_dependencies(base_collision_table
    ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS base_collision_table
  DESTINATION lib/${PROJECT_NAME}
)

# C++ dual motion functionality python binding
add_library(dual_motion SHARED src/dual_motion.cpp)
target_include_directories(dual_motion
    PUBLIC include)
ament_target_dependencies(dual_motion
    ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS dual_motion
  DESTINATION lib/${PROJECT_NAME}
)

# main robot interface for dual UR system
add_executable(dual_robot_interface.py
    src/moveit_test.cpp)
target_include_directories(dual_robot_interface.py
    PUBLIC include)
ament_target_dependencies(dual_robot_interface.py
    ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS dual_robot_interface.py
  DESTINATION lib/${PROJECT_NAME}
)

# gripper test interface
add_executable(gripper_test_node.py
    src/moveit_test.cpp)
target_include_directories(gripper_test_node.py
    PUBLIC include)
ament_target_dependencies(gripper_test_node.py
    ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS gripper_test_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  src/moveit_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/demo_robot_interface.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/dual_robot_interface.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/gripper_test_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/moveit_test.cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/curobo_planner.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/execute.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  src/test_robot_interface.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  src/overheat_safety_robot_interface.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  src/modular_executor.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
src/heater_path_executor.py
DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
src/toy_executor_stub.py
DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
src/lightweight_executor.py
DESTINATION lib/${PROJECT_NAME}
)


ament_package()
