import asyncio
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from rclpy.qos import QoSProfile
from typing import Optional

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec


def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    *,
    qos_profile: QoSProfile = QoSProfile(depth=10),
    time_to_wait=-1
):
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(msg_type, topic, lambda _: None, qos_profile=qos_profile)
    try:
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready and sigint_gc.handle.pointer in guards_ready:
            return False, None

        if subs_ready and sub.handle.pointer in subs_ready:
            msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
            if msg_info is not None:
                return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)

    return False, None

async def await_rclpy_future(rclpy_future):
    loop = asyncio.get_event_loop()
    wrapped_future = loop.create_future()

    def done_callback(fut):
        try:
            res = fut.result()
            loop.call_soon_threadsafe(wrapped_future.set_result, res)
        except Exception as e:
            loop.call_soon_threadsafe(wrapped_future.set_exception, e)

    rclpy_future.add_done_callback(done_callback)
    return await wrapped_future

class MyNode(Node):
    fk_srv_name_ = "compute_fk"
    timeout_sec_ = 5.0
    base_ = "base_link"
    end_effector_ = "tool0"
    joint_state_topic_ = "joint_states"

    def __init__(self):
        super().__init__('my_node')

        self.callback_group = ReentrantCallbackGroup()

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_, callback_group=self.callback_group)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        self.latest_coldest_point = None

        self.coldest_pose_subscriber = self.create_subscription(
            PoseStamped,
            "debug_cold_pose",
            self.cold_pose_callback,
            10,
            callback_group=self.callback_group
        )

        self.create_timer(2.0, self.schedule_control_loop, callback_group=self.callback_group)

    def cold_pose_callback(self, msg: PoseStamped):
        self.latest_coldest_point = msg
        self.get_logger().info(f"Received coldest position: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}")

    def get_joint_state(self) -> Optional[JointState]:
        success, msg = wait_for_message(JointState, self, self.joint_state_topic_, time_to_wait=1.0)
        if not success:
            self.get_logger().error("Failed to get current joint state")
            return None
        return msg

    async def get_fk_async(self) -> Optional[PoseStamped]:
        self.get_logger().info("Getting FK")

        for _ in range(10):
            joint_state = self.get_joint_state()
            if joint_state is not None:
                break
        else:
            self.get_logger().error("Joint state unavailable after retries.")
            return None

        robot_state = RobotState()
        robot_state.joint_state = joint_state

        request = GetPositionFK.Request()
        request.header.frame_id = self.base_
        request.header.stamp = self.get_clock().now().to_msg()
        request.fk_link_names.append(self.end_effector_)
        request.robot_state = robot_state

        future = self.fk_client_.call_async(request)
        response = await await_rclpy_future(future)

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"FK service call failed: error code {response.error_code.val}")
            return None

        self.get_logger().info("FK successful")
        return response.pose_stamped[0] if response.pose_stamped else None

    async def control_loop_async(self):
        cold_pose = self.latest_coldest_point
        if cold_pose is None:
            self.get_logger().warn("No cold pose received yet.")
            return

        self.get_logger().info(f"Cold pose: {cold_pose.pose}")

        current_pose = await self.get_fk_async()
        if current_pose is None:
            self.get_logger().error("Could not get current pose")
            return

        self.get_logger().info(f"Current FK pose: {current_pose.pose}")

    def schedule_control_loop(self):
        import asyncio
        asyncio.ensure_future(self.control_loop_async())


def main():
    rclpy.init()
    node = MyNode()

    loop = asyncio.get_event_loop()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        # Spin the executor in a thread, run asyncio event loop in main thread
        loop.create_task(loop.run_in_executor(None, executor.spin))
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        loop.stop()
        loop.close()


if __name__ == '__main__':
    main()
