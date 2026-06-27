import rclpy
from rclpy.node import Node
from rclpy.waitable import Waitable
from rclpy.task import Future
import threading
from typing import List, Tuple
from rclpy.executors import NumberOfEntities

from typing import Union, Callable

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from scipy.spatial.transform import Rotation as R

def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = 1,
    time_to_wait=-1
):
    """
    Wait for the next incoming message.

    :param msg_type: message type
    :param node: node to initialize the subscription on
    :param topic: topic name to wait for message
    :param qos_profile: QoS profile to use for the subscription
    :param time_to_wait: seconds to wait before returning
    :returns: (True, msg) if a message was successfully received, (False, None) if message
        could not be obtained or shutdown was triggered asynchronously on the context.
    """
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

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return False, None

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is not None:
                    return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)

    return False, None

class FutureWaitable(Waitable):
    def __init__(self, future: Future, cb_group):
        super().__init__(callback_group=cb_group)
        self._future = future
        self._event = threading.Event()
        self._has_triggered = False
        self._lock = threading.Lock()

    def wait(self, timeout=None):
        """Block the calling thread until the future is done."""
        self._event.wait(timeout=timeout)
        return self._future.result() if self._future.done() else None

    def get_ready_callbacks(self, data=None) -> List[Tuple[object, Callable]]:
        """Return callback to trigger when the future is done."""
        with self._lock:
            if self._future.done() and not self._has_triggered:
                print("get_ready_callbacks: future done, returning callback")
                self._has_triggered = True
                return [(data, lambda: self._event.set())]
        return []

    def get_num_entities(self) -> NumberOfEntities:
        return NumberOfEntities(
            num_subscriptions=0,
            num_guard_conditions=0,
            num_timers=0,
            num_clients=0,
            num_services=0,
        )

    def add_to_wait_set(self, wait_set):
        # This waitable does not interact directly with the wait set
        pass

    def is_ready(self, wait_set) -> bool:
        ready = self._future.done() and not self._has_triggered
        print(f"is_ready called: future done? {self._future.done()} triggered? {self._has_triggered} -> {ready}")
        return ready

    def take_data(self):
        # Return the future itself (or any identifying object) as "data"
        return self._future

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState

from moveit_msgs.msg import (
    RobotState,
    MoveItErrorCodes
)

class MyNode(Node):
    fk_srv_name_ = "compute_fk"
    timeout_sec_ = 5.0
    base_ = "base_link"
    end_effector_ = "tool0"
    joint_state_topic_ = "joint_states"

    def __init__(self):
        super().__init__('my_node')
        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        self.coldest_pose_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_loop_cb_group = MutuallyExclusiveCallbackGroup()

        # self.wait_set = Waitable(self.control_loop_cb_group)
        self.coldest_pose_subscriber = self.create_subscription(
            # PoseStamped,
            PoseStamped,
            "debug_cold_pose",
            # "hottest_pose",
            self.cold_pose_callback,
            10,
            callback_group=self.coldest_pose_cb_group
        )

        self.timer = self.create_timer(
            2, 
            self.control_loop, 
            # callback_group=self.control_loop_cb_group
        )

        self.executor: MultiThreadedExecutor = None

    def cold_pose_callback(self, msg: PoseStamped):
        """Store latest coldest point from topic"""
        # if not self.context.ok():
        #     return
        # with self.query_lock:
        self.latest_coldest_point = msg
        self.get_logger().info(f"Received coldest position: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}")

    def get_joint_state(self) -> JointState:
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None
        
        return current_joint_state

    def call_fk(self, request):
        future = self.fk_client_.call_async(request)
        waitable = FutureWaitable(future, self.control_loop_cb_group)
        self.add_waitable(waitable)

        # Block this function until the result is ready
        result = waitable.wait(timeout=5.0)

        if result is None:
            self.get_logger().error("FK failed or timed out")
            return None

        self.get_logger().info("FK succeeded")
        return result

    def get_fk(self) -> PoseStamped | None:
        self.get_logger().info("start of fk")
        for _ in range(10):
            current_joint_state = self.get_joint_state()
            if current_joint_state is not None:
                break
        if current_joint_state is None:
            self.get_logger().error("Failed to get current joint state")
            return None
        self.get_logger().info("got joint state in fk")

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()

        request.header.frame_id = f"{self.base_}"
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.end_effector_)
        request.robot_state = current_robot_state

        # future = self.fk_client_.call_async(request)
        self.get_logger().info("calling for fk")

        response = self.call_fk(request)
        # if future.result() is None:
        #     self.get_logger().error("Failed to get FK solution")
        #     return None
        
        # response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
            return None
        self.get_logger().info("got fk")
        return response.pose_stamped[0]

    def control_loop(self):
        cold_pose = self.latest_coldest_point
        self.get_logger().info(f"Got cold pose {cold_pose.pose}")
        current_pose = self.get_fk()
        if current_pose is None:
            self.get_logger().error("Failed to get current pose")
        self.get_logger().info(f"got fk {current_pose}")

def main():
    rclpy.init()
    node = MyNode()

    try:
        # Spin the executor to process callbacks and waitables
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
