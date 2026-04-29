#!/usr/bin/python3
# NOTE: Must use system Python (same as ROS 2). Do not use ``#!/usr/bin/env python3`` —
# when launched from a conda env, env resolves to conda Python whose libstdc++ is too
# old for rclpy (GLIBCXX_3.4.30).
"""ROS 2 / MuJoCo bridge for the Go2 quadruped.

Loads a MuJoCo MJCF (default: mujoco_menagerie/unitree_go2/scene.xml shipped
in this package's `description/mujoco/`), publishes joint_states + IMU + base
pose at 200 Hz on the SDK joint order:

  FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf

and steps physics at 1 kHz, applying torques received on /go2/cmd_torque.

External-disturbance test:
  ros2 service call /go2/disturb std_srvs/srv/Trigger {} (or just edit the
  service-less convenience parameter ``disturb_force``) — but the simplest
  thing while developing is to run this node with ``visualize:=true`` and use
  Ctrl + right-click drag in the MuJoCo passive viewer to push the robot.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, Vector3

import mujoco

# Joint name ordering used everywhere on the ROS side.
SDK_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def _resolve_default_mjcf() -> str:
    """Find the bundled scene.xml at runtime (works both from source and after install)."""
    # If installed via colcon, the file lives under the package's share/ directory.
    try:
        from ament_index_python.packages import get_package_share_directory  # type: ignore
        share = get_package_share_directory("go2_rl_control")
        candidate = os.path.join(share, "description", "mujoco", "scene.xml")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    # Fallback to source-tree layout.
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "description", "mujoco", "scene.xml"))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError("Could not locate go2 scene.xml; set ~mjcf_path explicitly.")


def _coerce_bool(val: Any, default: bool = True) -> bool:
    """ROS 2 launch often passes ``visualize`` as the string ``\"true\"``/``\"false\"``.
    ``bool(\"false\")`` is True in Python — never use bare ``bool()`` on strings."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim_node")

        self.declare_parameter("mjcf_path", _resolve_default_mjcf())
        self.declare_parameter("sim_dt", 0.001)            # 1 kHz physics
        self.declare_parameter("publish_rate", 200.0)      # 200 Hz ROS publish
        self.declare_parameter("visualize", True)
        self.declare_parameter("init_keyframe", "home")    # name in MJCF; "" to skip

        mjcf_path = self.get_parameter("mjcf_path").value
        self.sim_dt: float = float(self.get_parameter("sim_dt").value)
        self.pub_rate: float = float(self.get_parameter("publish_rate").value)
        self.visualize = _coerce_bool(self.get_parameter("visualize").value, True)
        init_kf: str = str(self.get_parameter("init_keyframe").value)

        disp = os.environ.get("DISPLAY", "")
        if self.visualize and not disp.strip():
            self.get_logger().error(
                "参数 visualize=true，但当前环境未设置 DISPLAY（常见于纯 SSH、Docker、云主机）。"
                "MuJoCo 的 GLFW 窗口无法弹出。解决办法：在本机图形桌面终端运行；"
                "或 SSH 使用 ssh -X/-Y；或 export DISPLAY=:0（仅当机器已有桌面会话）；"
                "或使用 noVNC/VNC 登录桌面后再启动。"
            )
        elif self.visualize:
            self.get_logger().info(f"DISPLAY={disp!r} — 将尝试打开 MuJoCo passive viewer")

        self.get_logger().info(f"Loading MJCF: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        if init_kf:
            try:
                kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, init_kf)
                if kf_id >= 0:
                    mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
                    self.get_logger().info(f"Loaded keyframe '{init_kf}'.")
            except Exception as e:
                self.get_logger().warn(f"Keyframe '{init_kf}' not found: {e}")

        # ---- joint / actuator index maps (MJCF order -> SDK order) ----
        self.joint_qposadr = []  # qpos addr in MJCF order, for the 12 SDK joints
        self.joint_qveladr = []
        self.actuator_id = []    # actuator index in MJCF order, for the 12 SDK joints
        for name in SDK_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint '{name}' not in MJCF")
            self.joint_qposadr.append(self.model.jnt_qposadr[jid])
            self.joint_qveladr.append(self.model.jnt_dofadr[jid])
            # actuator name convention in mujoco_menagerie/unitree_go2 strips the
            # trailing "_joint" -> "FR_hip"; tolerate either.
            for cand in (name.replace("_joint", ""), name):
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, cand)
                if aid >= 0:
                    self.actuator_id.append(aid)
                    break
            else:
                raise RuntimeError(f"Actuator for joint '{name}' not in MJCF")

        # IMU body / sensor lookups.  Many Go2 MJCFs use body name "trunk" or
        # "base"; resolve at runtime.
        self.base_body_id = self._find_body(["trunk", "base", "base_link"])
        self.get_logger().info(f"base body id={self.base_body_id} "
                                f"name={mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_id)}")

        # ---- ROS pub/sub ----
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.joint_pub = self.create_publisher(JointState, "/go2/joint_states", qos)
        self.imu_pub = self.create_publisher(Imu, "/go2/imu", qos)
        self.pose_pub = self.create_publisher(PoseStamped, "/go2/base_pose", qos)
        self.torque_sub = self.create_subscription(
            Float64MultiArray, "/go2/cmd_torque", self._torque_cb, 1
        )
        # /go2/disturb (geometry_msgs/Vector3) — applies an instantaneous N*s
        # impulse to the base body's linear momentum so we can probe the
        # policy's recovery behaviour from the command line:
        #   ros2 topic pub --once /go2/disturb geometry_msgs/msg/Vector3 \
        #       "{x: 30.0, y: 0.0, z: 0.0}"
        self.disturb_sub = self.create_subscription(
            Vector3, "/go2/disturb", self._disturb_cb, 1
        )

        self._cmd_torque = np.zeros(12, dtype=np.float64)
        self._torque_received = False
        self._pending_impulse = np.zeros(3, dtype=np.float64)
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        self.get_logger().info(
            f"MuJoCo sim running. physics={1.0/self.sim_dt:.0f} Hz publish={self.pub_rate:.0f} Hz "
            f"visualize={self.visualize}"
        )

    # ---- helpers ----
    def _find_body(self, candidates) -> int:
        for c in candidates:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, c)
            if i >= 0:
                return i
        # Fall back to the first body that is the floating base.
        for i in range(self.model.nbody):
            if self.model.body_jntnum[i] == 1 and self.model.jnt_type[self.model.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE:
                return i
        raise RuntimeError("Could not locate base body in MJCF")

    def _torque_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 12:
            return
        with self._lock:
            self._cmd_torque[:] = msg.data
            self._torque_received = True

    def _disturb_cb(self, msg: Vector3) -> None:
        with self._lock:
            self._pending_impulse[:] = (msg.x, msg.y, msg.z)
        self.get_logger().info(
            f"Disturbance impulse queued: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}) N*s"
        )

    # ---- sim loop ----
    def _sim_loop(self) -> None:
        viewer = None
        if self.visualize:
            try:
                from mujoco import viewer as mj_viewer
                viewer = mj_viewer.launch_passive(self.model, self.data)
                self.get_logger().info("MuJoCo passive viewer 已启动（若仍看不见窗口，请检查 DISPLAY / 远程桌面）。")
            except Exception as e:
                self.get_logger().error(
                    f"无法启动 MuJoCo passive viewer: {e}\n"
                    "常见原因：无 DISPLAY、OpenGL/GLFW 不可用、在 rootless SSH 会话跑图形。"
                    "仿真线程会继续运行（仅无窗口）；需要画面时请在本机桌面或 X11/VNC 环境下重试。"
                )
                viewer = None

        pub_period = 1.0 / self.pub_rate
        wall_now = time.perf_counter()
        next_pub_wall = wall_now
        next_sim_time = wall_now

        try:
            while not self._stop.is_set():
                with self._lock:
                    tau = self._cmd_torque.copy()
                    started = self._torque_received
                    impulse = self._pending_impulse.copy()
                    self._pending_impulse[:] = 0.0
                if np.any(impulse != 0.0):
                    base_mass = float(self.model.body_mass[self.base_body_id])
                    self.data.qvel[0:3] += impulse / max(base_mass, 1e-3)
                if started:
                    for i, aid in enumerate(self.actuator_id):
                        self.data.ctrl[aid] = float(tau[i])
                    mujoco.mj_step(self.model, self.data)
                else:
                    mujoco.mj_forward(self.model, self.data)

                wall_now = time.perf_counter()
                if wall_now >= next_pub_wall:
                    next_pub_wall = wall_now + pub_period
                    self._publish_state()

                if viewer is not None:
                    viewer.sync()

                # real-time pacing
                next_sim_time += self.sim_dt
                sleep = next_sim_time - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_sim_time = time.perf_counter()
        finally:
            if viewer is not None:
                try:
                    viewer.close()
                except Exception:
                    pass

    def _publish_state(self) -> None:
        now = self.get_clock().now().to_msg()

        # joint state
        js = JointState()
        js.header.stamp = now
        js.name = SDK_JOINT_NAMES
        js.position = [float(self.data.qpos[a]) for a in self.joint_qposadr]
        js.velocity = [float(self.data.qvel[a]) for a in self.joint_qveladr]
        js.effort = [float(self.data.actuator_force[a]) for a in self.actuator_id]
        self.joint_pub.publish(js)

        # base pose & IMU (from the floating base body / its 6-dof root joint)
        pos = self.data.xpos[self.base_body_id]
        # body quaternion in MuJoCo is (w, x, y, z) in xquat
        quat = self.data.xquat[self.base_body_id]
        # angular velocity in body frame: cvel returns (rot, lin) in world; convert
        # to body-frame angular vel via R^T * world_ang_vel.
        cvel = self.data.cvel[self.base_body_id]  # [wx, wy, wz, vx, vy, vz] (rot first)
        rot_world = cvel[:3]
        # rotation matrix from quat
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        rot_body = R.T @ rot_world

        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "imu_link"
        imu.orientation.w = float(quat[0])
        imu.orientation.x = float(quat[1])
        imu.orientation.y = float(quat[2])
        imu.orientation.z = float(quat[3])
        imu.angular_velocity.x = float(rot_body[0])
        imu.angular_velocity.y = float(rot_body[1])
        imu.angular_velocity.z = float(rot_body[2])
        # linear acc not used by the policy; leave zero.
        self.imu_pub.publish(imu)

        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = "world"
        ps.pose.position.x = float(pos[0])
        ps.pose.position.y = float(pos[1])
        ps.pose.position.z = float(pos[2])
        ps.pose.orientation = imu.orientation
        self.pose_pub.publish(ps)

    def destroy_node(self):  # type: ignore[override]
        self._stop.set()
        if self._sim_thread.is_alive():
            self._sim_thread.join(timeout=2.0)
        super().destroy_node()


def main(args: Optional[list] = None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
