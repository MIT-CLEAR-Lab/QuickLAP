import torch
import torchcontrol as toco
from torchcontrol.utils import to_tensor

from typing import Dict, Optional

from polymetis import RobotInterface
import time

import zmq
import numpy as np




##############################################
#
#            Impedance Controller
#
##############################################

class JointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space.
    """
    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.nn.Parameter(torch.zeros_like(self.joint_pos_desired))

    def set_goal(
        self,
        joint_pos_desired: torch.Tensor,
        joint_vel_desired: Optional[torch.Tensor] = None,
    ) -> None:
        """Update desired joint position/velocity on the local policy object.

        Note: if the policy is running on the robot via Polymetis, use
        `robot.update_current_policy(...)` (see helper below) to update the
        remote policy instance.
        """
        with torch.no_grad():
            self.joint_pos_desired.copy_(joint_pos_desired.view_as(self.joint_pos_desired))
            if joint_vel_desired is None:
                self.joint_vel_desired.zero_()
            else:
                self.joint_vel_desired.copy_(joint_vel_desired.view_as(self.joint_vel_desired))

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # Parse current state
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


def update_impedance_goal(
    robot: RobotInterface,
    joint_pos_desired,
    joint_vel_desired=None,
) -> None:
    """Update the running JointImpedanceControl policy goal on the robot.

    This is the intended way to change goals dynamically after calling
    `robot.send_torch_policy(policy, blocking=False)`.
    """
    q = to_tensor(joint_pos_desired)
    payload = {"joint_pos_desired": q}
    if joint_vel_desired is not None:
        payload["joint_vel_desired"] = to_tensor(joint_vel_desired)

    try:
        robot.update_current_policy(payload)
        
    except:
        # Restart the controller with the same policy
        policy = get_franka_impedance_controller(robot)
        robot.send_torch_policy(policy, blocking=False)

        

def get_franka_impedance_controller(robot):
    # Get the robot model for torch control and start impedance controlle
    robot_model = toco.models.RobotModelPinocchio(
        urdf_filename="/home/minyoung/fairo/polymetis/polymetis/python/polymetis/data/franka_panda/panda_arm.urdf",
        ee_link_name="panda_link8"
    )
    policy = JointImpedanceControl(
        joint_pos_current=robot.get_joint_positions(),
        Kp= 3.0 * torch.Tensor(robot.metadata.default_Kq),
        Kd= 0.1* torch.Tensor(robot.metadata.default_Kqd),
        robot_model=robot_model
    )
    return policy


def receive_joint_state_command(robot):
    '''
    Blocks to receive the state from robosuite.
    '''
    msg = sock.recv()              # blocking
    state = np.frombuffer(msg, dtype=np.float64) # state is 14-dim array ([q_des, q_dot_des]) 
    
    try:
        q_cur = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        q_cmd = np.asarray(state[:7], dtype=np.float64)
        max_abs_err = float(np.max(np.abs(q_cmd - q_cur)))

        # Threshold (radians): above this, do a blocking move-to.
        move_threshold = 0.35
        if max_abs_err > move_threshold:
            q_tensor = torch.as_tensor(q_cmd, dtype=torch.float)
            robot.move_to_joint_positions(q_tensor)
        
        # Restart the controller with the same policy
        policy = get_franka_impedance_controller(robot)
        robot.send_torch_policy(policy, blocking=False)

    finally:
        # Reply after (potentially) moving so the sender can treat this as an ack.
        sock.send(state.tobytes()) 

    return state[:7], state[7:14], state[14]

if __name__ == '__main__':
    # Initialize robot interface and listen for commands
    
    robot = RobotInterface(
        ip_address = "172.16.0.1"
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://0.0.0.0:5555")

    # Get the robot model for torch control and start impedance controller
    policy = get_franka_impedance_controller(robot)
    robot.send_torch_policy(policy, blocking=False)

    #set up a timer to refresh things at ~20 Hz (to match robosuite)
    hz = 20.0
    dt = 1.0 / hz
    next_t = time.perf_counter()

    while True:
        
        # get the desired position from robosuite and play it
        desired_position,  desired_velocity, gripper = recieve_joint_state_command(robot)
        update_impedance_goal(robot, joint_pos_desired=desired_position, joint_vel_desired=desired_velocity)
        
        # --- rate control (20 Hz) ---
        next_t += dt
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # we're running behind; resync to avoid accumulating lag
            next_t = time.perf_counter()
