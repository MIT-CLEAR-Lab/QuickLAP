"""MPC planner for robotic arm using gradient-based optimization."""

from typing import Union
import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from arm_world import ArmWorld
from base_rational_arm import BaseRationalArm
from arm_planner import ArmPlanner


@tf.function
def arm_dynamics_step(ee_pos, ee_vel, control, dt, damping=0.5):
    """
    Simplified dynamics model for end-effector motion.
    
    Assumes damped second-order dynamics where control commands
    cause acceleration toward target velocity.
    
    Args:
        ee_pos: Current end-effector position [x, y, z]
        ee_vel: Current end-effector velocity [vx, vy, vz]
        control: Control command [dx, dy, dz] (desired velocity direction)
        dt: Timestep duration
        damping: Damping coefficient
        
    Returns:
        Tuple of (new_pos, new_vel)
    """
    # Control represents desired velocity
    # Apply damping and acceleration toward desired velocity
    acc = (control - ee_vel) * damping
    
    # Update velocity with acceleration
    new_vel = ee_vel + acc * dt
    
    # Update position with velocity
    new_pos = ee_pos + new_vel * dt
    
    return new_pos, new_vel


class MPCArmPlanner(ArmPlanner):
    """
    MPC-based planner for robotic arm manipulation.
    
    Uses gradient ascent to optimize a sequence of control actions
    that maximize the expected reward over a planning horizon.
    """
    
    def __init__(
        self,
        world: ArmWorld,
        arm: BaseRationalArm,
        horizon: int = 10,
        n_iter: int = 20,
        learning_rate: float = 0.05,
        control_limits: tuple = (-0.2, 0.2),  # Velocity limits for safety
    ):
        """
        Initialize the MPC planner.
        
        Args:
            world: ArmWorld environment instance
            arm: BaseRationalArm with reward function and weights
            horizon: Planning horizon in timesteps
            n_iter: Number of gradient ascent iterations
            learning_rate: Learning rate for optimization
            control_limits: (min, max) bounds for control commands
        """
        super().__init__(world, arm)
        self.setup_planner(horizon, n_iter, learning_rate)
        
        self.control_low, self.control_high = control_limits
        self.dt = 0.05  # 20Hz control rate (match robosuite)
        self.damping = 0.8  # Damping for dynamics model
        
        # Target position for MPC initialization (set by get_next_action)
        self.current_target = None
        
        self.planning_processes = [self.initialize_parameters, self.optimize_plan]
    
    @tf.function
    def compute_grads(self, init_obs_dict):
        """
        Compute gradients of cumulative reward w.r.t. control sequence.
        
        Args:
            init_obs_dict: Dictionary of initial observations (as tensors)
            
        Returns:
            Gradient tensor
        """
        func_key = "grads"
        
        with tf.GradientTape() as tape:
            # Watch the control variable
            tape.watch(self.robot_control)
            
            # Extract initial state
            ee_pos = init_obs_dict["ee_pos"]
            ee_vel = init_obs_dict["ee_vel"]
            red_block_pos = init_obs_dict["red_block_pos"]
            green_block_pos = init_obs_dict["green_block_pos"]
            joint_pos = init_obs_dict["joint_pos"]
            joint_limits = init_obs_dict["joint_limits"]
            zone_b_pos = init_obs_dict.get("zone_b_pos", None)  # Optional for 6/7-feature version
            zone_c_pos = init_obs_dict.get("zone_c_pos", None)  # Optional for 7-feature version
            
            # Calculate initial offset between EE and red block (for grasped block dynamics)
            initial_ee_pos = init_obs_dict["ee_pos"]
            grasp_offset = red_block_pos - initial_ee_pos
            
            # Accumulate rewards over horizon
            rewards = []
            
            for t in range(self.horizon):
                # Get control for this timestep
                control = self.robot_control[t * self.NC : (t + 1) * self.NC]
                
                # Predict next state using simplified dynamics
                ee_pos, ee_vel = arm_dynamics_step(
                    ee_pos, ee_vel, control, self.dt, self.damping
                )
                
                # If block is grasped, it moves with the end effector
                # Update block position to follow EE (maintaining grasp offset)
                red_block_pos = ee_pos + grasp_offset
                
                # Compute features and reward at this state
                reward = self._compute_reward_tf(
                    ee_pos, ee_vel, red_block_pos, green_block_pos,
                    joint_pos, joint_limits, zone_b_pos, zone_c_pos
                )
                rewards.append(reward)
            
            # Total reward to maximize
            total_reward = tf.reduce_sum(rewards)
        
        # Compute gradient
        gradient = tape.gradient(total_reward, self.robot_control)
        
        # Handle None gradient (can happen if no connection)
        if gradient is None:
            gradient = tf.zeros_like(self.robot_control)
        
        drr_rc = tf.convert_to_tensor(gradient)
        
        self.report_run(func_key)
        return drr_rc
    
    def _compute_reward_tf(self, ee_pos, ee_vel, red_block_pos, green_block_pos,
                          joint_pos, joint_limits, zone_b_pos=None, zone_c_pos=None):
        """
        Compute reward using TensorFlow operations.
        
        Reimplements the reward function in TensorFlow for gradient computation.
        
        Args:
            ee_pos: End-effector position
            ee_vel: End-effector velocity
            red_block_pos: Red block position
            green_block_pos: Green block position
            joint_pos: Joint positions
            joint_limits: Joint limits tuple
            zone_b_pos: Target zone B position (optional, for 7-feature version)
            zone_c_pos: Obstacle zone C position (optional, for 7-feature version)
            
        Returns:
            Scalar reward value
        """
        # Import feature utils for TensorFlow computation
        import arm_feature_utils as feature_utils
        
        # Compute features (these functions handle TensorFlow tensors)
        feat_green_dist = feature_utils.distance_to_green_block(ee_pos, green_block_pos)
        feat_velocity = feature_utils.end_effector_velocity(ee_vel)
        feat_collision = feature_utils.collision_safety(ee_pos, green_block_pos)
        feat_joints = feature_utils.joint_safety(joint_pos, joint_limits)
        
        # Check if we have 7, 6, 5, or 4 features
        num_weights = len(self.arm.weights)
        
        if num_weights == 7 and zone_b_pos is not None and zone_c_pos is not None:
            # 7 features: includes block_to_zone, zone_c, and height_maintain
            feat_block_to_zone = feature_utils.distance_block_to_target_zone(red_block_pos, zone_b_pos)
            feat_zone_c_proximity = feature_utils.proximity_to_obstacle_zone(ee_pos, zone_c_pos)
            feat_height = feature_utils.maintain_transport_height(ee_pos)
            features = tf.stack([
                feat_green_dist,
                feat_velocity,
                feat_collision,
                feat_joints,
                feat_block_to_zone,
                feat_zone_c_proximity,
                feat_height
            ])
        elif num_weights == 6 and zone_b_pos is not None and zone_c_pos is not None:
            # 6 features: includes block_to_zone and zone_c (no height_maintain)
            feat_block_to_zone = feature_utils.distance_block_to_target_zone(red_block_pos, zone_b_pos)
            feat_zone_c_proximity = feature_utils.proximity_to_obstacle_zone(ee_pos, zone_c_pos)
            features = tf.stack([
                feat_green_dist,
                feat_velocity,
                feat_collision,
                feat_joints,
                feat_block_to_zone,
                feat_zone_c_proximity
            ])
        elif num_weights == 5 and zone_b_pos is not None:
            # 5 features: includes block_to_zone (original + block_to_zone)
            feat_red_dist = feature_utils.distance_to_red_block(ee_pos, red_block_pos)
            feat_block_to_zone = feature_utils.distance_block_to_target_zone(red_block_pos, zone_b_pos)
            features = tf.stack([
                feat_red_dist,
                feat_green_dist,
                feat_velocity,
                feat_collision,
                feat_joints,
                feat_block_to_zone
            ])
        else:
            # Original 4 features (no red_dist, no zones)
            features = tf.stack([
                feat_green_dist,
                feat_velocity,
                feat_collision,
                feat_joints
            ])
        
        # Convert weights to tensor
        weights = tf.constant(self.arm.weights, dtype=tf.float32)
        
        # Compute reward as dot product
        reward = tf.reduce_sum(features * weights)
        
        return reward
    
    def optimize_plan(
        self,
        init_obs: Union[None, dict] = None,
        void_graph_setup: bool = False,
        target_pos: Union[None, np.ndarray] = None,
    ):
        """
        Generate optimized control sequence using gradient ascent.
        
        Args:
            init_obs: Initial observation dictionary
            void_graph_setup: If True, only run one iteration to setup graph
            target_pos: Optional target position to initialize toward (e.g., zone B during transport)
        """
        if init_obs is None:
            init_obs = self.init_obs
        
        # Initialize controls
        ee_pos = init_obs["ee_pos"]
        
        if target_pos is not None:
            # Initialize with velocity toward explicit target (if provided)
            direction = target_pos - ee_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0.01:
                direction = direction / distance
                init_control = np.tile(direction * 0.05, self.horizon)  # 5cm/s toward target
                self.robot_control.assign(init_control.astype(np.float32))
            else:
                # Too close, just use zeros
                self.robot_control.assign(self.zeros_control)
        else:
            # No target provided - initialize with small RANDOM perturbation
            # This avoids NaN gradients from starting at exactly zero (constant reward problem)
            # Random gives MPC a starting point to explore without directional bias
            # TODO: Better initialization strategy?
            random_init = np.random.uniform(-0.02, 0.02, size=self.horizon * self.NC)
            self.robot_control.assign(random_init.astype(np.float32))
        
        # Convert numpy arrays to TensorFlow tensors for gradient computation
        init_obs_tf = {
            "ee_pos": tf.constant(init_obs["ee_pos"], dtype=tf.float32),
            "ee_vel": tf.constant(init_obs["ee_vel"], dtype=tf.float32),
            "red_block_pos": tf.constant(init_obs["red_block_pos"], dtype=tf.float32),
            "green_block_pos": tf.constant(init_obs["green_block_pos"], dtype=tf.float32),
            "joint_pos": tf.constant(init_obs["joint_pos"], dtype=tf.float32),
            "joint_limits": (
                tf.constant(init_obs["joint_limits"][0], dtype=tf.float32),
                tf.constant(init_obs["joint_limits"][1], dtype=tf.float32)
            ),
        }
        
        # Add zone_b_pos if present
        if "zone_b_pos" in init_obs:
            init_obs_tf["zone_b_pos"] = tf.constant(init_obs["zone_b_pos"], dtype=tf.float32)
        
        # Add zone_c_pos if present (for 7-feature version)
        if "zone_c_pos" in init_obs:
            init_obs_tf["zone_c_pos"] = tf.constant(init_obs["zone_c_pos"], dtype=tf.float32)
        
        # Run gradient ascent iterations
        for i in range(self.n_iter):
            # Compute gradients
            r_grad = self.compute_grads(init_obs_tf)
            
            # Check for NaN gradients
            if tf.reduce_any(tf.math.is_nan(r_grad)):
                print(f"WARNING: NaN gradient at iteration {i}, stopping optimization")
                break
            
            # Apply gradient (negative because we're maximizing, not minimizing)
            self.optimizer.apply_gradients([(-r_grad, self.robot_control)])
            
            # Project back into control limits
            self.robot_control.assign(
                tf.clip_by_value(
                    self.robot_control,
                    self.control_low,
                    self.control_high,
                )
            )
            
            # Early exit if just setting up graph
            if not self.graph_set_up and void_graph_setup:
                break
    
    def get_next_action(self, obs, gripper_state=1.0, target_pos=None):
        """
        Get the next action from the optimized plan.
        
        This is the main interface for the intervention arm.
        
        Args:
            obs: Current observation dictionary
            gripper_state: Gripper command (+1 closed, -1 open)
            target_pos: Optional target position for MPC initialization (e.g., zone B during transport)
            
        Returns:
            Action array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Store target for optimization
        self.current_target = target_pos
        
        # Generate plan from current state
        plan = self.generate_plan(obs)
        
        # Extract first action (MPC receding horizon)
        control = plan[0].numpy()  # Convert from TensorFlow to numpy
        
        # Check for NaN and replace with zeros if needed
        if np.any(np.isnan(control)):
            print(f"WARNING: NaN in MPC control, using zero action")
            control = np.zeros(self.NC)
        
        # Clip to safe limits
        control = np.clip(control, self.control_low, self.control_high)
        
        # Construct full action (add rotation and gripper)
        action = np.zeros(7)
        action[:3] = control  # Position commands
        action[3:6] = 0.0  # No rotation
        action[6] = gripper_state  # Gripper command
        
        return action

