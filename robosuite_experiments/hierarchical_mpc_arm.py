"""
Hierarchical MPC arm that combines phase-based task sequencing with MPC motion planning.

Phases control WHAT to do (grasp, transport, release)
MPC controls HOW to move (optimizing based on learned preferences)
"""

import copy
import numpy as np
from base_rational_arm import BaseRationalArm
from arm_world import ArmWorld
from mpc_arm_planner import MPCArmPlanner
import arm_feature_utils as feature_utils


class HierarchicalMPCArm(BaseRationalArm):
    """
    Hybrid arm combining phase-based sequencing with MPC motion planning.
    
    Task phases provide high-level structure (what to do)
    MPC optimizes low-level motion (how to do it respecting learned weights)
    """
    
    def __init__(
        self,
        world: ArmWorld,
        learner=None,
        utterance="Go faster",
        expert_weights=None,
        intervention_interval=(50, 70),
        base_weights=None,
        seed=None,
        planner_horizon=8,
        planner_n_iter=15,
    ):
        """
        Initialize hierarchical MPC arm.
        
        Args:
            world: ArmWorld environment instance
            learner: PHRILearner instance for learning from interventions
            utterance: What the human says during intervention
            expert_weights: Weights representing expert's true preferences (6 features)
            intervention_interval: Tuple (start, end) for intervention timesteps
            base_weights: Initial weights (6 features)
            seed: Random seed
            planner_horizon: MPC planning horizon
            planner_n_iter: Number of MPC optimization iterations
        """
        # Initialize with base weights - now 6 features (removed red_dist)
        if base_weights is None:
            # [green_dist, velocity, collision, joints, block_to_zone, zone_c_prox]
            base_weights = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 2.0])
        
        super().__init__(world, weights=base_weights, seed=seed)
        
        self.learner = learner
        self.utterance = utterance
        
        # Expert weights: high velocity (10x), high block_to_zone (5x), closer to zone C unintentionally (2x)
        if expert_weights is None:
            expert_weights = np.array([1.0, 10.0, 2.0, 2.0, 5.0, 2.0])
        self.expert_weights = np.array(expert_weights, dtype=np.float32)
        
        # Single intervention period
        self.intervention_start, self.intervention_end = intervention_interval
        
        # Tracking variables
        self.timestep = 0
        self.recording = False
        self.robot_trajectory = []
        self.human_trajectory = []
        
        # Task phase state machine
        self.task_phase = "approach"  # approach -> descend -> grasp -> lift -> transport -> place_descend -> release -> retract
        self.grasp_start_time = None
        self.release_start_time = None
        self.initial_block_pos = None  # Store block position at grasp time
        
        # Create MPC planners
        self.robot_planner = MPCArmPlanner(
            world=world,
            arm=self,
            horizon=planner_horizon,
            n_iter=planner_n_iter,
            learning_rate=0.05,
            control_limits=(-0.15, 0.15),
        )
        
        self.expert_planner = MPCArmPlanner(
            world=world,
            arm=self,
            horizon=planner_horizon,
            n_iter=planner_n_iter,
            learning_rate=0.05,
            control_limits=(-0.20, 0.20),  # Expert can move faster
        )
    
    def features(self, obs):
        """
        Compute features from current observation.
        
        Uses centralized feature computation from arm_feature_utils.
        
        Args:
            obs: Observation dictionary from environment
            
        Returns:
            numpy array of feature values [6 features]
        """
        return feature_utils.compute_features(obs, include_red_dist=False, include_zones=True)
    
    def is_intervention(self):
        """Check if current timestep is within intervention period."""
        return self.intervention_start <= self.timestep <= self.intervention_end
    
    def _get_subgoal_and_gripper(self, obs):
        """
        Determine subgoal position and gripper state based on current phase.
        
        USES EXACT SAME LOGIC AS intervention_arm.py for reliability!
        
        Args:
            obs: Current observation
            
        Returns:
            Tuple of (subgoal_position, gripper_state)
        """
        ee_pos = obs["ee_pos"]
        red_block_pos = obs["red_block_pos"]
        zone_b_pos = obs["zone_b_pos"]
        
        # Check distances for phase transitions 
        horizontal_dist = np.linalg.norm(ee_pos[:2] - red_block_pos[:2])
        
        # Phase transitions and subgoal determination (EXACT SAME AS intervention_arm.py)
        if self.task_phase == "approach":
            # Phase 1: Horizontal approach (move XY above block)
            subgoal = red_block_pos + np.array([0, 0, 0.15])  # 15cm above block
            gripper = -1.0  # Open gripper
            
            # Transition: when horizontally close, move to descend
            if horizontal_dist < 0.08:  # Relaxed threshold 
                self.task_phase = "descend"
                
        elif self.task_phase == "descend":
            # Phase 2: Descend to block (move Z down to grasp height)
            # Position gripper at block center height for grasping
            subgoal = red_block_pos + np.array([0, 0, 0.0])  # At block center height
            gripper = -1.0  # Keep open
            
            # Check distance to TARGET (not to block center!)
            dist_to_target = np.linalg.norm(ee_pos - subgoal)
            
            # Transition: when at target position (EXACT SAME AS intervention_arm.py line 240)
            if dist_to_target < 0.02:  # Within 2cm of target - TIGHT THRESHOLD
                self.task_phase = "grasp"
                self.grasp_start_time = self.timestep
                # Store initial block position for lift transition
                self.initial_block_pos = red_block_pos.copy()
                
        elif self.task_phase == "grasp":
            # Phase 3: Close gripper and WAIT (don't move!)
            subgoal = ee_pos  # Stay in place!
            gripper = 1.0  # Close gripper
            
            # Wait 80 timesteps (4 seconds at 20Hz) for gripper to fully close
            if self.timestep - self.grasp_start_time > 80:
                self.task_phase = "lift"
                
        elif self.task_phase == "lift":
            # Phase 4: Lift the block up to high position
            # Use initial block position for consistent target
            if self.initial_block_pos is not None:
                subgoal = self.initial_block_pos + np.array([0, 0, 0.20])  # Lift 20cm up
            else:
                subgoal = red_block_pos + np.array([0, 0, 0.20])
            gripper = 1.0  # Keep closed
            
            # Transition when lifted (check against INITIAL block position)
            if self.initial_block_pos is not None and ee_pos[2] > self.initial_block_pos[2] + 0.15:  # 15cm above INITIAL position
                self.task_phase = "transport"
                print(f"[t={self.timestep}] *** ENTERING TRANSPORT PHASE - MPC WILL OPTIMIZE TO MOVE BLOCK TO ZONE B ***")
                
        elif self.task_phase == "transport":
            # **MPC PHASE**: Move block to zone B (MPC optimizes based on block_to_zone feature!)
            subgoal = zone_b_pos + np.array([0, 0, 0.15])  # Above zone B
            gripper = 1.0  # Keep gripper closed
            
            horizontal_block_to_zone = np.linalg.norm(red_block_pos[:2] - zone_b_pos[:2])
            if horizontal_block_to_zone < 0.10:  # Within 10cm - relaxed threshold to exit before gradients vanish
                self.task_phase = "place_descend"
                
        elif self.task_phase == "place_descend":
            # Phase 6: Descend to place the block ON the zone
            # Target slightly above zone to place block gently
            subgoal = zone_b_pos + np.array([0, 0, 0.03])  # 3cm above zone center 
            gripper = 1.0  # Keep closed
            
            # Check if we've descended enough (distance to target)
            dist_to_target = np.linalg.norm(ee_pos - subgoal)
            if dist_to_target < 0.05:  # Within 5cm of target position 
                self.task_phase = "release"
                self.release_start_time = self.timestep
                
        elif self.task_phase == "release":
            # Phase 7: Open gripper to release block
            subgoal = ee_pos  # Stay in place
            gripper = -1.0  # Open gripper
            
            # Wait a bit for gripper to open and block to settle
            if self.timestep - self.release_start_time > 40:  # Wait 2 seconds 
                self.task_phase = "retract"
                
        else:  # retract
            # Phase 8: Lift arm up and away
            subgoal = zone_b_pos + np.array([0, 0, 0.20])  # Lift 20cm above zone B
            gripper = -1.0  # Keep open
        
        return subgoal, gripper
    
    def get_robot_action(self, obs):
        """
        Get robot action using hierarchical MPC.
        
        Phase machine determines subgoal, MPC optimizes how to reach it.
        """
        subgoal, gripper_state = self._get_subgoal_and_gripper(obs)
        
        # Use MPC ONLY for transport phase
        # Use direct control for all other phases
        use_mpc = self.task_phase in ["transport"]
        
        if use_mpc:
            # Use MPC to optimize transport with current weights
            action = self._mpc_to_subgoal(obs, subgoal, gripper_state, self.robot_planner)
        else:
            # Direct phase-based control 
            ee_pos = obs["ee_pos"]
            direction = subgoal - ee_pos
            distance = np.linalg.norm(direction)
            
            # Speed depends on phase
            if self.task_phase in ["grasp", "release"]:
                # Stay in place
                speed = 0.0
            else:
                # Normal speed for movement
                speed = 0.1
            
            action = np.zeros(7)
            
            # Command motion with deadzone
            if distance > 0.01:  # 1cm threshold - command full speed
                direction = direction / distance  # Normalize
                action[:3] = direction * speed
            elif distance > 0.001:  # Between 1cm and 1mm - slow approach
                direction = direction / distance
                action[:3] = direction * (speed * 0.1)
            # else: distance < 1mm - stop (action stays zero)
            
            action[3:6] = 0.0  # No rotation
            action[6] = gripper_state
        
        # Debug output
        if self.timestep % 20 == 0:
            ee_pos = obs["ee_pos"]
            red_dist = np.linalg.norm(ee_pos - obs["red_block_pos"])
            zone_dist = np.linalg.norm(obs["red_block_pos"] - obs["zone_b_pos"])
            gripper_str = "CLOSED" if gripper_state > 0 else "open"
            
            if self.task_phase in ["grasp", "release"]:
                wait_time = self.timestep - (self.grasp_start_time if self.task_phase == "grasp" else self.release_start_time)
                print(f"[t={self.timestep}] Phase: {self.task_phase} (waiting... {wait_time}/80), "
                      f"gripper={gripper_str}")
            elif self.task_phase == "lift":
                # Special debug for lift phase
                if self.initial_block_pos is not None:
                    height_above_initial = ee_pos[2] - self.initial_block_pos[2]
                    print(f"[t={self.timestep}] Phase: {self.task_phase}, "
                          f"height_above_initial={height_above_initial:.3f}m (need 0.15m), "
                          f"block_to_zone={zone_dist:.3f}m, "
                          f"action_mag={np.linalg.norm(action[:3]):.3f}, "
                          f"gripper={gripper_str}")
                else:
                    print(f"[t={self.timestep}] Phase: {self.task_phase}, "
                          f"red_dist={red_dist:.3f}m, "
                          f"block_to_zone={zone_dist:.3f}m, "
                          f"action_mag={np.linalg.norm(action[:3]):.3f}, "
                          f"gripper={gripper_str}")
            else:
                # Add block height for transport phase debugging
                if self.task_phase == "transport":
                    red_block_height = obs["red_block_pos"][2]
                    print(f"[t={self.timestep}] Phase: {self.task_phase}, "
                          f"red_dist={red_dist:.3f}m, "
                          f"block_height={red_block_height:.3f}m, "
                          f"block_to_zone={zone_dist:.3f}m, "
                          f"action_mag={np.linalg.norm(action[:3]):.3f}, "
                          f"gripper={gripper_str}")
                else:
                    print(f"[t={self.timestep}] Phase: {self.task_phase}, "
                          f"red_dist={red_dist:.3f}m, "
                          f"block_to_zone={zone_dist:.3f}m, "
                          f"action_mag={np.linalg.norm(action[:3]):.3f}, "
                          f"gripper={gripper_str}")
        
        return action
    
    def get_expert_action(self, obs):
        """Get expert action using same hierarchical structure but expert weights."""
        subgoal, gripper_state = self._get_subgoal_and_gripper(obs)
        
        # MPC ONLY for transport, direct control for everything else
        use_mpc = self.task_phase in ["transport"]
        
        if use_mpc:
            # Temporarily swap to expert weights
            saved_weights = self.weights.copy()
            self.weights = self.expert_weights.copy()
            
            action = self._mpc_to_subgoal(obs, subgoal, gripper_state, self.expert_planner)
            
            self.weights = saved_weights
        else:
            # Direct control (FASTER speed for expert)
            ee_pos = obs["ee_pos"]
            direction = subgoal - ee_pos
            distance = np.linalg.norm(direction)
            
            # Expert moves faster
            if self.task_phase in ["grasp", "release"]:
                speed = 0.0
            else:
                speed = 0.15  # Faster than robot
            
            action = np.zeros(7)
            
            if distance > 0.01:
                direction = direction / distance
                action[:3] = direction * speed
            elif distance > 0.001:
                direction = direction / distance
                action[:3] = direction * (speed * 0.1)
            
            action[3:6] = 0.0
            action[6] = gripper_state
        
        return action
    
    def _mpc_to_subgoal(self, obs, subgoal, gripper_state, planner):
        """
        Use MPC to plan motion based on reward function.
        
        MPC optimizes based purely on the reward weights. The subgoal is used
        only to initialize the MPC optimization (warm start).
        
        When very close to goal, switch to direct control to avoid oscillations
        from vanishing gradients in the exponential reward features.
        
        Args:
            obs: Current observation
            subgoal: Target position for the end-effector (used for MPC initialization)
            gripper_state: Gripper command
            planner: MPC planner instance
            
        Returns:
            Action array
        """
        # Check if we're very close to the goal (for transport phase)
        if self.task_phase == "transport":
            red_block_pos = obs["red_block_pos"]
            zone_b_pos = obs["zone_b_pos"]
            block_to_zone_dist = np.linalg.norm(red_block_pos[:2] - zone_b_pos[:2])
            
            # If block is very close to zone, use direct control to avoid MPC oscillations
            if block_to_zone_dist < 0.08:  # Within 8cm - switch to direct control
                ee_pos = obs["ee_pos"]
                direction = subgoal - ee_pos
                distance = np.linalg.norm(direction)
                
                action = np.zeros(7)
                if distance > 0.01:
                    direction = direction / distance
                    action[:3] = direction * 0.08  # Slow, careful approach
                action[3:6] = 0.0
                action[6] = gripper_state
                return action
        
        # Normal MPC operation for longer distances
        # Don't pass subgoal to avoid initialization bias - let reward weights decide!
        action = planner.get_next_action(obs, gripper_state, target_pos=None)
        
        # Clip to safe limits
        action[:3] = np.clip(action[:3], -0.15, 0.15)
        action[3:6] = 0.0  # No rotation
        action[6] = gripper_state
        
        return action
    
    def get_action(self, obs):
        """
        Get action for current timestep, handling interventions.
        """
        self.timestep += 1
        
        if self.is_intervention():
            if not self.recording:
                self.recording = True
                self.robot_trajectory = []
                self.human_trajectory = []
                print(f"[Timestep {self.timestep}] Started intervention: '{self.utterance}'")
            
            # Get both actions
            robot_action = self.get_robot_action(obs)
            expert_action = self.get_expert_action(obs)
            
            # Record trajectories
            self.robot_trajectory.append({
                "obs": copy.deepcopy(obs),
                "control": robot_action.copy()
            })
            
            self.human_trajectory.append({
                "obs": copy.deepcopy(obs),
                "control": expert_action.copy()
            })
            
            return expert_action
        
        else:
            # Check if we just finished intervention
            if self.recording:
                print(f"[Timestep {self.timestep}] Ended intervention, updating weights...")
                
                # Convert to format expected by learner
                robot_traj = {
                    "state": [step["obs"] for step in self.robot_trajectory],
                    "control": [step["control"] for step in self.robot_trajectory]
                }
                human_traj = {
                    "state": [step["obs"] for step in self.human_trajectory],
                    "control": [step["control"] for step in self.human_trajectory]
                }
                
                # Update weights via learner
                if self.learner is not None:
                    self.learner.update_weights(robot_traj, human_traj)
                    print(f"Updated weights: {self.weights}")
                
                self.recording = False
                self.robot_trajectory = []
                self.human_trajectory = []
            
            return self.get_robot_action(obs)
    
    def get_feature_descriptions(self):
        """
        Get natural language descriptions of features for LLM-based learners.
        
        Uses centralized descriptions from arm_feature_utils.
        """
        return feature_utils.get_feature_descriptions(include_red_dist=False, include_zones=True)

