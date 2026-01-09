"""
Intervention arm that simulates human interventions during task execution.
"""

import copy
import numpy as np
from base_rational_arm import BaseRationalArm
from arm_world import ArmWorld


class InterventionArm(BaseRationalArm):
    """
    Robotic arm agent with simulated expert interventions.
    
    During intervention periods, the arm is controlled by an "expert" policy
    that demonstrates the desired behavior (moving quickly toward green block).
    After intervention, the learner updates weights based on the demonstration.
    """
    
    def __init__(
        self,
        world: ArmWorld,
        learner,
        utterance="Go faster",
        expert_weights=None,
        intervention_interval=(50, 70),
        base_weights=None,
        seed=None,
    ):
        """
        Initialize the intervention arm.
        
        Args:
            world: ArmWorld environment instance
            learner: PHRILearner instance for learning from interventions
            utterance: What the human says during intervention
            expert_weights: Weights representing expert's true preferences
            intervention_interval: Tuple (start, end) for intervention timesteps
            base_weights: Initial weights (wrong preferences)
            seed: Random seed
        """
        # Initialize with base (wrong) weights
        if base_weights is None:
            base_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
        
        super().__init__(world, weights=base_weights, seed=seed)
        
        self.learner = learner
        self.utterance = utterance
        
        # Expert weights: high velocity (10x), others normal
        if expert_weights is None:
            expert_weights = np.array([1.0, 1.0, 10.0, 2.0, 2.0])
        self.expert_weights = np.array(expert_weights, dtype=np.float32)
        
        # Single intervention period
        self.intervention_start, self.intervention_end = intervention_interval
        
        # Tracking variables
        self.timestep = 0
        self.recording = False
        self.robot_trajectory = []
        self.human_trajectory = []
        
        # State machine for pick-and-place task (no backwards transitions!)
        self.task_phase = "approach"  # approach -> descend -> grasp -> lift -> move -> place_descend -> release -> retract
        self.grasp_start_time = None  # Track when grasping started
        self.release_start_time = None  # Track when release started
    
    def is_intervention(self):
        """Check if current timestep is within intervention period."""
        # DISABLED FOR NOW - return False to see base behavior
        return False
        # return self.intervention_start <= self.timestep <= self.intervention_end
    
    def get_expert_action(self, obs):
        """
        Generate expert action during intervention.
        
        OSC_POSE expects DELTA commands (small incremental movements).
        Expert moves FASTER (higher speed) to demonstrate "go faster".
        
        Args:
            obs: Current observation
            
        Returns:
            Action array for OSC controller [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ee_pos = obs["ee_pos"]
        red_block_pos = obs["red_block_pos"]
        zone_b_pos = obs["zone_b_pos"]
        
        # Expert speed - FASTER than robot (to show "go faster")
        speed = 0.15  # 1.5x robot speed (slower, more controlled)
        
        # Check distances for phase transitions
        horizontal_dist = np.linalg.norm(ee_pos[:2] - red_block_pos[:2])
        vertical_dist = abs(ee_pos[2] - red_block_pos[2])
        
        # Use same state machine as robot (for consistency)
        if self.task_phase == "approach":
            # Phase 1: Horizontal approach (move XY above block)
            target_pos = red_block_pos + np.array([0, 0, 0.15])  # 15cm above block
            gripper_action = -1.0  # Open gripper (robosuite: -1 = open)
            
            if horizontal_dist < 0.08:
                self.task_phase = "descend"
                
        elif self.task_phase == "descend":
            # Phase 2: Descend to block (move Z down to grasp height)
            # Position gripper at block center height for grasping
            target_pos = red_block_pos + np.array([0, 0, 0.0])  # At block center height
            gripper_action = -1.0  # Keep open (robosuite: -1 = open)
            
            # Check distance to TARGET (not to block center!)
            dist_to_target = np.linalg.norm(ee_pos - target_pos)
            
            if dist_to_target < 0.06:  # Within 6cm of target - relaxed for physical constraints
                self.task_phase = "grasp"
                self.grasp_start_time = self.timestep
                
        elif self.task_phase == "grasp":
            # Phase 3: Close gripper and WAIT (don't move!)
            target_pos = ee_pos  # Stay in place!
            gripper_action = 1.0  # Close gripper (robosuite: 1 = close)
            
            # Wait 80 timesteps (4 seconds at 20Hz) for gripper to fully close
            if self.timestep - self.grasp_start_time > 80:
                self.task_phase = "lift"
                
        elif self.task_phase == "lift":
            # Phase 4: Lift the block up (EXPERT IS FASTER)
            target_pos = red_block_pos + np.array([0, 0, 0.20])  # Lift 20cm up
            gripper_action = 1.0  # Keep closed (robosuite: 1 = close)
            
            if ee_pos[2] > red_block_pos[2] + 0.15:
                self.task_phase = "move"
                
        elif self.task_phase == "move":
            # Phase 5: Move to zone B (EXPERT IS FASTER)
            target_pos = zone_b_pos + np.array([0, 0, 0.15])  # Above zone B
            gripper_action = 1.0  # Keep gripper closed (robosuite: 1 = close)
            
            # Check if we're above zone B (horizontally close)
            horizontal_dist_to_zone = np.linalg.norm(ee_pos[:2] - zone_b_pos[:2])
            if horizontal_dist_to_zone < 0.03:  # Within 3cm horizontally (achievable)
                self.task_phase = "place_descend"
                
        elif self.task_phase == "place_descend":
            # Phase 6: Descend to place the block ON the zone
            # Target slightly above zone to place block gently (block bottom will touch surface)
            target_pos = zone_b_pos + np.array([0, 0, 0.03])  # 3cm above zone center (accounts for block height)
            gripper_action = 1.0  # Keep closed (robosuite: 1 = close)
            
            # Check if we've descended enough (distance to target, not absolute height)
            dist_to_target = np.linalg.norm(ee_pos - target_pos)
            if dist_to_target < 0.05:  # Within 5cm of target position
                self.task_phase = "release"
                self.release_start_time = self.timestep
                
        elif self.task_phase == "release":
            # Phase 7: Open gripper to release block
            target_pos = ee_pos  # Stay in place
            gripper_action = -1.0  # Open gripper (robosuite: -1 = open)
            
            # Wait a bit for gripper to open and block to settle
            if self.timestep - self.release_start_time > 40:  # Wait 2 seconds
                self.task_phase = "retract"
                
        else:  # task_phase == "retract"
            # Phase 8: Lift arm up and away
            target_pos = zone_b_pos + np.array([0, 0, 0.20])  # Lift 20cm above zone B
            gripper_action = -1.0  # Keep open
        
        # Constant velocity control for OSC delta commands
        direction = target_pos - ee_pos
        distance = np.linalg.norm(direction)
        
        # For OSC_POSE controller: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = np.zeros(7)
        
        # Command motion with deadzone to avoid oscillation when at target
        if distance > 0.01:  # 1cm threshold - command full speed
            direction = direction / distance  # Normalize
            action[:3] = direction * speed  # Constant velocity (FASTER)
        elif distance > 0.001:  # Between 1cm and 1mm - slow approach
            direction = direction / distance  # Normalize
            action[:3] = direction * (speed * 0.1)  # 10% speed for fine positioning
        # else: distance < 1mm - stop (action stays zero)
        
        action[3:6] = 0.0  # No rotation - keep orientation stable!
        action[6] = gripper_action  # Gripper is last dimension
        
        return action
    
    def get_robot_action(self, obs):
        """
        Generate robot's normal action (what it would do without intervention).
        
        OSC_POSE expects DELTA commands (small incremental movements).
        Robot moves SLOWER (lower speed) - baseline behavior with low velocity preference.
        
        Args:
            obs: Current observation
            
        Returns:
            Action array for OSC controller [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ee_pos = obs["ee_pos"]
        red_block_pos = obs["red_block_pos"]
        zone_b_pos = obs["zone_b_pos"]
        
        # Robot speed - increased from 0.08 (default OSC action space is [-1, 1])
        speed = 0.1  # Normalized action magnitude
        
        # Check distances for phase transitions
        horizontal_dist = np.linalg.norm(ee_pos[:2] - red_block_pos[:2])
        vertical_dist = abs(ee_pos[2] - red_block_pos[2])
        
        # State machine 
        if self.task_phase == "approach":
            # Phase 1: Horizontal approach (move XY above block)
            target_pos = red_block_pos + np.array([0, 0, 0.15])  # 15cm above block
            gripper_action = -1.0  # Open gripper (robosuite: -1 = open)
            
            # Transition: when horizontally close, move to descend
            if horizontal_dist < 0.08:  # Relaxed threshold
                self.task_phase = "descend"
                
        elif self.task_phase == "descend":
            # Phase 2: Descend to block (move Z down to grasp height)
            # Position gripper at block center height for grasping
            target_pos = red_block_pos + np.array([0, 0, 0.0])  # At block center height
            gripper_action = -1.0  # Keep open (robosuite: -1 = open)
            
            # Check distance to TARGET (not to block center!)
            dist_to_target = np.linalg.norm(ee_pos - target_pos)
            
            # Transition: when at target position (relaxed threshold since robot can't get arbitrarily close)
            if dist_to_target < 0.02:  # Within 6cm of target - relaxed for physical constraints
                self.task_phase = "grasp"
                self.grasp_start_time = self.timestep
                
        elif self.task_phase == "grasp":
            # Phase 3: Close gripper and WAIT (don't move!)
            target_pos = ee_pos  # Stay in place!
            gripper_action = 1.0  # Close gripper (robosuite: 1 = close)
            
            # Wait 80 timesteps (4 seconds at 20Hz) for gripper to fully close
            if self.timestep - self.grasp_start_time > 80:
                self.task_phase = "lift"
                
        elif self.task_phase == "lift":
            # Phase 4: Lift the block up
            target_pos = red_block_pos + np.array([0, 0, 0.20])  # Lift 20cm up
            gripper_action = 1.0  # Keep closed (robosuite: 1 = close)
            
            # Transition when lifted
            if ee_pos[2] > red_block_pos[2] + 0.15:  # 15cm above
                self.task_phase = "move"
                
        elif self.task_phase == "move":
            # Phase 5: Move to zone B
            target_pos = zone_b_pos + np.array([0, 0, 0.15])  # Above zone B
            gripper_action = 1.0  # Keep gripper closed (robosuite: 1 = close)
            
            # Check if we're above zone B (horizontally close)
            horizontal_dist_to_zone = np.linalg.norm(ee_pos[:2] - zone_b_pos[:2])
            if horizontal_dist_to_zone < 0.03:  # Within 3cm horizontally (achievable)
                self.task_phase = "place_descend"
                
        elif self.task_phase == "place_descend":
            # Phase 6: Descend to place the block ON the zone
            # Target slightly above zone to place block gently (block bottom will touch surface)
            target_pos = zone_b_pos + np.array([0, 0, 0.03])  # 3cm above zone center (accounts for block height)
            gripper_action = 1.0  # Keep closed (robosuite: 1 = close)
            
            # Check if we've descended enough (distance to target, not absolute height)
            dist_to_target = np.linalg.norm(ee_pos - target_pos)
            if dist_to_target < 0.05:  # Within 5cm of target position
                self.task_phase = "release"
                self.release_start_time = self.timestep
                
        elif self.task_phase == "release":
            # Phase 7: Open gripper to release block
            target_pos = ee_pos  # Stay in place
            gripper_action = -1.0  # Open gripper (robosuite: -1 = open)
            
            # Wait a bit for gripper to open and block to settle
            if self.timestep - self.release_start_time > 40:  # Wait 2 seconds
                self.task_phase = "retract"
                
        else:  # task_phase == "retract"
            # Phase 8: Lift arm up and away
            target_pos = zone_b_pos + np.array([0, 0, 0.20])  # Lift 20cm above zone B
            gripper_action = -1.0  # Keep open
        
        # Constant velocity control for OSC delta commands
        direction = target_pos - ee_pos
        distance = np.linalg.norm(direction)
        
        # For OSC_POSE controller: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = np.zeros(7)
        
        # Command motion with deadzone to avoid oscillation when at target
        if distance > 0.01:  # 1cm threshold - command full speed
            direction = direction / distance  # Normalize
            action[:3] = direction * speed  # Constant velocity
        elif distance > 0.001:  # Between 1cm and 1mm - slow approach
            direction = direction / distance  # Normalize
            action[:3] = direction * (speed * 0.1)  # 10% speed for fine positioning
        # else: distance < 1mm - stop (action stays zero)
        
        action[3:6] = 0.0  # No rotation - keep orientation stable!
        action[6] = gripper_action  # Gripper is last dimension
        
        # Debug output
        if self.timestep % 20 == 0:  # Every 1 second at 20Hz
            motion_state = "stopped" if distance < 0.001 else ("slow" if distance < 0.01 else "moving")
            gripper_state = "CLOSED" if gripper_action > 0 else "open"  # robosuite: +1 = close, -1 = open
            
            # Special info for grasp phase
            if self.task_phase == "grasp":
                wait_time = self.timestep - self.grasp_start_time
                print(f"[t={self.timestep}] Phase: {self.task_phase} (closing... {wait_time}/80), "
                      f"gripper={gripper_state}")
            else:
                print(f"[t={self.timestep}] Phase: {self.task_phase} ({motion_state}), "
                      f"target_dist={distance:.4f}m, " 
                      f"action_mag={np.linalg.norm(action[:3]):.3f}, "
                      f"gripper={gripper_state}")
        
        return action
    
    def get_action(self, obs):
        """
        Get action for current timestep, handling interventions.
        
        During intervention: use expert action and record both trajectories.
        Outside intervention: use robot's normal action.
        
        Args:
            obs: Current observation
            
        Returns:
            Action to execute
        """
        self.timestep += 1
        
        if self.is_intervention():
            if not self.recording:
                # Start recording
                self.recording = True
                self.robot_trajectory = []
                self.human_trajectory = []
                print(f"[Timestep {self.timestep}] Started intervention: '{self.utterance}'")
            
            # Get both actions
            robot_action = self.get_robot_action(obs)
            expert_action = self.get_expert_action(obs)
            
            # Record trajectories
            # Store the full observation dict for feature computation
            # Need deep copy since obs contains nested structures
            self.robot_trajectory.append({
                "obs": copy.deepcopy(obs),  # Deep copy observation dict
                "control": robot_action.copy()
            })
            
            self.human_trajectory.append({
                "obs": copy.deepcopy(obs),  # Deep copy observation dict
                "control": expert_action.copy()
            })
            
            # Execute expert action
            return expert_action
        
        else:
            # Check if we just finished intervention
            if self.recording:
                print(f"[Timestep {self.timestep}] Ended intervention, updating weights...")
                
                # Convert to format expected by learner
                # Learner needs "state" key with observation dicts
                robot_traj = {
                    "state": [step["obs"] for step in self.robot_trajectory],
                    "control": [step["control"] for step in self.robot_trajectory]
                }
                human_traj = {
                    "state": [step["obs"] for step in self.human_trajectory],
                    "control": [step["control"] for step in self.human_trajectory]
                }
                
                # Update weights via learner
                self.learner.update_weights(robot_traj, human_traj)
                
                # Get updated weights from learner (stored in car.weights by learner)
                print(f"Updated weights: {self.weights}")
                
                self.recording = False
                self.robot_trajectory = []
                self.human_trajectory = []
            
            # Execute normal robot action
            return self.get_robot_action(obs)
    
    def _obs_to_state(self, obs):
        """
        Convert observation dict to state array for learner compatibility.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            State array
        """
        # Concatenate relevant state information
        state = np.concatenate([
            obs["ee_pos"],
            obs["ee_vel"],
            obs["joint_pos"],
            obs["red_block_pos"],
            obs["green_block_pos"]
        ])
        return state
    
    def get_feature_descriptions(self):
        """
        Get natural language descriptions of features for LLM-based learners.
        
        Inherits from BaseRationalArm which uses centralized descriptions.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return super().get_feature_descriptions()

