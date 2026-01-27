"""
Visualization script for robosuite experiments with Hierarchical MPC.

This script demonstrates the hybrid approach:
- Phases control WHAT to do (grasp, transport, release)
- MPC controls HOW to move (optimizing based on learned weights)
- Physical input (keyboard/spacemouse) for human intervention

During transport phase, MPC optimizes to maximize block_to_zone feature!

On macOS, run this with: mjpython visualize_experiment.py
On Linux, run with: python visualize_experiment.py

Physical Input Controls (when enabled):
  Position: W/S (Y), A/D (X), Q/E (Z)
  Orientation: I/K (pitch), J/L (yaw), U/O (roll)
  Gripper: SPACE to toggle
  Exit: ESC
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime
import dotenv
dotenv.load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_world import ArmWorld
from hierarchical_mpc_arm import HierarchicalMPCArm
from robosuite_phri_learner import RobosuitePHRILearner
from robosuite_learners import (
    RobosuiteMaskedLLMPHRILearner,
    RobosuiteAdaptGatedLLMPHRILearner,
)
from arm_feature_utils import (
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_EXPERT_WEIGHTS,
    FEATURE_NAMES,
)

# Robosuite's built-in keyboard device
from robosuite.devices import Keyboard


import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
IP_ADDRESS = '128.30.29.23'
sock.connect(f"tcp://{IP_ADDRESS}:5555")


def main():
    """Run experiment with hierarchical MPC and visualization."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run robosuite visualization with MPC")
    parser.add_argument(
        "--physical-input",
        action="store_true",
        help="Enable keyboard input for human intervention (robosuite built-in)"
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Sensitivity for keyboard input (default: 1.0)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2000,
        help="Episode length in timesteps (default: 2000)"
    )
    args = parser.parse_args()
    
    use_physical_input = args.physical_input
    
    print("="*70)
    print("HIERARCHICAL MPC ARM - WITH INTERVENTION LEARNING")
    print("="*70)
    print("\nKey Features:")
    print("  1. Phase-based task sequencing (approach → grasp → transport → release)")
    print("  2. MPC motion planning during transport phase")
    print("  3. 7 features including 'block_to_target_zone', 'zone_c_proximity', and 'height_maintain'")
    if use_physical_input:
        print("  4. PHYSICAL INPUT ENABLED - YOU are the expert!")
        print("     Controls: W/S (Y), A/D (X), Q/E (Z), SPACE (gripper), ESC (exit)")
    else:
        print("  4. Simulated expert intervention")
    print("  5. Weight learning from intervention using QuickLAP")
    print()
    print("Note: On macOS, this requires mjpython!")
    print()
    
    # Create world with visualization
    world = ArmWorld(
        has_renderer=True,  # Enable visualization
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=args.horizon,
        seed=42,
    )
    
    # Use centralized default weights from arm_feature_utils
    # 7 features: [green_dist, velocity, collision, joints, block_to_zone, zone_c_prox, height_maintain]
    base_weights = DEFAULT_BASE_WEIGHTS.copy()
    
    print("Creating Hierarchical MPC Arm with Learning...")
    print(f"  - Base weights: {base_weights}")
    print(f"    {FEATURE_NAMES}")
    if use_physical_input:
        print("  - Expert: HUMAN (physical input)")
    else:
        print("  - Expert: simulated")
    print()

    utterance = "Go faster"
    api_key = 'none' #os.getenv("OPENAI_API_KEY")

    # Create hierarchical MPC arm first without learner
    # When using physical input, human IS the expert (no simulated expert weights)
    if use_physical_input:
        expert_weights = None  # Human is the expert
    else:
        expert_weights = DEFAULT_EXPERT_WEIGHTS.copy()
    
    arm = HierarchicalMPCArm(
        world=world,
        learner=None,  # Will set after creation
        utterance=utterance,
        expert_weights=expert_weights,
        intervention_interval=(999999, 999999),  # Disable simulated intervention
        base_weights=base_weights,
        seed=42,
        planner_horizon=8,   # Short horizon for speed
        planner_n_iter=20,   # More iterations needed when starting from zero
    )
    
    # Now create learner with the arm
    learner = RobosuiteAdaptGatedLLMPHRILearner(
        arm, utterance, arm.get_feature_descriptions(), openai_api_key=api_key
    )
    arm.learner = learner
    
    print("Feature descriptions:")
    for i, (name, desc) in enumerate(arm.get_feature_descriptions().items()):
        print(f"  [{i}] {name}: {desc[:80]}...")
    print()
    
    # Initialize robosuite's built-in keyboard device if enabled
    keyboard_device = None
    if use_physical_input:
        keyboard_device = Keyboard(
            env=world.env,
            pos_sensitivity=args.input_scale,
            rot_sensitivity=args.input_scale * 0.5,  # Less sensitive rotation
        )
        # Wire up keyboard callback to the viewer
        world.env.viewer.add_keypress_callback(keyboard_device.on_press)
        print("Keyboard input initialized (robosuite built-in)!")
        print("  Controls: Arrow keys for XY, Q/E for Z")
        print("  Rotation: mouse drag or I/K, J/L, U/O")
        print("  Gripper: space bar")
        print()
    
    try:
        robot = world.env.robots[0]
        print("Running simulation with visualization...")
        if not use_physical_input:
            print("Watch for 'TRANSPORT PHASE' message - that's when MPC optimizes block movement!")
        print()
        
        obs = world.get_observation()
        total_reward = 0.0
        cumulative_rewards = []
        
        for t in range(args.horizon):
            # Get robot's planned action (before human input)
            robot_action = arm.get_action(obs)
            action = robot_action.copy() 
            
            # Track if human provided input this frame
            human_input_this_frame = False
            
            # Add keyboard input if enabled (using robosuite's built-in device)
            # Only allow physical input during TRANSPORT/MOVE phase (when human guidance matters)
            # Skip first 10 frames to let keyboard device initialize (avoid false positives)
            in_transport_phase = arm.task_phase in ["transport", "move"]
            if use_physical_input and keyboard_device and t >= 10 and in_transport_phase:
                # Get human input from keyboard device
                # Returns dict with 'right_delta' (6,) and 'right_gripper' keys
                device_action = keyboard_device.input2action()
                
                if device_action is not None:
                    right_delta = device_action.get("right_delta", np.zeros(6))
                    right_gripper = device_action.get("right_gripper", 0)
                    if isinstance(right_gripper, np.ndarray):
                        right_gripper = float(right_gripper.item()) if right_gripper.size == 1 else 0
                    
                    # Check if human is actively providing input (position/orientation only)
                    # Use higher thresholds to avoid false positives from device noise
                    # NOTE: We ignore gripper input during transport - robot must keep holding the block
                    delta_magnitude = np.linalg.norm(right_delta)
                    if delta_magnitude > 0.01:
                        human_input_this_frame = True
                        
                        # Apply human correction to position/orientation ONLY
                        # Do NOT override gripper - robot needs to keep it closed during transport
                        action[:6] += right_delta
                        
                        # Signal intervention for learning
                        arm.signal_physical_intervention(obs, robot_action, action)
            
            # Update intervention state (handles cooldown and triggers learning)
            if use_physical_input:
                arm.update_physical_intervention_state()
            
            # Step environment
            obs, _, done, _ = world.step(action)

            state = np.hstack([robot._joint_positions, robot._joint_velocities, [action[-1]]]) #TODO: Add gripper state somehow..            
            sock.send(state.tobytes())             # blocking send
            reply = sock.recv()                    # blocking recieve
            
            # Compute reward
            reward = arm.reward_fn(obs)
            total_reward += reward
            cumulative_rewards.append(total_reward)
            
            # Render
            world.render()
            
            # Print summary every 100 steps
            if t % 100 == 0 and t > 0:
                features = arm.features(obs)
                print(f"\n[Step {t}]")
                print(f"  Phase: {arm.task_phase}")
                print(f"  Cumulative reward: {total_reward:.2f}")
                print(f"  Current weights: {arm.weights}")
                print(f"  Block-to-zone feature: {features[4]:.4f}")
                print(f"  Zone-C proximity feature: {features[5]:.4f}")
                print(f"  Height maintain feature: {features[6]:.4f}")
            
            if done:
                print(f"\nEpisode ended at step {t}")
                break
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETED!")
        print(f"{'='*70}")
        print(f"Total steps: {t+1}")
        print(f"Final reward: {total_reward:.2f}")
        print(f"Final phase: {arm.task_phase}")
        print(f"Final weights: {arm.weights}")
        print()
        
        # Check if task was successful
        final_obs = world.get_observation()
        block_to_zone_dist = np.linalg.norm(
            final_obs["red_block_pos"] - final_obs["zone_b_pos"]
        )
        
        if block_to_zone_dist < 0.15:
            print("✓ SUCCESS! Block reached target zone!")
        else:
            print(f"✗ Block not at target (distance: {block_to_zone_dist:.3f}m)")
        
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nOn macOS, please run with: mjpython visualize_experiment.py")
        print("Or run without visualization by modifying has_renderer=False")
    
    finally:
        world.close()


if __name__ == "__main__":
    main()

