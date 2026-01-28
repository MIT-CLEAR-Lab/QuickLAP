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
  Keyboard:
    Position: W/S (Y), A/D (X), Q/E (Z)
    Orientation: I/K (pitch), J/L (yaw), U/O (roll)
    Gripper: SPACE to toggle
    Exit: ESC
  SpaceMouse (--spacemouse flag):
    6-DOF control with translation and rotation
    Left button: toggle gripper
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime
import dotenv
import mujoco
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

# SpaceMouse input (Linux only, requires evdev)
try:
    from franka_spacemouse import SpaceMouseInput
    SPACEMOUSE_AVAILABLE = True
except ImportError:
    SPACEMOUSE_AVAILABLE = False


# ZMQ for remote robot communication (optional)
ZMQ_AVAILABLE = False
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    pass



def render_counterfactual_marker(viewer, counterfactual_pos, actual_pos):
    """
    Render a marker showing the counterfactual robot position.
    
    Uses MuJoCo's viewer to draw a red sphere where the robot "would be"
    without the intervention, and a line connecting it to the actual position.
    
    Args:
        viewer: MuJoCo viewer instance
        counterfactual_pos: [x, y, z] position of counterfactual EE
        actual_pos: [x, y, z] actual EE position
    """
    if counterfactual_pos is None or viewer is None:
        return
    
    # Red sphere at counterfactual position
    mujoco.mjv_initGeom(
        viewer.viewer.user_scn.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.02, 0, 0]),  # size (radius)
        counterfactual_pos.astype(np.float64),  # position
        np.eye(3).flatten().astype(np.float64),  # rotation matrix
        np.array([1.0, 0.2, 0.2, 0.7]),  # RGBA (red, semi-transparent)
    )
    viewer.viewer.user_scn.ngeom = 1
    
    # Line connecting counterfactual to actual (shows divergence)
    midpoint = (counterfactual_pos + actual_pos) / 2
    direction = actual_pos - counterfactual_pos
    length = np.linalg.norm(direction)
    
    if length > 0.001:  # Only draw if there's meaningful divergence
        # Capsule oriented along the line
        # MuJoCo capsule: size[0] = radius, size[1] = half-length
        mujoco.mjv_initGeom(
            viewer.viewer.user_scn.geoms[1],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([0.005, length / 2, 0]),  # size
            midpoint.astype(np.float64),  # position (center of capsule)
            _rotation_matrix_from_direction(direction).flatten().astype(np.float64),
            np.array([1.0, 0.5, 0.0, 0.5]),  # RGBA (orange, semi-transparent)
        )
        viewer.viewer.user_scn.ngeom = 2


def _rotation_matrix_from_direction(direction):
    """Create rotation matrix to align Z-axis with given direction."""
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    # Find perpendicular vectors
    if abs(direction[2]) < 0.9:
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])
    
    x_axis = np.cross(up, direction)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(direction, x_axis)
    
    # Rotation matrix with direction as Z-axis
    rot = np.column_stack([x_axis, y_axis, direction])
    return rot

t = 0
mouse = SpaceMouseInput(sensitivity=.003)

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
    parser.add_argument(
        "--spacemouse",
        action="store_true",
        help="Enable SpaceMouse input for human intervention (Linux only, requires evdev)"
    )
    parser.add_argument(
        "--spacemouse-device",
        type=str,
        default="/dev/input/event3",
        help="SpaceMouse device path (default: /dev/input/event3). Run 'ls -l /dev/input/by-id/*SpaceMouse*' to find correct device."
    )
    parser.add_argument(
        "--zmq",
        action="store_true",
        help="Enable ZMQ communication with remote robot"
    )
    parser.add_argument(
        "--zmq-address",
        type=str,
        default="128.30.29.23",
        help="ZMQ server IP address (default: 128.30.29.23)"
    )
    parser.add_argument(
        "--zmq-port",
        type=int,
        default=5555,
        help="ZMQ server port (default: 5555)"
    )
    args = parser.parse_args()
    
    use_physical_input = args.physical_input or args.spacemouse
    use_keyboard = args.physical_input
    use_spacemouse = args.spacemouse
    
    # Check spacemouse availability
    if use_spacemouse and not SPACEMOUSE_AVAILABLE:
        print("WARNING: SpaceMouse requested but evdev not available (Linux only).")
        print("         Falling back to keyboard-only mode.")
        use_spacemouse = False
        if not use_keyboard:
            use_keyboard = True  # Enable keyboard as fallback
    
    # Initialize ZMQ socket for remote robot communication
    zmq_socket = None
    if args.zmq:
        if not ZMQ_AVAILABLE:
            print("WARNING: ZMQ requested but pyzmq not installed.")
            print("         Install with: pip install pyzmq")
        else:
            try:
                zmq_ctx = zmq.Context()
                zmq_socket = zmq_ctx.socket(zmq.REQ)
                zmq_address = f"tcp://{args.zmq_address}:{args.zmq_port}"
                zmq_socket.connect(zmq_address)
                print(f"ZMQ connected to {zmq_address}")
            except Exception as e:
                print(f"WARNING: Failed to connect ZMQ socket: {e}")
                zmq_socket = None
    
    print("="*70)
    print("HIERARCHICAL MPC ARM - WITH INTERVENTION LEARNING")
    print("="*70)
    print("\nKey Features:")
    print("  1. Phase-based task sequencing (approach → grasp → transport → release)")
    print("  2. MPC motion planning during transport phase")
    print("  3. 7 features including 'block_to_target_zone', 'zone_c_clearance', and 'height_maintain'")
    if use_physical_input:
        print("  4. PHYSICAL INPUT ENABLED - YOU are the expert!")
        if use_keyboard:
            print("     Keyboard: W/S (Y), A/D (X), Q/E (Z), SPACE (gripper), ESC (exit)")
        if use_spacemouse:
            print(f"     SpaceMouse: 6-DOF control (device: {args.spacemouse_device})")
    else:
        print("  4. Simulated expert intervention")
        print("     🔴 RED SPHERE shows counterfactual (where robot WOULD be without intervention)")
        print("     🟠 ORANGE LINE shows divergence between counterfactual and actual position")
    print("  5. Weight learning from intervention using QuickLAP")
    if zmq_socket is not None:
        print(f"  6. ZMQ remote robot sync enabled ({args.zmq_address}:{args.zmq_port})")
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
    # 7 features: [green_clearance, velocity, collision, joints, block_to_zone, zone_c_clearance, height_maintain]
    base_weights = DEFAULT_BASE_WEIGHTS.copy()
    
    # When using physical input, human IS the expert (no simulated expert weights)
    if use_physical_input:
        expert_weights = None  # Human is the expert
    else:
        expert_weights = DEFAULT_EXPERT_WEIGHTS.copy()

    utterance = "WOW MOVE!"
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("Creating Hierarchical MPC Arm with Learning...")
    print(f"  - Base weights: {base_weights}")
    print(f"    {FEATURE_NAMES}")
    if use_physical_input:
        print("  - Expert: HUMAN (physical input)")
        print("  - Expert weights: None (human provides corrections)")
    else:
        print("  - Expert: SIMULATED")
        print(f"  - Expert weights: {expert_weights}")
    print()
    
    arm = HierarchicalMPCArm(
        world=world,
        learner=None,  # Will set after creation
        utterance=utterance,
        expert_weights=expert_weights,
        intervention_interval=(780, 800),  # Disable simulated intervention
        # intervention_interval=(99999, 99999),  # Disable simulated intervention
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
    
    # Initialize input devices
    keyboard_device = None
    spacemouse_device = None
    
    if use_keyboard:
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
    
    if use_spacemouse:
        try:
            # Create SpaceMouse with custom device path if specified
            spacemouse_device = SpaceMouseInput()
            # Override device path if user specified one
            if args.spacemouse_device != "/dev/input/event3":
                from evdev import InputDevice
                spacemouse_device.device = InputDevice(args.spacemouse_device)
            print(f"SpaceMouse initialized (device: {args.spacemouse_device})!")
            print("  6-DOF control: translate and rotate")
            print("  Left button: toggle gripper")
            print()
        except Exception as e:
            print(f"WARNING: Failed to initialize SpaceMouse: {e}")
            print("         Continuing without SpaceMouse.")
            spacemouse_device = None
            use_spacemouse = False
    
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
            
            # Add physical input if enabled (keyboard and/or spacemouse)
            # Only allow physical input during TRANSPORT/MOVE phase (when human guidance matters)
            # Skip first 10 frames to let devices initialize (avoid false positives)
            in_transport_phase = arm.task_phase in ["transport", "move"]
            if use_physical_input and t >= 10 and in_transport_phase:
                human_delta = np.zeros(6)
                
                # Get keyboard input (robosuite's built-in device)
                if keyboard_device is not None:
                    device_action = keyboard_device.input2action()
                    if device_action is not None:
                        right_delta = device_action.get("right_delta", np.zeros(6))
                        right_gripper = device_action.get("right_gripper", 0)
                        if isinstance(right_gripper, np.ndarray):
                            right_gripper = float(right_gripper.item()) if right_gripper.size == 1 else 0
                        human_delta += right_delta
                
                # Get spacemouse input
                if spacemouse_device is not None:
                    spacemouse_cmd = spacemouse_device.get_input()
                    # spacemouse_cmd is [x, y, z, rx, ry, rz, gripper]
                    # Map to action delta: position (first 3) and orientation (next 3)
                    spacemouse_delta = np.array([
                        spacemouse_cmd[0],   # x translation
                        spacemouse_cmd[1],   # y translation
                        spacemouse_cmd[2],   # z translation
                        spacemouse_cmd[3],   # rx rotation
                        spacemouse_cmd[4],   # ry rotation
                        spacemouse_cmd[5],   # rz rotation
                    ])
                    human_delta += spacemouse_delta
                
                # Check if human is actively providing input (position/orientation only)
                # Use higher thresholds to avoid false positives from device noise
                # NOTE: We ignore gripper input during transport - robot must keep holding the block
                delta_magnitude = np.linalg.norm(human_delta)
                if delta_magnitude > 0.01:
                    human_input_this_frame = True
                    
                    # Apply human correction to position/orientation ONLY
                    # Do NOT override gripper - robot needs to keep it closed during transport
                    action[:6] += human_delta
                    
                    # Signal intervention for learning
                    arm.signal_physical_intervention(obs, robot_action, action)
            
            # Update intervention state (handles cooldown and triggers learning)
            if use_physical_input:
                arm.update_physical_intervention_state()
            
            # Step environment
            obs, _, done, _ = world.step(action)


            t+= 1
            if t > 25:
                state = np.hstack([robot._joint_positions, robot._joint_velocities, [action[-1]]]) #TODO: Add gripper state somehow..            
                sock.send(state.tobytes())             # blocking send
                reply = sock.recv()                    # blocking recieve
                t=0
            # Send state to remote robot via ZMQ (if enabled)
            if zmq_socket is not None:
                state = np.hstack([robot._joint_positions, robot._joint_velocities, [action[-1]]])  # TODO: Add gripper state somehow..
                zmq_socket.send(state.tobytes())  # blocking send
                reply = zmq_socket.recv()  # blocking receive
            
            # Compute reward
            reward = arm.reward_fn(obs)
            total_reward += reward
            cumulative_rewards.append(total_reward)
            
            # Render counterfactual marker during simulated intervention
            # (shows where robot "would be" without expert intervention)
            if not use_physical_input and arm.recording and arm.robot_sim_state is not None:
                counterfactual_pos = arm.robot_sim_state["ee_pos"]
                actual_pos = obs["ee_pos"]
                try:
                    render_counterfactual_marker(world.env.viewer, counterfactual_pos, actual_pos)
                except Exception as e:
                    pass  # Silently ignore rendering errors
            
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
        if zmq_socket is not None:
            zmq_socket.close()


if __name__ == "__main__":
    main()

# # Without ZMQ (default)
# python visualize_experiment.py

# # With ZMQ using defaults
# python visualize_experiment.py --zmq

# # With ZMQ custom address/port
# python visualize_experiment.py --zmq --zmq-address 192.168.1.100 --zmq-port 5556

# # Combined with spacemouse
# python visualize_experiment.py --spacemouse --zmq --zmq-address 128.30.29.23
