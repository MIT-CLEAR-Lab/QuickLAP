"""
Visualization script for robosuite experiments with Hierarchical MPC.

This script demonstrates the hybrid approach:
- Phases control WHAT to do (grasp, transport, release)
- MPC controls HOW to move (optimizing based on learned weights)

During transport phase, MPC optimizes to maximize block_to_zone feature!

On macOS, run this with: mjpython visualize_experiment.py
On Linux, run with: python visualize_experiment.py
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_world import ArmWorld
from hierarchical_mpc_arm import HierarchicalMPCArm
from robosuite_phri_learner import RobosuitePHRILearner


def main():
    """Run experiment with hierarchical MPC and visualization."""
    print("="*70)
    print("HIERARCHICAL MPC ARM - WITH INTERVENTION LEARNING")
    print("="*70)
    print("\nKey Features:")
    print("  1. Phase-based task sequencing (approach → grasp → transport → release)")
    print("  2. MPC motion planning during transport phase")
    print("  3. 6 features including 'block_to_target_zone' and 'zone_c_proximity'")
    print("  4. Human intervention at t=800-900 to demonstrate expert behavior")
    print("  5. Weight learning from intervention using PHRI")
    print()
    print("Note: On macOS, this requires mjpython!")
    print()
    
    # Create world with visualization
    world = ArmWorld(
        has_renderer=True,  # Enable visualization
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=2000,
        seed=42,
    )
    
    
    # 6 features: [green_dist, velocity, collision, joints, block_to_zone, zone_c_prox]
    # Note: zone_c_prox has NEGATIVE weight to avoid obstacle zone C
    base_weights = np.array([1.0, 1.0, 2.0, 2.0, 10.0, -2.0])
    
    print("Creating Hierarchical MPC Arm with Learning...")
    print(f"  - Base weights: {base_weights}")
    print("    [green_dist, velocity, collision, joints, block_to_zone, zone_c_prox]")
    print("  - Intervention: t=800-900 during transport phase")
    print("  - Utterance: 'Go faster and avoid zone C'")
    print()

    # Create hierarchical MPC arm first without learner
    arm = HierarchicalMPCArm(
        world=world,
        learner=None,  # Will set after creation
        utterance="Go faster",
        expert_weights=np.array([3.0, 5.0, 2.0, 2.0, 10.0, 4.0]),  # Expert preferences
        intervention_interval=(99999, 99999),  # Intervention during transport 
        base_weights=base_weights,
        seed=42,
        planner_horizon=8,   # Short horizon for speed
        planner_n_iter=20,   # More iterations needed when starting from zero
    )
    
    # Now create learner with the arm
    learner = RobosuitePHRILearner(arm, log_file="learning_log_robosuite.txt")
    arm.learner = learner
    
    print("Feature descriptions:")
    for i, (name, desc) in enumerate(arm.get_feature_descriptions().items()):
        print(f"  [{i}] {name}: {desc[:80]}...")
    print()
    
    try:
        print("Running simulation with visualization...")
        print("Watch for 'TRANSPORT PHASE' message - that's when MPC optimizes block movement!")
        print()
        
        obs = world.get_observation()
        total_reward = 0.0
        cumulative_rewards = []
        
        for t in range(2000):
            # Get action from hierarchical MPC arm
            action = arm.get_action(obs)
            
            # Step environment
            obs, _, done, _ = world.step(action)
            
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

