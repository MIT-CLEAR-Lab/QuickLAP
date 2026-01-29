#!/usr/bin/env python3
"""
Main interface for running user studies

This script provides a simple terminal interface for selecting environments
and running user studies with unified data management.
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime

import dotenv
import numpy as np
import pandas as pd

from arm_world import ArmWorld
from arm_feature_utils import DEFAULT_BASE_WEIGHTS, DEFAULT_EXPERT_WEIGHTS
from franka_spacemouse import SpaceMouseInput
from user_arm import UserArm
from robosuite_learners import RobosuiteAdaptGatedLLMPHRILearner

import zmq

# Load environment variables
dotenv.load_dotenv()

SERVER_IP = "128.30.29.25"


def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj


class UserStudyManager:
    """Manages user studies across multiple environments with unified data storage."""

    # Method display name mapping
    METHOD_NAMES = {
        "phri": "Physical Correction Only",
        "llm": "Physical Correction + Language",
        "language": "Language Correction Only",
    }
    METHODS = {
        "phri": 0,
        "llm": 1,
        "language": 2,
    }

    def __init__(self, participant_id: int, experiment_sequence: list[str]):
        """
        Initialize the user study manager.

        Args:
            participant_id: Unique identifier for the participant
            experiment_sequence: List of (method, environment) tuples to run sequentially
                               method: "phri" or "llm"
                               environment: "cone_avoid", "puddle_avoid", "cone_car_avoid", "cone_car_avoid_four"
        """
        self.participant_id = participant_id

        # Set up session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{participant_id}_{timestamp}"
        self.save_dir = os.path.join("user_study", "sessions", self.session_id)
        os.makedirs(self.save_dir, exist_ok=True)

        self.experiment_sequence = experiment_sequence
        self.current_experiment_index = 0

        # Check if LLM-based methods are needed for any experiments
        self.llm_needed = any(
            method in ["phri", "llm", "language", "test"]
            for method in experiment_sequence
        )
        self.openai_api_key = None
        if self.llm_needed:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                print(
                    "Warning: No OpenAI API key found. LLM, Language, and Test experiments will be skipped."
                )
                # Filter out LLM-based experiments
                self.experiment_sequence = [
                    m for m in experiment_sequence if m == "phri"
                ]

        # Session data
        self.session_data = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "experiment_sequence": experiment_sequence,
            "experiments_completed": [],
            "total_duration": 0,
        }

        # Save initial session info
        self.save_session_info()

        self.mouse = SpaceMouseInput()

        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://{SERVER_IP}:5555")

        print(f"\nUser Study Session Started")
        print(f"Participant ID: {self.participant_id}")
        print(f"Session ID: {self.session_id}")
        print(f"Total Experiments: {len(self.experiment_sequence)}")
        print(f"Data Directory: {self.save_dir}")
        print("-" * 60)

    def display_experiment_sequence(self):
        """Display the experiment sequence and current progress."""
        print("\n📋 Experiment Sequence:")
        print("=" * 80)

        for i, method in enumerate(self.experiment_sequence):
            method_name = self.METHOD_NAMES.get(method, method)

            if i < self.current_experiment_index:
                status = "✅ Completed"
            elif i == self.current_experiment_index:
                status = "▶️  Current"
            else:
                status = "⭕ Pending"

            print(f"{i+1:2d}. {status:12} {method_name:17}")

        progress = (self.current_experiment_index / len(self.experiment_sequence)) * 100
        print(
            f"\nProgress: {self.current_experiment_index}/{len(self.experiment_sequence)} ({progress:.1f}%)"
        )
        print("=" * 80)

    def run_experiment(self, method: str) -> bool:
        """
        Run a specific experiment with given method and environment.

        Args:
            method: Learning method ("phri" or "llm")
            env_name: Environment name

        Returns:
            bool: True if successful, False otherwise
        """

        # Create experiment-specific save directory
        experiment_name = f"{method}"
        exp_save_dir = os.path.join(self.save_dir, experiment_name)
        os.makedirs(exp_save_dir, exist_ok=True)

        try:
            start_time = time.time()

            # Show optimal behavior demonstration (skip for test method)
            input("Press Enter to start the experiment...")

            # Now setup actual experiment world
            world = ArmWorld(
                has_renderer=True,  # Enable visualization
                has_offscreen_renderer=False,
                use_camera_obs=False,
                control_freq=20,
                horizon=1200,
                seed=42,
            )
            arm = UserArm(
                world=world,
                learner=None,
                base_weights=DEFAULT_BASE_WEIGHTS.copy(),
                seed=42,
                planner_horizon=8,  # Short horizon for speed
                planner_n_iter=20,  # More iterations needed when starting from zero
            )
            arm.learner = RobosuiteAdaptGatedLLMPHRILearner(
                arm,
                "",
                arm.get_feature_descriptions(),
                openai_api_key=self.openai_api_key,
                use_speech_input=True,
                audio_file_path=None,
            )
            robot = world.env.robots[0]

            # Save experiment configuration
            exp_config_data = {
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "optimal_weights": DEFAULT_EXPERT_WEIGHTS.tolist(),
                "initial_weights": arm.weights.tolist(),
                "initial_observation": serialize(world.get_observation()),
                "learner_type": method,
            }

            with open(os.path.join(exp_save_dir, "experiment_config.json"), "w") as f:
                json.dump(exp_config_data, f, indent=2)

            # Run simulation
            step_data = []

            obs = world.get_observation()
            for t in range(world.horizon):
                step_start_time = time.time()
                robot_action = arm.get_action(obs)
                action = robot_action.copy()

                state = np.hstack(
                    [robot._joint_positions, robot._joint_velocities, [0]]
                )  # TODO: Add gripper state somehow...
                self.sock.send(state.tobytes())  # blocking send
                reply = self.sock.recv()

                # Add keyboard input if enabled (using robosuite's built-in device)
                # Only allow physical input during TRANSPORT/MOVE phase (when human guidance matters)
                # Skip first 10 frames to let keyboard device initialize (avoid false positives)
                in_transport_phase = arm.task_phase in ["transport", "move"]
                if t >= 10 and in_transport_phase:
                    # Get human input from keyboard device
                    # Returns dict with 'right_delta' (6,) and 'right_gripper' keys
                    device_action = self.mouse.get_input()

                    if device_action is not None:
                        # Check if human is actively providing input (position/orientation only)
                        # Use higher thresholds to avoid false positives from device noise
                        # NOTE: We ignore gripper input during transport - robot must keep holding the block
                        delta_magnitude = np.linalg.norm(device_action[:6])
                        if delta_magnitude > 0.01:
                            # Apply human correction to position/orientation ONLY
                            # Do NOT override gripper - robot needs to keep it closed during transport
                            action[:6] += device_action[:6]

                            # Signal intervention for learning
                            arm.signal_physical_intervention(obs, robot_action, action)

                # Update intervention state (handles cooldown and triggers learning)
                arm.update_physical_intervention_state()

                # Step environment
                obs, _, done, _ = world.step(action)

                # Compute reward
                reward = arm.reward_fn(obs)

                # Render counterfactual marker during simulated intervention
                # (shows where robot "would be" without expert intervention)
                # if (
                #     not use_physical_input
                #     and arm.recording
                #     and arm.robot_sim_state is not None
                # ):
                #     counterfactual_pos = arm.robot_sim_state["ee_pos"]
                #     actual_pos = obs["ee_pos"]
                #     try:
                #         render_counterfactual_marker(
                #             world.env.viewer, counterfactual_pos, actual_pos
                #         )
                #     except Exception as e:
                #         pass  # Silently ignore rendering errors

                # Render
                world.render()

                current_features = arm.features(obs)
                step_metrics = {
                    "timestep": t,
                    "obs": serialize(obs),
                    "weights": arm.weights.tolist(),
                    "features": current_features.tolist(),
                    "reward": reward,
                    "is_intervention": arm.is_intervention(),
                    "time": time.time(),
                }
                step_data.append(step_metrics)

                if t % 50 == 0:
                    progress = (t / world.horizon) * 100
                    print(f"Progress: {progress:.1f}% ({t}/{world.horizon} steps)")

                # Maintain frame rate
                step_duration = time.time() - step_start_time
                time.sleep(max(0, 1.0 / 30 - step_duration))

            end_time = time.time()
            duration = end_time - start_time

            # Calculate weight similarity to optimal
            weight_similarity = np.dot(arm.weights, DEFAULT_EXPERT_WEIGHTS) / (
                np.linalg.norm(arm.weights) * np.linalg.norm(DEFAULT_EXPERT_WEIGHTS)
            )

            # Save results
            results_data = {
                "method": method,
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "duration": duration,
                "completed": True,
                "final_weights": arm.weights.tolist(),
                "optimal_weights": DEFAULT_EXPERT_WEIGHTS.tolist(),
                "weight_similarity": float(weight_similarity),
                "step_data": step_data,
                "car_step_data": arm.step_data if hasattr(arm, "step_data") else [],
            }

            with open(os.path.join(exp_save_dir, "results.json"), "w") as f:
                # print(results_data)
                json.dump(results_data, f, indent=2, default=str)

            # Update session data
            exp_completion = {
                "method": method,
                "experiment_name": experiment_name,
                "completed_at": datetime.now().isoformat(),
                "duration": duration,
                "final_weights": arm.weights.tolist(),
            }
            self.session_data["experiments_completed"].append(exp_completion)
            self.session_data["total_duration"] += duration
            self.save_session_info()

            print(f"\nEnvironment completed successfully!")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Final weights: {arm.weights.tolist()}")
            print(f"Weight similarity to optimal: {weight_similarity:.3f}")
            print("X" * 60)
            return True

        except Exception as e:
            print(f"\n❌ Error during environment execution: {e}")
            traceback.print_exc()

            # Save error info
            error_data = {
                "method": method,
                "experiment_name": experiment_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }

            with open(os.path.join(exp_save_dir, "error.json"), "w") as f:
                json.dump(error_data, f, indent=2)

            return False
        finally:
            world.close()
        
    def run_demo(self) -> bool:
        """
        Run a specific experiment with given method and environment.

        Args:
            method: Learning method ("phri" or "llm")
            env_name: Environment name

        Returns:
            bool: True if successful, False otherwise
        """

        try:
            start_time = time.time()

            # Show optimal behavior demonstration (skip for test method)
            input("Press Enter to start the test experiment...")

            # Now setup actual experiment world
            world = ArmWorld(
                has_renderer=True,  # Enable visualization
                has_offscreen_renderer=False,
                use_camera_obs=False,
                control_freq=20,
                horizon=1200,
                seed=42,
            )
            arm = UserArm(
                world=world,
                learner=None,
                base_weights=DEFAULT_EXPERT_WEIGHTS.copy(),
                seed=42,
                planner_horizon=8,  # Short horizon for speed
                planner_n_iter=20,  # More iterations needed when starting from zero
            )
            robot = world.env.robots[0]

            obs = world.get_observation()
            for t in range(world.horizon):
                step_start_time = time.time()
                robot_action = arm.get_action(obs)
                action = robot_action.copy()

                state = np.hstack(
                    [robot._joint_positions, robot._joint_velocities, [0]]
                )  # TODO: Add gripper state somehow...
                self.sock.send(state.tobytes())  # blocking send
                reply = self.sock.recv()

                if t >= 10:
                    # Get human input from keyboard device
                    # Returns dict with 'right_delta' (6,) and 'right_gripper' keys
                    device_action = self.mouse.get_input()

                    if device_action is not None:
                        # Check if human is actively providing input (position/orientation only)
                        # Use higher thresholds to avoid false positives from device noise
                        # NOTE: We ignore gripper input during transport - robot must keep holding the block
                        delta_magnitude = np.linalg.norm(device_action[:6])
                        if delta_magnitude > 0.01:
                            # Apply human correction to position/orientation ONLY
                            # Do NOT override gripper - robot needs to keep it closed during transport
                            action[:6] += device_action[:6]

                            # Signal intervention for learning
                            arm.signal_physical_intervention(obs, robot_action, action)
                # Update intervention state (handles cooldown and triggers learning)
                arm.update_physical_intervention_state()

                # Step environment
                obs, _, done, _ = world.step(action)

                world.render()

                if t % 50 == 0:
                    progress = (t / world.horizon) * 100
                    print(f"Progress: {progress:.1f}% ({t}/{world.horizon} steps)")

                # Maintain frame rate
                step_duration = time.time() - step_start_time
                time.sleep(max(0, 1.0 / 30 - step_duration))

            end_time = time.time()
            duration = end_time - start_time

            print(f"\nDemo Environment completed successfully!")
            print(f"Duration: {duration:.1f} seconds")
            print("X" * 60)
            return True
        except Exception as e:
            print(f"\n❌ Error during environment execution: {e}")
            traceback.print_exc()
            return False
        finally:
            world.close()

    def save_session_info(self):
        """Save current session information."""
        with open(os.path.join(self.save_dir, "session_info.json"), "w") as f:
            json.dump(self.session_data, f, indent=2)

    def show_session_summary(self):
        """Display session summary."""
        print(f"\n📊 Session Summary")
        print("=" * 80)
        print(f"Participant: {self.participant_id}")
        print(f"Session ID: {self.session_id}")
        print(f"Total Duration: {self.session_data['total_duration']:.1f} seconds")
        print(
            f"Experiments Completed: {len(self.session_data['experiments_completed'])}/{len(self.experiment_sequence)}"
        )
        print("-" * 80)

        for exp_completion in self.session_data["experiments_completed"]:
            method_name = self.METHOD_NAMES.get(
                exp_completion["method"], f"❓ {exp_completion['method']}"
            )
            print(
                f"  ✅ {method_name:17} | {exp_completion['duration']:6.1f}s"
            )

        remaining_count = len(self.experiment_sequence) - len(
            self.session_data["experiments_completed"]
        )
        if remaining_count > 0:
            print(f"\nRemaining Experiments: {remaining_count}")

        print(f"\nData saved to: {self.save_dir}")

    def run(self):
        """Run all experiments in the predefined sequence."""
        try:
            # Display the full sequence at the start
            self.display_experiment_sequence()

            print(f"\nDemo phase")

            self.run_demo()

            print(f"\n🎯 Ready to start {len(self.experiment_sequence)} experiments")
            input("Press Enter to begin the first experiment...")

            # Run each experiment in sequence
            while self.current_experiment_index < len(self.experiment_sequence):
                method = self.experiment_sequence[self.current_experiment_index]

                # Display current experiment info
                method_display = self.METHOD_NAMES.get(method, method)
                print(f"\n{'='*80}")
                print(
                    f"EXPERIMENT {self.current_experiment_index + 1}/{len(self.experiment_sequence)}"
                )
                print(f"Method: {method_display}")
                print(f"{'='*80}")

                # Skip LLM-based experiments if no API key
                if (
                    method in ["phri", "llm", "language", "test"]
                    and not self.openai_api_key
                ):
                    print(f"⏭️  Skipping {method} experiment (no API key)")
                    self.current_experiment_index += 1
                    continue

                try:
                    success = self.run_experiment(method)

                    if success:

                        # Check if there are more experiments
                        if self.current_experiment_index + 1 < len(
                            self.experiment_sequence
                        ):
                            next_idx = self.current_experiment_index + 1
                            print(f"\n🎉 Experiment completed!")
                            print(
                                f"📊 Progress: {next_idx}/{len(self.experiment_sequence)}"
                            )

                            # Show what's next
                            next_method = self.experiment_sequence[next_idx]
                            next_method_name = self.METHOD_NAMES.get(
                                next_method, f"❓ {next_method}"
                            )
                            print(f"\n\n\n🔜 Next: {next_method_name}\n\n\n")

                            input("\nPress Enter to continue to the next experiment...")

                            self.current_experiment_index += 1
                        else:
                            print(f"\n🎊 All experiments completed!")
                            break
                    else:
                        # Handle failure
                        retry = (
                            input(f"\n❌ Experiment failed. Retry? (y/n/skip): ")
                            .strip()
                            .lower()
                        )
                        if retry == "y":
                            continue  # Retry same experiment
                        elif retry == "skip":
                            self.current_experiment_index += 1  # Skip to next
                        else:
                            break  # Exit study

                except KeyboardInterrupt:
                    print(f"\n\n⏸️ Experiment interrupted by user.")
                    choice = (
                        input(
                            "Continue to next experiment (c), retry this one (r), or quit (q)? "
                        )
                        .strip()
                        .lower()
                    )
                    if choice == "c":
                        self.current_experiment_index += 1
                    elif choice == "r":
                        continue  # Retry same experiment
                    else:
                        break  # Quit study

        finally:
            self.show_session_summary()
            print(f"\n🎯 Session data saved to: {self.save_dir}")


def get_sequence(participant_id: int):
    df = pd.read_csv("participants.csv")
    participant_row = df[df["PID"] == participant_id]

    if participant_row.empty:
        raise ValueError(f"No condition found for participant ID {participant_id}")

    algorithm_conditions = participant_row.iloc[0]["algs"].split(",")

    alg_translation = {-1: "test", 0: "phri", 1: "llm", 2: "language"}

    return [alg_translation[int(alg)] for alg in algorithm_conditions]


def main(experiment_sequence=None):
    """
    Main entry point.

    Args:
        experiment_sequence: Optional list of (method, environment) tuples.
                           If None, uses default sequence.
    """
    print("Multimodal Intervention User Study")
    print("=" * 80)

    # Initialize and run study manager
    try:
        PID = int(input("Enter participant ID: "))
        experiment_sequence = get_sequence(PID)
        print(f"Assigned experiment sequence: {experiment_sequence}")

        study_manager = UserStudyManager(
            participant_id=PID, experiment_sequence=experiment_sequence
        )
        study_manager.run()
    except KeyboardInterrupt:
        print("\n\n👋 Study session terminated by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
