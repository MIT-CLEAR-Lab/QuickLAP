import numpy as np
import tensorflow as tf

from interact_drive.car.base_rational_car import BaseRationalCar
from interact_drive.learner.phri_learner import PHRILearner

NUM_INTERVENTIONS = 4


class InterventionCar(BaseRationalCar):
    """HybridCar with simulated expert guidance for cone avoidance."""

    def __init__(
        self,
        expert_weights: np.ndarray,
        intervention_intervals: list[tuple[int, int]],
        learner: PHRILearner,
        *args,
        **kwargs,
    ):
        """
        Initialize the car with a learner and weights for expert intervention.

        Args:
            learner: The PHRILearner instance for learning.
            expert_weights: Weights for the expert intervention.
            intervention_intervals: List of tuples specifying the start and end
                intervals for expert intervention. Both inclusive, with 1 being
                the first time step.
        """
        super().__init__(*args, **kwargs)
        self.recording = False
        self.human_trajectory = []
        self.recorded_trajectories = []
        self.robot_trajectory = []
        self.time_step = 0
        self.expert_planner = None
        self.learner = learner
        self.expert_weights = expert_weights
        self.intervention_intervals = intervention_intervals
        self.world_horizon = 220

    def is_intervention(self) -> bool:
        """Check if the current time step is within the intervention period."""
        for start, end in self.intervention_intervals:
            if start <= self.time_step <= end:
                return True
        return False

    def get_feature_descriptions(self) -> dict[str, str]:
        return {
            "speed_desirability": "How close the car's speed is to the desired speed limit. This feature gives a maximum value of 1.0 when the car's speed matches the speed limit exactly, and decreases parabolically as the speed deviates from the optimal value.",
            "lane_alignment": "How well the car stays centered in its lane and maintains proper heading. This feature approaches 1 when the car is perfectly centered in a lane, and decreases quadratically as the car deviates from the lane center.",
            "off_road": "Penalty for driving off the road. This feature gives a value of 1 when the car is within road boundaries, and decreases quadratically as the car moves beyond the left or right limits of the road.",
            "cone_distance": "How well the car maintains safe distance from traffic cones. This feature gives a value of 1 when the car is far from obstacles and approaches 0 when the car gets very close to obstacles. Increase this feature weight would make the car drive further away from cones.",
        }

    def _get_next_control(self) -> tf.Tensor:
        """Get next control action with expert intervention when appropriate."""
        self.time_step += 1

        if self.is_intervention():
            if not self.expert_planner:
                from interact_drive.planner import NaivePlanner

                expert_planner_args = self.planner_args.copy()
                self.expert_planner = NaivePlanner(
                    self.env, self, **expert_planner_args
                )
            assert self.planner

            if not self.recording:
                self.recording = True
                self.human_trajectory = []
                self.robot_trajectory = []
                print("Started recording simulated expert trajectory")

            # Get robot's normal plan
            robot_state = (
                self.state
                if not self.robot_trajectory
                else self.dynamics_fn(
                    tf.constant(self.robot_trajectory[-1]["state"], dtype=tf.float32),
                    tf.constant(self.robot_trajectory[-1]["control"], dtype=tf.float32),
                    self.planner.world.dt,
                )
            )
            robot_overall_state = self.env.state.copy()
            robot_overall_state[0] = robot_state
            robot_plan = self.planner.generate_plan(init_state=robot_overall_state)
            planner_control = tf.identity(robot_plan[0])

            # Generate expert plan with modified weights
            orig_weights = self.weights.copy()
            self.weights = self.expert_weights
            expert_plan = self.expert_planner.generate_plan()
            self.weights = orig_weights

            # Use expert plan as the "human" intervention
            control = tf.constant(expert_plan[0], dtype=tf.float32)

            # Record both trajectories
            self.human_trajectory.append(
                {
                    "state": self.state.numpy(),
                    "control": control.numpy(),
                }
            )

            self.robot_trajectory.append(
                {
                    "state": robot_state.numpy(),
                    "control": planner_control.numpy(),
                }
            )

            return control
        else:
            planner_control = super()._get_next_control()
            if self.recording:
                if len(self.human_trajectory) > 0:
                    current_traj = {
                        "state": [step["state"] for step in self.human_trajectory],
                        "control": [step["control"] for step in self.human_trajectory],
                    }
                    robot_traj = {
                        "state": [step["state"] for step in self.robot_trajectory],
                        "control": [step["control"] for step in self.robot_trajectory],
                    }
                    self.learner.update_weights(robot_traj, current_traj)
                    print(f"Updated weights after {len(self.human_trajectory)} steps")

                self.recording = False
                self.human_trajectory = []
                self.robot_trajectory = []
                print("Stopped recording")

            return planner_control
