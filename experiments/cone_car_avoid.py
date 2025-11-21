"""
Cone avoidance experiment with simulated expert trajectory.
"""

import numpy as np
from typing import Iterable, Union

import tensorflow as tf
import random
from collections import defaultdict
from interact_drive import feature_utils
from experiments.experiment import InterventionExperiment
from experiments.intervention_car import InterventionCar, NUM_INTERVENTIONS
from interact_drive.learner.phri_learner import PHRILearner
from interact_drive.world import ThreeLaneConeCarWorld
from interact_drive.car import FixedVelocityCar


class ConeCarInterventionCar(InterventionCar):
    """HybridCar with simulated expert guidance for cone avoidance."""

    def __init__(self, learner_factory, *args, **kwargs):
        learner = learner_factory(self)
        super().__init__(
            np.array([10.0, 2.5, 40.0, 30.0, 0.0, 3.0]),
            [(45, 55), (85, 95), (125, 135), (165, 175)][:NUM_INTERVENTIONS],
            learner,
            *args,
            **kwargs,
        )
        self.world_horizon = 440

    def get_feature_descriptions(self):
        """Return descriptions of the features for LLM."""
        return {
            "speed_desirability": "How close the car's speed is to the desired speed limit. This feature gives a maximum value of 1.0 when the car's speed matches the speed limit exactly, and decreases parabolically as the speed deviates from the optimal value.",
            "lane_alignment": "How well the car stays centered in its lane and maintains proper heading. This feature approaches 1 when the car is perfectly centered in a lane, and decreases quadratically as the car deviates from the lane center.",
            "off_road": "Penalty for driving off the road. This feature gives a value of 1 when the car is within road boundaries, and decreases quadratically as the car moves beyond the left or right limits of the road.",
            "cone_distance": "How well the car maintains safe distance from traffic cones. This feature gives a value of 1 when the car is far from obstacles and approaches 0 when the car gets very close to obstacles.",
            "car_distance": "How well the car maintains safe distance from other vehicles. This feature gives a value of 1 when the car is far from other vehicles and approaches 0 when the car gets very close to other vehicles ahead.",
            "puddle_distance": "How well the car avoids puddles. This feature gives a value of 1 when the car is far from puddles and approaches 0 when the car gets very close to puddles.",
        }

    def base_weights(self):
        # [Speed, Lane, Off-road, Cone, Car, Puddle]
        return [10.0, 2.5, 40.0, 1.0, 40.0, 3.0]
    

    @tf.function
    def features(
        self,
        state: Iterable[Union[tf.Tensor, tf.Variable]],
    ) -> tf.Tensor:
        """
        Features:
            - speed_desirability: Quadratic that peaks at the car's speed limit
            - lane_alignment: Combination of distance to lane median and northward heading
            - off_road: Penalty for driving off road
            - cone_distance: Exponential penalty for getting close to cones
            - car_distance: Exponential penalty for getting close to other cars
            - puddle_distance: Exponential penalty for getting close to puddles

        Args:
            state: the state of the world
            control: the controls of this car in that state
            consider_other_cars: If False will ignore other cars for feature comps

        Returns:
            tf.Tensor: the five features this car cares about
        """
        feats = []

        car_state = state[self.index]

        # Speed feature
        feats.append(feature_utils.speed_desirability(car_state, self.SPEED_LIMIT))

        # Lane alignment feature
        lane_widths = np.array([lane.w for lane in self.env.lanes], dtype=np.float32)
        lane_feat = feature_utils.lane_alignment(
            car_state, self.lane_medians_t, self.lane_normals_t, lane_widths
        )
        feats.append(lane_feat)

        # Off-road feature
        feats.append(
            feature_utils.off_road(
                car_state,
                self.env.car_box,
                self.env.left_limit,
                self.env.right_limit,
                lane_width=lane_widths[0],
            )
        )

        # Cone distance feature - using obstacle locations
        locs_by_type = defaultdict(list)
        wh_by_type = defaultdict(list)
        for i in range(len(self.env.obstacles)):
            locs_by_type[self.env.obstacles[i][0]].append(self.env.obs_locs[i])
            wh_by_type[self.env.obstacles[i][0]].append(self.env.obs_wh[i])

        # Add cone distance feature
        feats.append(
            feature_utils.distance_to_obstacle(
                car_state, locs_by_type["cone"], wh_by_type["cone"]
            )
        )

        # Add car distance feature
        car_feat = feature_utils.distance_to_car(
            state,  # Full world state
            car_state,  # Current car's state
            self.index,  # Current car's index
            self.env.car_wh,
        )
        feats.append(car_feat)

        # Puddle distance feature
        feats.append(
            feature_utils.distance_to_obstacle(
                car_state, locs_by_type["puddle"], wh_by_type["puddle"]
            )
        )

        return tf.stack(feats, axis=-1)


class ConeCarAvoidExperiment(InterventionExperiment):
    def __init__(
        self,
        exp_name: str = "cone_avoid",
        save_dir: str | None = None,
        verbose: bool = False,
    ):
        """Initialize the cone avoidance experiment."""
        super().__init__(exp_name, save_dir, verbose)

    def setup_world(self, learner_factory, seed: int) -> ThreeLaneConeCarWorld:
        """Set up the car world with cones."""
        random.seed(seed)
        cone_positions = [
            [0.14, 2.0],
            [0.14, 4],
            [0.14, 8],
            [0.14, 10],
            [0.14, 12],
            [0.14, 14],
            [0.14, 16],
            [0.14, 13],
            [0, 6],
            [0, 9],
            [0, 11],
            [0, 13],
            [0, 15],
            [0, 17],
            [-0.14, 2.0],
            [-0.14, 9.0], 
            [-0.14, 11.0],
            [-0.14, 14.0],
        ]
        puddle_positions = [
            [-0.05, 2.0],
            [-0.1, 6],
            [-0.05, 6],
            [-0.1, 8],
            [0.1, 10],
            [-0.1, 12],
        ]
        for i in range(4, len(cone_positions)):
            cone_positions[i][0] += 0.01 * random.random() - 0.005
            cone_positions[i][1] += 0.1 * random.random() - 0.05
        for i in range(2, len(puddle_positions)):
            puddle_positions[i][0] += 0.01 * random.random() - 0.005
            puddle_positions[i][1] += 0.1 * random.random() - 0.05

        world = ThreeLaneConeCarWorld(
            cone_positions=cone_positions,
            puddle_positions=puddle_positions,
            visualizer_args={
                "name": self.exp_name,
                "display_y": True,
                "follow_main_car": True,
            },
        )

        car = ConeCarInterventionCar(
            learner_factory,
            world,
            np.array([0.14, -0.5, 0.8, np.pi / 2]),
            color="orange",
            planner_type="Naive",
            planner_args={
                "horizon": 5,
                "n_iter": 10,
                "h_index": 1,
            },
        )
        other_car_1 = FixedVelocityCar(
            world,
            np.array([0.14, 0.8, 0.45, np.pi]),
            velocity=0.45,
            color="gray",
            opacity=1.0,
        )
        other_car_2 = FixedVelocityCar(
            world,
            np.array([0, -0.2, 0.5, np.pi]),
            velocity=0.5,
            color="gray",
            opacity=1.0,
        )
        other_car_3 = FixedVelocityCar(
            world,
            np.array([-0.14, -0.5, 0.52, np.pi]),
            velocity=0.52,
            color="gray",
            opacity=1.0,
        )
        other_car_4 = FixedVelocityCar(
            world,
            np.array([-0.14, 0.5, 0.42, np.pi]),
            velocity=0.42,
            color="gray",
            opacity=1.0,
        )
        world.add_cars([car, other_car_1, other_car_2, other_car_3, other_car_4])
        return world


def main():
    experiment = ConeCarAvoidExperiment(verbose=True)
    experiment.run(lambda car: PHRILearner(car))

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
