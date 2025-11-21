import random
from collections import defaultdict
from typing import Callable, Iterable, Union

import numpy as np
import tensorflow as tf

from experiments.experiment import InterventionExperiment
from experiments.intervention_car import InterventionCar, NUM_INTERVENTIONS
from interact_drive import feature_utils
from interact_drive.learner.phri_learner import PHRILearner
from interact_drive.world import TwoLanePuddleCarWorld


class PuddleInterventionCar(InterventionCar):
    """
    The experiment is to determine whether or not the car can learn to avoid cones while
    not picking up a negative weight for puddles.
    """

    def __init__(
        self, learner_factory: Callable[[InterventionCar], PHRILearner], *args, **kwargs
    ):
        learner = learner_factory(self)
        super().__init__(
            np.array([5.0, 1.5, 10.0, 20.0, 1.0]),
            [(45, 55), (85, 95), (125, 135), (165, 175)][:NUM_INTERVENTIONS],
            learner,
            *args,
            **kwargs,
        )
        self.world_horizon = 440

    def get_feature_descriptions(self) -> dict[str, str]:
        return {
            "speed_desirability": "How close the car's speed is to the desired speed limit. This feature gives a maximum value of 1.0 when the car's speed matches the speed limit exactly, and decreases parabolically as the speed deviates from the optimal value.",
            "lane_alignment": "How well the car stays centered in its lane and maintains proper heading. This feature approaches 1 when the car is perfectly centered in a lane, and decreases quadratically as the car deviates from the lane center.",
            "off_road": "Penalty for driving off the road. This feature gives a value of 1 when the car is within road boundaries, and decreases quadratically as the car moves beyond the left or right limits of the road.",
            "cone_distance": "How well the car maintains safe distance from traffic cones. This feature gives a value of 1 when the car is far from obstacles and approaches 0 when the car gets very close to obstacles.",
            "puddle_distance": "How well the car avoids puddles. This feature gives a value of 1 when the car is far from puddles and approaches 0 when the car gets very close to puddles.",
        }

    def base_weights(self):
        # [Speed, Lane, Off-road, Cone, Puddle]
        return [5.0, 1.5, 10.0, 0.0, 1.0]

    @tf.function
    def features(
        self,
        state: Iterable[Union[tf.Tensor, tf.Variable]],
    ) -> tf.Tensor:
        """
        Features:
            - speed_desirability: Quadratic that peaks at the car's speed limit
            - lane_allignment: Combination of distance to lane median and northward heading.
            - off_road: Penalty for driving off road
            - cone_distance: Exponential penalty for getting close to cones

        For more information on the features see their corresponding implementations
        in the feature_utils file.

        Args:
            state: the state of the world.
            control: the controls of this car in that state.
            consider_other_cars: If False will ignore other cars for feature comps.

        Returns:
            tf.Tensor: the four features this car cares about.
        """

        feats = []

        car_state = state[self.index]

        feats.append(feature_utils.speed_desirability(car_state, self.SPEED_LIMIT))

        lane_widths = np.array([lane.w for lane in self.env.lanes], dtype=np.float32)
        lane_feat = feature_utils.lane_alignment(
            car_state, self.lane_medians_t, self.lane_normals_t, lane_widths
        )
        feats.append(lane_feat)

        feats.append(
            feature_utils.off_road(
                car_state,
                self.env.car_box,
                self.env.left_limit,
                self.env.right_limit,
                lane_widths[0],
            )
        )

        locs_by_type = defaultdict(list)
        wh_by_type = defaultdict(list)
        for i in range(len(self.env.obstacles)):
            locs_by_type[self.env.obstacles[i][0]].append(self.env.obs_locs[i])
            wh_by_type[self.env.obstacles[i][0]].append(self.env.obs_wh[i])

        feats.append(
            feature_utils.distance_to_obstacle(
                car_state, locs_by_type["cone"], wh_by_type["cone"]
            )
        )

        feats.append(
            feature_utils.distance_to_obstacle(
                car_state, locs_by_type["puddle"], wh_by_type["puddle"]
            )
        )

        return tf.stack(feats, axis=-1)


class PuddleAvoidExperiment(InterventionExperiment):
    def __init__(
        self,
        exp_name: str = "puddle_avoid",
        save_dir: str | None = None,
        verbose: bool = False,
    ):
        """Initialize the cone avoidance experiment."""
        super().__init__(exp_name, save_dir, verbose)

    def setup_world(
        self, learner_factory: Callable[[InterventionCar], PHRILearner], seed: int
    ) -> TwoLanePuddleCarWorld:
        """Set up the car world with cones."""
        random.seed(seed)
        cone_positions = [
            [0, 2.0],
            [-0.15, 4.0],
            [0, 6.0],
            [-0.15, 8.0],
            [0, 10.0],
            [0.02, 10.1],
            [-0.02, 10.2],
            [-0.15, 12.0],
            [-0.13, 13.0],
            [0.0, 16.0],
            [-0.15, 17.0],
        ]
        puddle_positions = [
            [-0.1, 2.0],
            [-0.1, 1.8],
            [-0.1, 1.9],
            [-0.1, 2.2],
            [-0.10, 4.5],
            [-0.05, 5.0],
            [-0.02, 6.4],
            [-0.05, 6.5],
            [-0.13, 8.4],
            [-0.15, 9],
            [0.05, 9.1],
            [0.02, 10.4],
            [0.0, 13.5],
            [0.02, 13.6],
            [0.0, 14.5],
            [0.0, 15.7],
            [-0.15, 16.7],
            [0.0, 17.5],
        ]
        for i in range(4, len(cone_positions)):
            cone_positions[i][0] += 0.01 * random.random() - 0.005
            cone_positions[i][1] += 0.1 * random.random() - 0.05
        for i in range(2, len(puddle_positions)):
            puddle_positions[i][0] += 0.01 * random.random() - 0.005
            puddle_positions[i][1] += 0.1 * random.random() - 0.05

        world = TwoLanePuddleCarWorld(
            cone_positions=cone_positions,
            puddle_positions=puddle_positions,
            visualizer_args={
                "name": self.exp_name,
                "display_y": True,
                "follow_main_car": True,
            },
        )

        car = PuddleInterventionCar(
            learner_factory,
            world,
            np.array([0, -0.5, 0.8, np.pi / 2]),
            color="orange",
            planner_type="Naive",
            planner_args={
                "horizon": 5,
                "n_iter": 10,
                "h_index": 1,
            },
        )
        world.add_cars([car])
        return world


def main():
    experiment = PuddleAvoidExperiment(verbose=True)
    experiment.run(lambda car: PHRILearner(car))

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
