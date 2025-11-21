from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive import feature_utils
from interact_drive.car.linear_reward_car import LinearRewardCar
from interact_drive.car.planner_car import PlannerCar


class BaseRationalCar(LinearRewardCar, PlannerCar):
    """
    Vanilla car that has no ulterior motive.  It simply
    wants to drive forward, ideally centered in some lane,
    and not crash.

    Further it doesn't want to drive faster than it's own
    personal speed limit which is set to 0.6 by default.
    """

    def __init__(
        self,
        env,
        init_position,
        color,
        planner_type=None,
        planner_args=None,
        speed_limit=0.5,
        custom_weights=None,
        friction=0.2,
        opacity=0.8,
        init_speed=None,
        init_heading=np.pi / 2,
    ):
        if custom_weights:
            weights = custom_weights
        else:
            weights = self.base_weights()

        if planner_type is None:
            # Default planner is naive with switching disabled.
            exp_params = {"horizon": 5, "n_iter": 20, "h_index": 1}
            # Naive planner by default
            planner_type = "Naive"
            planner_args = {"planner_specific_args": {"Naive": exp_params}}

        weights = np.array(weights)

        self.SPEED_LIMIT = speed_limit
        # If initial speed not provided use speed limit
        if init_speed is None:
            init_speed = self.SPEED_LIMIT

        init_state = init_position
        init_state[2] = init_speed
        init_state[3] = init_heading

        super().__init__(
            env,
            init_state,
            weights=weights,
            color=color,
            friction=friction,
            opacity=opacity,
            planner_type=planner_type,
            planner_args=planner_args,
        )

    def base_weights(self):
        """
        Base weights for the features that together result
        in standard car behavior.  Used across experiments.

        For information on features see below
        """
        # [Speed, Lane, Off-road, Cone]
        return [10.0, 2.5, 20.0, 40.0]

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

        feats.append(
            feature_utils.distance_to_obstacle(
                car_state, self.env.obs_locs, self.env.obs_wh
            )
        )

        return tf.stack(feats, axis=-1)
