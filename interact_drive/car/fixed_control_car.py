"""Module containing a fixed control car."""

from typing import TYPE_CHECKING, Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.car.car import Car

if TYPE_CHECKING:
    from interact_drive.world import CarWorld


class FixedControlCar(Car):
    """
    A car where the controls are fixed.
    """

    def __init__(
        self,
        env: "CarWorld",
        init_state: Union[np.ndarray, tf.Tensor, Iterable],
        control: Union[np.ndarray, tf.Tensor, Iterable],
        color: str = "gray",
        opacity: float = 1.0,
        **kwargs,
    ):
        super().__init__(env, init_state, color, opacity, **kwargs)
        self.control = tf.constant(control, dtype=tf.float32)
        self.control_already_determined_for_current_step = True

    def step(self, dt):
        self.state = self.dynamics_fn(self.state, self.control, dt)

    @tf.function
    def reward_fn(self, world_state, self_control):
        return 0

    def _get_next_control(self):
        return self.control
