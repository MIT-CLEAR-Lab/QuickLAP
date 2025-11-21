"""
Cone avoidance experiment with simulated expert trajectory.
"""

import random
from typing import Callable

import numpy as np

from experiments.experiment import InterventionExperiment
from experiments.intervention_car import InterventionCar, NUM_INTERVENTIONS
from interact_drive.learner.phri_learner import PHRILearner
from interact_drive.world import TwoLaneConeCarWorld


class ConeInterventionCar(InterventionCar):
    """HybridCar with simulated expert guidance for cone avoidance."""

    def __init__(
        self, learner_factory: Callable[[InterventionCar], PHRILearner], *args, **kwargs
    ):
        learner = learner_factory(self)
        super().__init__(
            np.array([10.0, 2.5, 20.0, 40.0]),
            [(45, 55), (85, 95), (125, 135), (165, 175)][:NUM_INTERVENTIONS],
            learner,
            *args,
            **kwargs,
        )
        self.world_horizon = 440

    def base_weights(self):
        # [Speed, Lane, Off-road, Cone]
        return [10.0, 2.5, 20.0, 1.0]

    # Features are the same as default


class ConeAvoidExperiment(InterventionExperiment):
    def __init__(
        self,
        exp_name: str = "cone_avoid",
        save_dir: str | None = None,
        verbose: bool = False,
    ):
        """Initialize the cone avoidance experiment."""
        super().__init__(exp_name, save_dir, verbose)

    def setup_world(
        self, learner_factory: Callable[[InterventionCar], PHRILearner], seed: int
    ) -> TwoLaneConeCarWorld:
        """Set up the car world with cones."""
        random.seed(seed)
        cone_positions = [
            [0.0, 2.0],
            [-0.15, 4.0],
            [0, 6.0],
            [-0.15, 8.0],
            [0, 10.0],
            [-0.15, 12.0],
            [-0.15, 13.0],
            [0.0, 14.0],
            [-0.2, 15.0],
            [-0.15, 16.0],
            [0.0, 17.0],
            [-0.15, 18.0],
        ]
        for i in range(4, len(cone_positions)):
            cone_positions[i][0] += 0.01 * random.random() - 0.005
            cone_positions[i][1] += 0.1 * random.random() - 0.05

        world = TwoLaneConeCarWorld(
            cone_positions=cone_positions,
            visualizer_args={
                "name": self.exp_name,
                "display_y": True,
                "follow_main_car": True,
            },
        )

        car = ConeInterventionCar(
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
    experiment = ConeAvoidExperiment(verbose=True)
    experiment.run(lambda car: PHRILearner(car))

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
