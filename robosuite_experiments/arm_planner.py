"""Base class for robotic arm planners."""

import numpy as np
import tensorflow as tf

from arm_world import ArmWorld
from base_rational_arm import BaseRationalArm


class ArmPlanner:
    """
    Base class for trajectory planning for a robotic arm.
    
    Similar to CarPlanner but adapted for robotic manipulation.
    """
    
    def __init__(self, world: ArmWorld, arm: BaseRationalArm):
        """
        Initialize the planner.
        
        Args:
            world: ArmWorld environment instance
            arm: BaseRationalArm instance with reward function
        """
        self.world = world
        self.arm = arm
        
        self.graph_set_up = False
        self.graphs_ran = set()
        
        # Control dimensionality (3D position commands for OSC controller)
        self.NC = 3  # dx, dy, dz (we'll keep rotation fixed)
        
        self.last_state_computed_for = None
        self.planning_processes = []
        self.process_reqs = {}
        self.next_process_index = 0
    
    def setup_planner(self, horizon: int, n_iter: int, learning_rate: float = 0.03):
        """
        Initialize parameters for the planner.
        
        Args:
            horizon: Planning horizon in timesteps
            n_iter: Number of optimization iterations
            learning_rate: Learning rate for gradient ascent
        """
        self.horizon = horizon
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        
        self.initialize_optimizer()
        
        # Initialize control sequence to zeros
        self.zeros_control = np.zeros(self.NC * self.horizon, dtype=np.float32)
        self.robot_control = tf.Variable(self.zeros_control)
    
    def initialize_optimizer(self):
        """Initialize the optimizer for gradient ascent."""
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
    
    def compute(self, process_name, init_obs=None, void_graph_setup=False):
        """
        Execute planning computation.
        
        Args:
            process_name: Name of the computation process
            init_obs: Initial observation (current state)
            void_graph_setup: If True, only setup graph without full optimization
        """
        if init_obs is None:
            self.init_obs = self.world.get_observation()
        else:
            self.init_obs = init_obs
        
        if process_name in self.process_reqs:
            process_index = self.process_reqs[process_name]
        elif process_name == "plan":
            process_index = len(self.planning_processes)
        else:
            process_index = 0
        
        # Convert observation to flat array for comparison
        flat_obs = self._flatten_obs(self.init_obs)
        
        if self.last_state_computed_for is None or not np.allclose(
            flat_obs, self.last_state_computed_for
        ):
            self.next_process_index = 0
        
        while self.next_process_index < process_index:
            self.planning_processes[self.next_process_index](
                self.init_obs, void_graph_setup
            )
            self.next_process_index += 1
        
        if not self.graph_set_up and void_graph_setup:
            self.next_process_index = 0
        else:
            self.last_state_computed_for = flat_obs
    
    def _flatten_obs(self, obs):
        """Flatten observation dictionary to array for comparison."""
        return np.concatenate([
            obs["ee_pos"],
            obs["ee_vel"],
            obs["red_block_pos"],
            obs["green_block_pos"]
        ])
    
    def initialize_parameters(self, init_obs=None, void_graph_setup=False):
        """Initialize control parameters to zeros."""
        self.graph_set_up = True
        self.robot_control.assign(self.zeros_control)
    
    def generate_plan(self, init_obs=None):
        """
        Generate an optimized plan from the current state.
        
        Args:
            init_obs: Initial observation
            
        Returns:
            List of control actions for each timestep in horizon
        """
        self.compute("plan", init_obs)
        return self.split_robot_control
    
    @property
    def split_robot_control(self):
        """
        Split flat control vector into per-timestep controls.
        
        Returns:
            List of control arrays, one per timestep
        """
        return [
            self.robot_control[t * self.NC : (t + 1) * self.NC]
            for t in range(self.horizon)
        ]
    
    def report_run(self, func_key):
        """Report that a computational graph has been run."""
        if func_key in self.graphs_ran:
            return
        
        self.graphs_ran.add(func_key)
        self.graph_set_up = False
        
        if hasattr(self.world, 'verbose') and self.world.verbose:
            print(f"Arm Planner {func_key} Graph Set Up")

