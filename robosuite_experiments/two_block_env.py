"""
Custom robosuite environment with red block (target) and green block (obstacle).
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.mjcf_utils import new_site
from robosuite.utils.observables import Observable, sensor


class TwoBlockPickPlace(ManipulationEnv):
    """
    Environment with red block (target) and green block (obstacle) for pick-and-place task.
    
    The robot must move the red block from Zone A to Zone B while avoiding the green block.
    """
    
    def __init__(
        self,
        robots,
        table_full_size=(0.91, 1.5, 0.05),
        table_offset=(0, 0, 0.9),
        **kwargs
    ):
        self.table_full_size = table_full_size
        self.table_offset = np.array(table_offset)
        
        # Initial robot configuration (slightly above table)
        self.initial_configuration = np.array([0, 0.5, 0, -1.5, 0, 2.0, 0.785])
        
        # Remove parameters that ManipulationEnv doesn't accept
        # These are specific to standardized environments
        kwargs.pop('use_object_obs', None)
        kwargs.pop('reward_shaping', None)
        
        # Add ignore_done to prevent early termination
        if 'ignore_done' not in kwargs:
            kwargs['ignore_done'] = True
        
        super().__init__(
            robots=robots,
            **kwargs
        )
    
    def _load_model(self):
        """Builds the MJCF model with table, red block, and green block."""
        super()._load_model()
        
        # Position the robot at table
        base_pos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(base_pos)
        self.robots[0].init_qpos = self.initial_configuration
        
        # Create table arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        
        # Create RED BLOCK (target) - will spawn in Zone A
        self.red_block = BoxObject(
            name="red_block",
            size=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],  # Red
        )
        
        # Create GREEN BLOCK (obstacle) - fixed in center
        self.green_block = BoxObject(
            name="green_block",
            size=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],  # Green
        )
        
        self.objects = [self.red_block, self.green_block]
        
        # Set up placement sampler
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        
        # RED BLOCK: spawns in Zone A (left side of table) with slight randomization
        red_sampler = UniformRandomSampler(
            name="RedBlockSampler",
            mujoco_objects=self.red_block,
            x_range=[-0.03, 0.03],  # Small randomization around Zone A
            y_range=[-0.03, 0.03],
            rotation=None,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset + np.array([-0.2, -0.3, 0.02]),  # Zone A position
            z_offset=0.1,
        )
        self.placement_initializer.append_sampler(red_sampler)
        
        # GREEN BLOCK: fixed in center of table (obstacle on path)
        green_sampler = UniformRandomSampler(
            name="GreenBlockSampler",
            mujoco_objects=self.green_block,
            x_range=[0.0, 0.0],  # No randomization - fixed position
            y_range=[0.0, 0.0],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=self.table_offset + np.array([0.05, -0.05, 0.025]),  
            z_offset=0.0,
        )
        self.placement_initializer.append_sampler(green_sampler)
        
        # Add visual markers for Zone A and Zone B
        robot_base_body = self.robots[0].robot_model.worldbody.find(".//body")
        
        # Zone A marker (red - where red block starts)
        zone_a_marker = new_site(
            name="zone_a_marker",
            pos=np.array([-0.2, -0.3, 0.92]),  # Zone A position
            rgba=(1, 0, 0, 0.3),  # Red, semi-transparent
            size=(0.05,)  # Larger marker
        )
        mujoco_arena.worldbody.append(zone_a_marker)
        
        # Zone B marker (green - target zone)
        zone_b_marker = new_site(
            name="zone_b_marker",
            pos=np.array([0.2, 0.3, 0.92]),  # Zone B position
            rgba=(0, 1, 0, 0.3),  # Green, semi-transparent
            size=(0.05,)
        )

        mujoco_arena.worldbody.append(zone_b_marker)
        
        zone_c_marker = new_site(
            name="zone_c_marker",
            pos=np.array([-0.15, 0.25, 0.92]),  # Zone C position
            rgba=(0, 0, 1, 0.3),  # Blue, semi-transparent
            size=(0.05,)
        )
        mujoco_arena.worldbody.append(zone_c_marker)

        # Create the final task model
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )
    
    def _reset_internal(self):
        """Reset simulation and object positions."""
        super()._reset_internal()
        
        # Reset robot to initial configuration
        self.robots[0].set_robot_joint_positions(self.initial_configuration)
        
        # Sample and set positions for both blocks
        object_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(obj_pos), np.array(obj_quat)])
            )
    
    def reward(self, action=None):
        # We compute our own rewards in the learning framework
        return 0.0
    
    def _check_success(self):
        # We don't use robosuite's success metric
        return False
    
    def _setup_observables(self):
        """
        Set up observables to expose block positions.
        """
        observables = super()._setup_observables()
        
        # Create modality for custom object observations
        pf = self.robots[0].robot_model.naming_prefix
        
        # Observable for red block position
        @sensor(modality=f"{pf}object")
        def red_block_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(self.red_block.root_body)])
        
        # Observable for green block position
        @sensor(modality=f"{pf}object")
        def green_block_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(self.green_block.root_body)])
        
        sensors = [red_block_pos, green_block_pos]
        names = ["red_block_pos", "green_block_pos"]
        
        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )
        
        return observables

