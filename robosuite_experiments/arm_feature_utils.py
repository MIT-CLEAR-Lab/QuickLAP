"""
Feature utility functions for robotic arm manipulation tasks.

These functions compute various features from the arm's state that can be used
to define reward functions for learning from human interventions.
"""

import numpy as np
import tensorflow as tf


# =============================================================================
# CENTRALIZED WEIGHT CONSTANTS (SINGLE SOURCE OF TRUTH)
# =============================================================================
# Feature order for pick-and-place (7 features):
# [green_clearance, velocity, collision, joints, block_to_zone, zone_c_clearance, height_maintain]
#
# Feature semantics (all use POSITIVE weights for desired behavior):
#   - green_clearance: Higher = FARTHER from green block. Use POSITIVE weight to avoid.
#   - velocity: Higher = closer to target speed. Use POSITIVE weight for faster motion.
#   - collision: Higher = safer (farther from obstacle). Use POSITIVE weight.
#   - joints: Higher = safer (farther from joint limits). Use POSITIVE weight.
#   - block_to_zone: Higher = block closer to target zone. Use POSITIVE weight.
#   - zone_c_clearance: Higher = FARTHER from obstacle zone C. Use POSITIVE weight to avoid.
#   - height_maintain: Higher = closer to target transport height. Use POSITIVE weight.

FEATURE_NAMES = [
    "green_clearance", "velocity", "collision", "joints", 
    "block_to_zone", "zone_c_clearance", "height_maintain"
]

# Default base weights for robot (before learning)
DEFAULT_BASE_WEIGHTS = np.array([
    0.0,    # green_clearance: avoid green block (positive = stay far)
    1.0,    # velocity: move at good speed
    1.0,    # collision: stay safe (but constant at large distances)
    1.0,    # joints: stay away from joint limits
    20.0,   # block_to_zone: move block toward target (main objective)
    0.0,    # zone_c_clearance: avoid obstacle zone C
    1.0,    # height_maintain: maintain transport height
], dtype=np.float32)

# Default expert weights (what the human demonstrator physical input represents)
DEFAULT_EXPERT_WEIGHTS = np.array([
    15.0,   # green_clearance: strongly avoid green block (positive = stay far)
    3.0,    # velocity: expert moves faster
    1.0,    # collision: stay safe
    1.0,    # joints: stay safe
    20.0,   # block_to_zone: strongly prioritize moving to target
    -10.0,   # zone_c_clearance: strongly avoid obstacle zone C (positive = stay far)
    1.0,    # height_maintain: maintain transport height
], dtype=np.float32)

# Transport height for the height maintenance feature (meters)
DEFAULT_TRANSPORT_HEIGHT = 1.02

# =============================================================================
# SPATIAL LAYOUT CONTEXT (for LLM prompts)
# =============================================================================
# This provides the LLM with spatial understanding of the environment
# to help distinguish intentional actions from geometric side effects.

SPATIAL_LAYOUT_CONTEXT = """
Environment Spatial Layout:
- Table surface is at z=0.95m
- Zone A (start): x=-0.3, y=-0.3 
- Zone B (target): x=0.1, y=0.3 
- Zone C (obstacle zone): x=-0.25, y=0.15 
- Green block (obstacle): x=-0.05, y=0.05 
"""


def distance_to_red_block(ee_pos, red_block_pos):
    """
    Compute proximity feature to red block (target object).
    
    Returns exponential proximity value - higher when closer to red block.
    Value approaches 1 when very close, approaches 0 when far away.
    
    Args:
        ee_pos: End-effector position (3D vector)
        red_block_pos: Red block position (3D vector)
        
    Returns:
        Feature value in [0, 1] range
    """
    if isinstance(ee_pos, np.ndarray):
        distance = np.linalg.norm(ee_pos - red_block_pos)
        return float(np.exp(-3.0 * distance))
    else:
        distance = tf.norm(ee_pos - red_block_pos)
        return tf.exp(-3.0 * distance)


def clearance_from_green_block(ee_pos, green_block_pos):
    """
    Compute clearance feature from green block (obstacle).
    
    Returns exponential clearance value - higher when FARTHER from green block.
    Value approaches 0 when very close, approaches 1 when far away.
    A POSITIVE weight encourages the robot to stay far from the obstacle.
    
    NOTE: Uses HORIZONTAL (X-Y) distance only, ignoring Z (height).
    This is more relevant during transport when the robot is at a different
    height than the obstacle.
    
    Args:
        ee_pos: End-effector position (3D vector)
        green_block_pos: Green block position (3D vector)
        
    Returns:
        Feature value in [0, 1] range (higher = farther = safer)
    """
    if isinstance(ee_pos, np.ndarray):
        # Use only X-Y distance (ignore Z/height)
        horizontal_distance = np.linalg.norm(ee_pos[:2] - green_block_pos[:2])
        return float(1.0 - np.exp(-3.0 * horizontal_distance))
    else:
        # TensorFlow version - use only X-Y distance
        horizontal_distance = tf.norm(ee_pos[:2] - green_block_pos[:2])
        return 1.0 - tf.exp(-3.0 * horizontal_distance)


def end_effector_velocity(ee_vel, target_speed=0.3):
    """
    Compute velocity desirability feature.
    
    Returns quadratic desirability that peaks at target_speed.
    Value is 1.0 when at target speed, decreases as speed deviates.
    
    Args:
        ee_vel: End-effector velocity (3D vector)
        target_speed: Desired speed magnitude in m/s
        
    Returns:
        Feature value in [0, 1] range
    """
    if isinstance(ee_vel, np.ndarray):
        speed = np.linalg.norm(ee_vel)
        # Quadratic that peaks at target_speed
        deviation = np.abs(speed - target_speed)
        return float(np.exp(-2.0 * deviation))
    else:
        speed = tf.norm(ee_vel)
        deviation = tf.abs(speed - target_speed)
        return tf.exp(-2.0 * deviation)


def collision_safety(ee_pos, green_block_pos, threshold=0.1):
    """
    Compute collision avoidance safety feature.
    
    Returns safety value - higher when farther from obstacle.
    Value approaches 0 when very close (unsafe), approaches 1 when far (safe).
    
    Args:
        ee_pos: End-effector position (3D vector)
        green_block_pos: Green block position (3D vector)
        threshold: Minimum safe distance in meters
        
    Returns:
        Feature value in [0, 1] range
    """
    if isinstance(ee_pos, np.ndarray):
        distance = np.linalg.norm(ee_pos - green_block_pos)
        if distance < threshold:
            # Exponential penalty when too close
            return float(np.exp(-10.0 * (threshold - distance)))
        else:
            return 1.0
    else:
        distance = tf.norm(ee_pos - green_block_pos)
        # Smooth version using sigmoid
        return tf.sigmoid(10.0 * (distance - threshold))


def joint_safety(joint_pos, joint_limits):
    """
    Compute joint limit safety feature.
    
    Returns safety value - higher when farther from joint limits.
    Value decreases quadratically as joints approach their limits.
    
    Args:
        joint_pos: Current joint positions (N-dimensional vector)
        joint_limits: Joint limits as (min_limits, max_limits) tuple
        
    Returns:
        Feature value in [0, 1] range
    """
    min_limits, max_limits = joint_limits
    
    if isinstance(joint_pos, np.ndarray):
        # Compute normalized distance to nearest limit
        dist_to_lower = (joint_pos - min_limits) / (max_limits - min_limits)
        dist_to_upper = (max_limits - joint_pos) / (max_limits - min_limits)
        
        # Take minimum distance for each joint
        min_dist = np.minimum(dist_to_lower, dist_to_upper)
        
        # Average penalty across all joints
        # Penalty increases as we get closer to limits
        penalty = np.mean(np.exp(-5.0 * min_dist))
        return float(1.0 - penalty)
    else:
        # TensorFlow version
        dist_to_lower = (joint_pos - min_limits) / (max_limits - min_limits)
        dist_to_upper = (max_limits - joint_pos) / (max_limits - min_limits)
        
        min_dist = tf.minimum(dist_to_lower, dist_to_upper)
        penalty = tf.reduce_mean(tf.exp(-5.0 * min_dist))
        return 1.0 - penalty


def distance_block_to_target_zone(red_block_pos, zone_b_pos):
    """
    Compute proximity of red block to target zone B.
    
    Returns exponential proximity value - higher when block is closer to zone B.
    This feature is key for the transport phase where the goal is to move
    the block to the target zone.
    
    NOTE: Uses HORIZONTAL (X-Y) distance only, ignoring Z (height).
    During transport, the block is held at a constant height. The robot will
    descend to place the block in a separate phase.
    
    Args:
        red_block_pos: Red block position (3D vector)
        zone_b_pos: Target zone B position (3D vector)
        
    Returns:
        Feature value in [0, 1] range
    """
    if isinstance(red_block_pos, np.ndarray):
        # Use only X-Y distance (ignore Z/height)
        horizontal_distance = np.linalg.norm(red_block_pos[:2] - zone_b_pos[:2])
        return float(np.exp(-3.0 * horizontal_distance))
    else:
        # TensorFlow version - use only X-Y distance
        horizontal_distance = tf.norm(red_block_pos[:2] - zone_b_pos[:2])
        return tf.exp(-3.0 * horizontal_distance)


def clearance_from_obstacle_zone(ee_pos, zone_c_pos):
    """
    Compute clearance of end-effector from obstacle zone C.
    
    Returns exponential clearance value - higher when EE is FARTHER from zone C.
    A POSITIVE weight encourages the robot to stay far from zone C.
    
    Args:
        ee_pos: End-effector position (3D vector)
        zone_c_pos: Obstacle zone C position (3D vector)
        
    Returns:
        Feature value in [0, 1] range (higher = farther = safer)
    """
    if isinstance(ee_pos, np.ndarray):
        distance = np.linalg.norm(ee_pos - zone_c_pos)
        return float(1.0 - np.exp(-3.0 * distance))
    else:
        distance = tf.norm(ee_pos - zone_c_pos)
        return 1.0 - tf.exp(-3.0 * distance)


def maintain_transport_height(ee_pos, target_height=None):
    """
    Compute height maintenance feature for transport phase.
    
    Returns exponential value - higher when closer to target transport height.
    This helps the robot maintain a consistent height during transport,
    preventing drift in the Z direction.
    
    Args:
        ee_pos: End-effector position (3D vector)
        target_height: Target height in meters (defaults to DEFAULT_TRANSPORT_HEIGHT)
        
    Returns:
        Feature value in [0, 1] range (higher = closer to target height)
    """
    if target_height is None:
        target_height = DEFAULT_TRANSPORT_HEIGHT
    
    if isinstance(ee_pos, np.ndarray):
        height_error = np.abs(ee_pos[2] - target_height)
        return float(np.exp(-5.0 * height_error))
    else:
        height_error = tf.abs(ee_pos[2] - target_height)
        return tf.exp(-5.0 * height_error)


# =============================================================================
# CENTRALIZED FEATURE SYSTEM
# =============================================================================

def compute_features(obs, include_red_dist=False, include_zones=False, transport_height=None):
    """
    Compute all features from observation (SINGLE SOURCE OF TRUTH).
    
    This is the centralized feature computation that all agents should use.
    
    Feature Sets:
    - Legacy (5 features): include_red_dist=True, include_zones=False
      [red_dist, green_clearance, velocity, collision, joints]
    - Pick-and-Place (7 features): include_red_dist=False, include_zones=True
      [green_clearance, velocity, collision, joints, block_to_zone, zone_c_clearance, height_maintain]
    
    Args:
        obs: Observation dictionary containing:
            - ee_pos: End-effector position
            - ee_vel: End-effector velocity
            - joint_pos: Joint positions
            - red_block_pos: Red block position (optional, if include_red_dist=True)
            - green_block_pos: Green block (obstacle) position
            - joint_limits: Joint limits (min, max) tuple
            - zone_b_pos: Target zone B position (optional, if include_zones=True)
            - zone_c_pos: Obstacle zone C position (optional, if include_zones=True)
        include_red_dist: Whether to include distance to red block (legacy feature)
        include_zones: Whether to include zone-based features (for pick-and-place)
        transport_height: Target height for height maintenance feature (defaults to DEFAULT_TRANSPORT_HEIGHT)
        
    Returns:
        numpy array of feature values
    """
    ee_pos = obs["ee_pos"]
    ee_vel = obs["ee_vel"]
    joint_pos = obs["joint_pos"]
    green_block_pos = obs["green_block_pos"]
    joint_limits = obs["joint_limits"]
    
    features = []
    
    # Legacy feature: distance to red block
    if include_red_dist:
        red_block_pos = obs.get("red_block_pos")
        if red_block_pos is not None:
            feat_red_dist = distance_to_red_block(ee_pos, red_block_pos)
            features.append(feat_red_dist)
    
    # Core features (always present)
    feat_green_clearance = clearance_from_green_block(ee_pos, green_block_pos)
    feat_velocity = end_effector_velocity(ee_vel)
    feat_collision = collision_safety(ee_pos, green_block_pos)
    feat_joints = joint_safety(joint_pos, joint_limits)
    
    features.extend([
        feat_green_clearance,
        feat_velocity,
        feat_collision,
        feat_joints,
    ])
    
    # Zone-based features (for pick-and-place tasks)
    if include_zones:
        red_block_pos = obs.get("red_block_pos")
        zone_b_pos = obs.get("zone_b_pos")
        zone_c_pos = obs.get("zone_c_pos")
        
        if red_block_pos is not None and zone_b_pos is not None:
            feat_block_to_zone = distance_block_to_target_zone(red_block_pos, zone_b_pos)
            features.append(feat_block_to_zone)
        
        if zone_c_pos is not None:
            feat_zone_c_clearance = clearance_from_obstacle_zone(ee_pos, zone_c_pos)
            features.append(feat_zone_c_clearance)
        
        # Height maintenance feature for transport phase
        feat_height = maintain_transport_height(ee_pos, transport_height)
        features.append(feat_height)
    
    return np.array(features, dtype=np.float32)


def get_feature_descriptions(include_red_dist=False, include_zones=False):
    """
    Get natural language descriptions of features (SINGLE SOURCE OF TRUTH).
    
    This is the centralized feature description that all learners should use.
    
    Args:
        include_red_dist: Whether to include distance to red block (legacy feature)
        include_zones: Whether to include zone-based feature descriptions
        
    Returns:
        Dictionary mapping feature names to descriptions
    """
    descriptions = {}
    
    # Legacy feature
    if include_red_dist:
        descriptions["distance_to_red_block"] = "Distance from end effector to the red block (target object being moved). Higher values mean staying closer to the red block during manipulation."
    
    # Core features
    descriptions.update({
        "green_clearance": "Clearance from the green block (obstacle). HIGHER values mean FARTHER from the obstacle. Increasing this weight makes the robot stay farther away; decreasing it makes the robot get closer.",
        "velocity": "Speed of end effector movement. Higher values mean faster motion. Increasing this weight makes the robot move faster.",
        "collision_safety": "Safety penalty for getting too close to obstacles (green block). Higher values mean more conservative collision avoidance.",
        "joint_safety": "Safety penalty for joint configurations near limits. Higher values mean more conservative joint movements.",
    })
    
    # Zone-based features (pick-and-place)
    if include_zones:
        descriptions["block_to_target_zone"] = "Horizontal proximity of the red block to the GOAL target zone B (ignores height). Higher values mean the block is closer to where it needs to be placed in the X-Y plane. This is the key feature for SUCCESSFUL task completion during the transport phase."
        descriptions["zone_c_clearance"] = "Clearance from obstacle zone C. HIGHER values mean FARTHER from zone C. Increasing this weight makes the robot stay farther away; decreasing it makes the robot get closer."
        descriptions["height_maintain"] = "Height maintenance during transport. Higher values mean the end effector is closer to the target transport height. This prevents vertical drift during horizontal transport and ensures stable block carrying."
    
    return descriptions

