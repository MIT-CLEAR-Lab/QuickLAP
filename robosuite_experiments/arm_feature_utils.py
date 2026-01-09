"""
Feature utility functions for robotic arm manipulation tasks.

These functions compute various features from the arm's state that can be used
to define reward functions for learning from human interventions.
"""

import numpy as np
import tensorflow as tf


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


def distance_to_green_block(ee_pos, green_block_pos):
    """
    Compute proximity feature to green block (obstacle).
    
    Returns exponential proximity value - higher when closer to green block.
    Value approaches 1 when very close, approaches 0 when far away.
    
    Args:
        ee_pos: End-effector position (3D vector)
        green_block_pos: Green block position (3D vector)
        
    Returns:
        Feature value in [0, 1] range
    """
    if isinstance(ee_pos, np.ndarray):
        distance = np.linalg.norm(ee_pos - green_block_pos)
        return float(np.exp(-3.0 * distance))
    else:
        distance = tf.norm(ee_pos - green_block_pos)
        return tf.exp(-3.0 * distance)


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


def proximity_to_obstacle_zone(ee_pos, zone_c_pos):
    """
    Compute proximity of end-effector to obstacle zone C.
    
    Returns exponential proximity value - higher when EE is closer to zone C.
    This feature should be given a NEGATIVE weight to encourage avoiding zone C.
    
    Args:
        ee_pos: End-effector position (3D vector)
        zone_c_pos: Obstacle zone C position (3D vector)
        
    Returns:
        Feature value in [0, 1] range (higher = closer = worse)
    """
    if isinstance(ee_pos, np.ndarray):
        distance = np.linalg.norm(ee_pos - zone_c_pos)
        return float(np.exp(-3.0 * distance))
    else:
        distance = tf.norm(ee_pos - zone_c_pos)
        return tf.exp(-3.0 * distance)


# =============================================================================
# CENTRALIZED FEATURE SYSTEM
# =============================================================================

def compute_features(obs, include_red_dist=False, include_zones=False):
    """
    Compute all features from observation (SINGLE SOURCE OF TRUTH).
    
    This is the centralized feature computation that all agents should use.
    
    Feature Sets:
    - Legacy (5 features): include_red_dist=True, include_zones=False
      [red_dist, green_dist, velocity, collision, joints]
    - Pick-and-Place (6 features): include_red_dist=False, include_zones=True
      [green_dist, velocity, collision, joints, block_to_zone, zone_c_prox]
    
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
    feat_green_dist = distance_to_green_block(ee_pos, green_block_pos)
    feat_velocity = end_effector_velocity(ee_vel)
    feat_collision = collision_safety(ee_pos, green_block_pos)
    feat_joints = joint_safety(joint_pos, joint_limits)
    
    features.extend([
        feat_green_dist,
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
            feat_zone_c_proximity = proximity_to_obstacle_zone(ee_pos, zone_c_pos)
            features.append(feat_zone_c_proximity)
    
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
        "distance_to_green_block": "Distance from end effector to the green block (obstacle on path). Higher values mean the robot is moving closer to the green block, which may indicate avoidance behavior or confusion.",
        "velocity": "Speed of end effector movement. Higher values mean faster motion. Increasing this weight makes the robot move more quickly to complete the task.",
        "collision_safety": "Safety penalty for getting too close to obstacles (green block). Higher values mean more conservative collision avoidance.",
        "joint_safety": "Safety penalty for joint configurations near limits. Higher values mean more conservative joint movements.",
    })
    
    # Zone-based features (pick-and-place)
    if include_zones:
        descriptions["block_to_target_zone"] = "Horizontal proximity of the red block to the target zone B (ignores height). Higher values mean the block is closer to where it needs to be placed in the X-Y plane. This is the key feature for successful task completion during the transport phase."
        descriptions["zone_c_proximity"] = "Proximity of end effector to obstacle zone C. Higher values mean getting closer to zone C. This should have a NEGATIVE weight to encourage avoiding zone C during transport."
    
    return descriptions

