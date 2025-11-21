"""Defines useful features for feature-based cars."""

import tensorflow as tf


@tf.function
def lane_alignment(car_state, lane_medians, lane_normals, lane_widths):
    sqd_lane_dists = tf.reduce_sum(
        ((car_state[:2] - lane_medians) * lane_normals) ** 2, axis=1
    )
    lane_rewards = 1.0 - (sqd_lane_dists / (lane_widths**2 / 4)) ** 2
    return tf.maximum(0.0, tf.reduce_max(lane_rewards))


"""
Parameter controlling penalty for driving off road.
We begin to penalize the car if it is more than
OFF_ROAD_BUFFER further out than the leftmost
or rightmost lane medians.
"""
OFF_ROAD_BUFFER = -0.04


@tf.function
def off_road(car_state, car_box, left_limit, right_limit, lane_width):
    """
    Args:
        car_state: the state of the car
        car_box: Tensor containing the offsets of car corners
        left_limit: Central x value of leftmost lane median
        right_limit: Central x value of rightmost lane median


    Returns:
        tf.Tensor: off road feature 0 if in road, linearly positive if off road
    """

    cos_t = tf.cos(car_state[3])
    sin_t = tf.sin(car_state[3])
    car_rotation = [[cos_t, -sin_t], [sin_t, cos_t]]

    car_pos = tf.reshape(car_state[:2], (2, 1))

    car_corners_x = (tf.matmul(car_rotation, car_box) + car_pos)[0]

    off_road_amt = tf.reduce_max(
        [
            tf.reduce_max(car_corners_x) - (right_limit + OFF_ROAD_BUFFER),
            (left_limit - OFF_ROAD_BUFFER) - tf.reduce_min(car_corners_x),
            0.0,
        ]
    )

    return (
        1 - off_road_amt**2 / (lane_width**2)
        if off_road_amt < lane_width
        else 2 - 2 * off_road_amt / lane_width
    )


@tf.function
def speed_desirability(car_state, speed_limit=0.6):
    """
    Args:
        car_state: the state of the car

    Returns:
        tf.Tensor: Parabolic desirability which is maximal (1.0) at speed_limit.
    """
    f_vel = car_state[2] * tf.sin(car_state[3])
    desirability = -f_vel * (f_vel - 2 * speed_limit) / (speed_limit**2)

    return desirability


@tf.function
def distance_to_obstacle(car_state, obs_locs, obs_wh):
    """
    Computes a feature that encourages maintaining an appropriate distance from obstacles

    Args:
        car_state: [x, y, vel, heading] tensor of car state
        obs_locs: [N, 2] tensor of obstacle locations
        obs_wh: [N, 2] tensor of obstacle width/height

    Returns:
        tf.Tensor: Feature value encouraging proper cone avoidance distance
    """
    # Constants
    REPULSION_DISTANCE = 0.4
    HORIZONTAL_SENSITIVITY = 3.0

    if not obs_locs:
        return tf.constant(1.0, dtype=tf.float32)

    car_pos = car_state[:2]
    reward = tf.constant(1.0, dtype=tf.float32)
    for loc, obs_wh in zip(obs_locs, obs_wh):
        rel_position = tf.squeeze(loc) - car_pos
        if rel_position[1] < 0:
            continue
        distance = tf.maximum(
            0.0, tf.minimum(REPULSION_DISTANCE, rel_position[1] - obs_wh[1] / 2)
        )
        # How far away from the edge of the obstacle the car is, normalized by obstacle width
        horizontal_separation = (abs(rel_position[0]) - obs_wh[0] / 2 - 0.01) / obs_wh[
            0
        ]

        reward = tf.minimum(
            reward,
            1
            - (1 - distance / REPULSION_DISTANCE)
            / (1 + tf.exp(HORIZONTAL_SENSITIVITY * horizontal_separation)),
        )

    return tf.squeeze(reward)


@tf.function
def distance_to_car(
    world_state,
    car_state,
    car_index,
    car_wh,
):
    """
    Computes a feature that encourages maintaining an appropriate distance from other cars,
    using the exact same “ahead‐only” repulsion logic as your cone function.

    Args:
        world_state:  [N, ≥2] tensor or list of car states (each at least [x, y, …])
        car_state:    [x, y, vel, heading] tensor of ego‐car state
        car_index:    index of the ego car in world_state
        car_wh:       [width, length] for ego car (we’ll reuse for all cars)
        car_box:      unused (kept for API consistency)
        obs_locs:     unused (kept for API consistency)
        obs_wh:       unused (kept for API consistency)
        obs_ang:      unused (kept for API consistency)
        consider_other_cars: whether to apply repulsion at all

    Returns:
        tf.Tensor scalar ∈ [0,1], where 1.0 = fully safe (no car too close ahead),
        and →0 as you drive into another car’s bounding box.
    """

    REPULSION_DISTANCE = 0.5
    HORIZONTAL_SENSITIVITY = 4.2

    if not tf.is_tensor(world_state):
        if not world_state:
            return tf.constant(1.0, tf.float32)
        world_state = tf.stack(world_state)

    num_cars = tf.shape(world_state)[0]
    if num_cars <= 1:
        return tf.constant(1.0, tf.float32)

    car_pos = car_state[:2]  # ego [x,y]
    reward = tf.constant(1.0, tf.float32)
    w, ℓ = tf.squeeze(car_wh)[1], tf.squeeze(car_wh)[0]

    for i in tf.range(num_cars):
        # skip self
        if tf.equal(i, tf.cast(car_index, i.dtype)):
            continue

        other = world_state[i]
        other_pos = other[:2]  # [x,y] of other car
        # tf.print("Other car", i, other_pos, car_pos)
        rel = other_pos - car_pos  # vector from ego → other
        dx, dy = rel[0], rel[1]

        # distance from ego to *front face* of other car, capped
        distance = tf.minimum(REPULSION_DISTANCE, tf.abs(dy) - ℓ)

        # normalized side gap (0 = touching, >0 = widths away)
        horizontal_separation = (tf.abs(dx) - w) / w

        # tf.print(i, distance, horizontal_separation)

        # same smooth‐min penalty:
        #   big penalty if too close ahead, →1 if clear
        current_reward = 1.0 - (
            (1.0 - distance / REPULSION_DISTANCE)
            / (1.0 + tf.exp(HORIZONTAL_SENSITIVITY * horizontal_separation))
        )
        reward = tf.minimum(reward, current_reward)

    # tf.print(reward)
    return tf.squeeze(reward)
