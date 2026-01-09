# Robosuite Pick-and-Place with Hierarchical MPC

A robotic manipulation system that combines phase-based task sequencing with Model Predictive Control (MPC) for learning from human interventions. The robot learns to perform pick-and-place tasks by observing human corrections during task execution.

## Overview

This system implements a **hierarchical control strategy** for robotic manipulation:
- **Phase-based control** for discrete actions (grasping, releasing)
- **MPC optimization** for continuous movement (transport phase)
- **PHRI learning** to update reward weights from human demonstrations

### Key Features

**Hybrid Control Architecture**: Reliable phase-based grasping + intelligent MPC transport  
**Reward-Based Learning**: Linear reward function with 6 interpretable features  
**Human-in-the-Loop**: Learn from physical interventions during execution  
**Robosuite Integration**: Realistic physics simulation with OSC_POSE control  

---

## Quick Start

### 1. Run with Visualization

```bash
cd /path/to/QuickLAP
source venv/bin/activate
python robosuite_experiments/visualize_experiment.py
```

This will:
- Open a simulation window showing the robot arm
- Execute a pick-and-place task with one intervention
- Display debug output showing phase transitions and distances
- Save logs to `robosuite_experiments/logs/`

### 2. Run Batch Experiments

```bash
python robosuite_experiments/run_experiments.py
```

This runs multiple experiments with different configurations and saves results for analysis.

---

## Architecture

### Task Flow

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE-BASED CONTROL (Direct Control)                       │
├─────────────────────────────────────────────────────────────┤
│  1. approach    → Move above red block (open gripper)       │
│  2. descend     → Lower to grasp height                     │
│  3. grasp       → Close gripper, wait 80 steps              │
│  4. lift        → Lift block 15cm up                        │
├─────────────────────────────────────────────────────────────┤
│  MPC CONTROL (Optimization-Based)                           │
├─────────────────────────────────────────────────────────────┤
│  5. transport   → MPC optimizes path to zone B ⭐           │
│                   (Uses reward function with learned weights)│
├─────────────────────────────────────────────────────────────┤
│  PHASE-BASED CONTROL (Direct Control)                       │
├─────────────────────────────────────────────────────────────┤
│  6. place_descend → Lower to 3cm above zone B               │
│  7. release       → Open gripper, wait 40 steps             │
│  8. retract       → Lift arm away                           │
└─────────────────────────────────────────────────────────────┘
```

### Why Hybrid Control?

**Phase-based control** is best for:
- Discrete actions (gripper open/close)
- Precise positional requirements (grasp height)
- Sequential task structure

**MPC** excels at:
- Continuous motion optimization
- Respecting learned preferences (velocity, efficiency)
- Online replanning with feedback

---

## Reward Function

The robot's behavior is guided by a **linear reward function**:

```
reward = w₁·f₁ + w₂·f₂ + w₃·f₃ + w₄·f₄ + w₅·f₅ + w₆·f₆
```

### Features

| # | Feature | Description | Good Value |
|---|---------|-------------|------------|
| 1 | `distance_to_red_block` | Proximity to red block | High (close) |
| 2 | `distance_to_green_block` | Distance from obstacle | Low (far) |
| 3 | `end_effector_velocity` | Movement speed | High (fast) |
| 4 | `collision_safety` | Avoidance of green block | High (safe) |
| 5 | `joint_safety` | Joint limit compliance | High (safe) |
| 6 | `distance_block_to_target_zone` | Block proximity to zone B | High (close) |

### Weight Learning

The system learns weights through **Physical Human-Robot Interaction (PHRI)**:

1. **Robot executes** task with initial weights
2. **Human intervenes** by taking control (e.g., "go faster")
3. **System compares** robot vs. human trajectories
4. **Weights update** to better match human preferences

**Example**: If human moves faster than robot, `w₃` (velocity) increases.

---

## File Structure

### Core Implementation

```
robosuite_experiments/
├── hierarchical_mpc_arm.py        Main implementation (hybrid control)
├── mpc_arm_planner.py             MPC planner with gradient-based optimization
├── arm_planner.py                 Base planner class
└── intervention_arm.py            phase-based control (reference)
```

### Supporting Components

```
├── base_rational_arm.py           Base arm class with reward computation
├── arm_feature_utils.py           Feature computation functions
├── arm_world.py                   Environment wrapper
├── two_block_env.py               Robosuite environment setup
└── robosuite_phri_learner.py      PHRI learning algorithm
```

### Experiment Scripts

```
├── visualize_experiment.py         Run with visualization (single episode)
├── pick_place_experiment.py       Experiment runner
└── run_experiments.py             Batch experiments
```

---

## Key Concepts

### 1. MPC Dynamics

During MPC rollout, the **block moves with the end-effector**:

```python
# Initial grasp offset
grasp_offset = red_block_pos - ee_pos

# During rollout (each timestep):
ee_pos = ee_pos + ee_vel * dt       # Arm moves
red_block_pos = ee_pos + grasp_offset  # Block follows arm!
```

This allows `distance_block_to_target_zone` to change during planning, providing gradients for optimization.

### 2. MPC Initialization

**Critical for performance**: MPC initializes control sequence toward the subgoal:

```python
# Transport phase subgoal
subgoal = zone_b_pos + [0, 0, 0.15]  # 15cm above zone B

# MPC initializes toward subgoal (not red block starting position!)
action = planner.get_next_action(obs, gripper_state, target_pos=subgoal)
```

### 3. Weight Boosting During Transport

To prioritize reaching the target zone during transport:

```python
if self.task_phase == "transport":
    self.weights[5] = 10.0  # Boost block_to_zone weight!
```

This ensures MPC strongly prefers moving toward zone B over other objectives.

---

## Customization

### Change Task Phases

Edit `hierarchical_mpc_arm.py`, method `_get_subgoal_and_gripper()`:

```python
elif self.task_phase == "your_new_phase":
    subgoal = your_target_position
    gripper = your_gripper_state
    
    # Transition condition
    if your_condition:
        self.task_phase = "next_phase"
```

### Add New Features

1. **Define feature** in `arm_feature_utils.py`:
```python
def your_new_feature(obs_data):
    """Compute your feature from observation."""
    return feature_value  # Should be in [0, 1] range
```

2. **Add to features()** in `hierarchical_mpc_arm.py`:
```python
def features(self, obs):
    # ... existing features ...
    feat_your_feature = feature_utils.your_new_feature(obs["data"])
    
    return np.array([
        feat_red_dist,
        # ... other features ...
        feat_your_feature  # Add at end
    ])
```

3. **Update weight initialization**:
```python
base_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0])  # 7 features now
```

### Tune MPC Parameters

In `visualize_experiment.py`:

```python
arm = HierarchicalMPCArm(
    world=world,
    planner_horizon=8,      # Longer = more look-ahead, slower
    planner_n_iter=15,      # More iterations = better optimization, slower
    base_weights=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 1.0]),
    expert_weights=np.array([1.0, 1.0, 10.0, 2.0, 2.0, 5.0])
)
```

---

## Debugging

### Common Issues

**Problem**: Robot doesn't grasp block  
**Solution**: Check phase thresholds in `_get_subgoal_and_gripper()`:
```python
if dist_to_target < 0.02:  # Adjust this threshold
    self.task_phase = "grasp"
```

**Problem**: Block doesn't move toward zone B during transport  
**Solution**: 
1. Check `block_to_zone` weight is boosted (should be 10.0)
2. Verify MPC receives `target_pos=subgoal` in `_mpc_to_subgoal()`
3. Check that `zone_b_pos` is in observation

**Problem**: NaN gradients in MPC  
**Solution**: Usually happens if:
- Block dynamics not updating in MPC rollout
- Feature returns NaN for some states
- Control limits too aggressive

### Debug Output

Enable detailed logging by checking output during phases:

```
[t=640] Phase: transport, red_dist=0.150m, block_height=0.820m, 
        block_to_zone=0.721m, action_mag=0.193, gripper=CLOSED
```

Watch for:
- `block_to_zone` should **decrease** during transport
- `block_height` should stay constant (block is held)
- `red_dist` can vary (arm adjusts grip)

---

## Performance Tips

1. **Horizon vs. Iterations**: 
   - Horizon=8, Iterations=15 is a good balance
   - Longer horizon = smoother plans but slower
   - More iterations = better optimization but slower

2. **Phase Thresholds**:
   - Looser thresholds = faster transitions but less precise
   - Tighter thresholds = more precise but may get stuck

3. **Weight Boosting**:
   - Higher boost = stronger prioritization but less balanced
   - Current 10x boost works well for transport

---

## Comparison: Phase-Based vs. Hierarchical MPC

| Aspect | Pure Phase-Based (`intervention_arm.py`) | Hierarchical MPC (`hierarchical_mpc_arm.py`) |
|--------|------------------------------------------|---------------------------------------------|
| **Control** | Direct positional commands | Optimization-based (transport only) |
| **Learning** | Weights don't affect execution | Weights directly control transport behavior |
| **Speed** | Fixed speed per phase | Can learn to move faster via velocity weight |
| **Efficiency** | Fixed trajectory | Optimizes based on learned preferences |
| **Reliability** | Very stable | Stable (phases for critical actions) |
| **Complexity** | Simple state machine | Hybrid (phase + MPC) |

**Use Phase-Based** when: Reliability and simplicity are paramount  
**Use Hierarchical MPC** when: Learning from human feedback is important
