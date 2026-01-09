# Robosuite Pick-and-Place Experiments

A robotic manipulation system that learns from human interventions. The robot performs pick-and-place tasks and improves by observing human corrections during execution.

## Overview

This system combines:
- **Phase-based control** for discrete actions (grasping, releasing)
- **MPC optimization** for continuous movement (transport phase)
- **Learning from demonstrations** to update behavior based on human feedback

The robot learns a reward function with 6 interpretable features (velocity, collision safety, goal proximity, etc.) and adjusts weights based on human interventions.

---

## Quick Start

### Prerequisites

```bash
cd /path/to/QuickLAP
source venv/bin/activate
```

### Run Single Experiment with Visualization

See the robot perform a pick-and-place task in real-time:

```bash
cd robosuite_experiments
python visualize_experiment.py
```

**What it does:**
- Opens a simulation window showing the robot arm
- Executes pick-and-place from Zone A to Zone B
- Shows debug output (phases, distances, rewards)
- Demonstrates base behavior without learning

**On macOS**: Use `mjpython visualize_experiment.py` for better performance.

### Run Learning Experiments

Run multiple experiments with different learning methods:

```bash
python run_experiments.py -n 5
```

**What it does:**
- Tests multiple learning algorithms (naive PHRI, LLM-based, oracle)
- Runs each configuration N times with different seeds
- Saves results to `logs/` directory with timestamped folders
- Generates comparison plots and metrics

**Options:**
- `-n <number>`: Number of runs per configuration (default: 1)
- `--notes "<text>"`: Add notes to experiment log

### Run Custom Experiment

```bash
python pick_place_experiment.py
```

Programmatic interface for running experiments with custom configurations.

---

## Scripts Overview

### Main Execution Scripts

| Script | Purpose | Use When |
|--------|---------|----------|
| `visualize_experiment.py` | Single run with GUI | Debugging, demonstrations |
| `run_experiments.py` | Batch experiments | Comparing learning methods |
| `pick_place_experiment.py` | Custom experiments | Programmatic control needed |

### Core Implementation

| File | What It Does |
|------|--------------|
| `hierarchical_mpc_arm.py` | Main robot controller (hybrid phase + MPC) |
| `intervention_arm.py` | Simpler phase-based controller (baseline) |
| `mpc_arm_planner.py` | MPC optimization for transport |
| `arm_feature_utils.py` | Feature computation (centralized) |
| `robosuite_phri_learner.py` | Learning algorithm from interventions |

### Environment & Support

| File | What It Does |
|------|--------------|
| `arm_world.py` | Environment wrapper for robosuite |
| `two_block_env.py` | Custom environment (red block, green obstacle) |
| `base_rational_arm.py` | Base class with reward computation |
| `arm_planner.py` | Base planner interface |

---

## Task Description

**Goal**: Pick up the red block from Zone A and place it in Zone B.

**Obstacles**: Green block in the path, Zone C area to avoid.

**Phases**:
1. **Approach** - Move above red block
2. **Descend** - Lower to grasp height
3. **Grasp** - Close gripper and wait
4. **Lift** - Lift block up
5. **Transport** - MPC optimizes path to Zone B (learning affects this!)
6. **Place & Descend** - Lower to zone
7. **Release** - Open gripper
8. **Retract** - Move away

Only the **transport** phase uses MPC optimization where learned weights matter. Other phases use direct control for reliability.

---

## Features & Rewards

The robot optimizes a weighted sum of 6 features:

| Feature | Description | Typical Weight |
|---------|-------------|----------------|
| `distance_to_green_block` | Proximity to obstacle | 1.0 |
| `velocity` | Movement speed | 1.0 (learnable) |
| `collision_safety` | Avoidance margin | 2.0 |
| `joint_safety` | Joint limit safety | 2.0 |
| `block_to_target_zone` | Goal proximity | 10.0 |
| `zone_c_proximity` | Obstacle zone avoidance | -2.0 |

**Learning**: When a human intervenes (e.g., "go faster"), the system compares human vs. robot behavior and updates weights to match human preferences.

---

## Results & Logs

Results are saved to timestamped directories in `logs/`:

```
logs/YYYY_MM_DD_HH_MM_SS/
├── results.json              # Raw experiment data
├── report.txt                # Human-readable summary
└── visualizations/
    ├── reward_comparison.png
    ├── weights_comparison.png
    ├── weights_mse_comparison.png
    └── regret_comparison.png
```

---

## Configuration

### Modify Base Behavior

Edit `visualize_experiment.py`:

```python
base_weights = np.array([1.0, 1.0, 2.0, 2.0, 10.0, -2.0])
# [green_dist, velocity, collision, joints, block_to_zone, zone_c_prox]
```

### Change MPC Settings

```python
arm = HierarchicalMPCArm(
    planner_horizon=8,    # Planning lookahead (default: 8)
    planner_n_iter=20,    # Optimization iterations (default: 15-20)
)
```

Higher values = better optimization but slower execution.

### Add Features

1. Define feature function in `arm_feature_utils.py`
2. Add to `compute_features()` function
3. Add description to `get_feature_descriptions()`
4. Update weight arrays to match new feature count
