# QuickLAP: Quick Language-Action Preference Learning

This repository contains the complete implementation of QuickLAP and all experimental code used to generate the results in our paper.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# 2. Set your OpenAI API key (either method works)
export OPENAI_API_KEY="your_key_here"
# OR create a .env file with: OPENAI_API_KEY=your_key_here
# 3. Run main experiments
python run_experiments.py -n 5
```

## Repository Structure

```
├── interact_drive/                         # Core simulation framework
│   ├── learner/                            # All learning algorithms
│   │   ├── phri_learner.py                 # Physical-only baseline
│   │   ├── masked_learner.py               # Masked attention baseline
│   │   └── adapt_gated_llm_learner.py      # QuickLAP (main method)
│   ├── planner/                            # Planners
│   ├── car/                                # Car agents
│   ├── feature_utils.py                    # Features functions
│   ├── world.py                            # Driving environments
├── experiments/                            # Scenario configurations
│   ├── cone_avoid.py                       # Simple cone scenario (C)
│   ├── puddle_avoid.py                     # Cone + puddle (CP)
│   ├── cone_car_avoid.py                   # 3-lane scenario (CPC-3)
│   ├── cone_car_avoid_four.py              # 4-lane scenario (CPC-4)
│   ├── experiment.py                       # Base experiment class
│   ├── intervention_car.py                 # Base intervention car class
│   └── run_experiments.py                  # Main experimental script
```

## Key Files

### Core Algorithm Implementation

- **`adapt_gated_llm_learner.py`**: Complete QuickLAP implementation with dual-LLM system
- **`adapt_gated_llm_selector.py`**: LLM prompt templates and JSON parsing

### Experimental Setup

- **`run_experiments.py`**: Reproduces all paper results (Figure 3a, 3b)
- **Lines 20-25**: Ground truth reward weights for each scenario
- **Lines 52-58**: Natural language utterances (currently testing with "Be careful.")
- **Lines 59-71**: Learner factory definitions (masked_dphi vs adapt_gated_llm)

### Baseline Implementations

- **`phri_learner.py`**: Physical correction only (Eq. 2 from paper)
- **`masked_learner.py`**: Attention-gated physical updates

## Reproducing Paper Results

### Main Results

```bash
python run_experiments.py -n 5
```

This runs 5 seeds × 6 utterance × 4 scenarios = 120 experiments per method.

### Individual Scenario Testing

```bash
# Test individual scenarios with visualization
python experiments/puddle_avoid.py
python experiments/cone_car_avoid.py
python experiments/cone_car_avoid_four.py
```

### Quick Verification

For quick verification, comment out all but one combination in `run_experiments.py` and then run:
```bash
# Single run for testing
python run_experiments.py -n 1
```

## Intervention Configuration

Each scenario can be configured with different numbers of interventions by modifying the `NUM_INTERVENTION` variable in `experiments/intervention_car.py`.

**Current Configuration**: All scenarios use 4 interventions at approximately timesteps (45-55), (85-95), (130-140), (170-180).

## Expected Results

Results are saved to `logs/YYYY_MM_DD_HH_MM_SS/` with detailed JSON output.

## Hyperparameters

Key parameters are defined in the learner files:
- **Learning rate (α)**: 1.0
- **Language confidence scale (σ)**: 1.2
- **Capping factor**: 5.0 × feature difference
- **LLM temperature**: 0.1 (attention), 0.3 (preference)


## Citation

If you find this repository useful, please cite:

```bibtex
@misc{nader2025quicklapquicklanguageactionpreference,
      title={QuickLAP: Quick Language-Action Preference Learning for Autonomous Driving Agents}, 
      author={Jordan Abi Nader and David Lee and Nathaniel Dennler and Andreea Bobu},
      year={2025},
      eprint={2511.17855},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.17855}, 
}
```

## Acknowledgments

This work builds upon model_switching repository: https://github.com/arjunsripathy/model_switching
