# Autonomous Assembly of Neural Structure (AANS)

A computational framework for self-organizing recurrent neural networks. The system adapts its internal connectivity in response to environmental demands, based on principles of variational free energy minimization. The core implementation uses Triton for high-performance, block-sparse recurrent operations.

This document provides instructions for running the project and a map from the conceptual documentation (`index.html`) to the codebase. For the theoretical underpinnings, refer to `index.html`.

## I. Execution

### Prerequisites

*   Python 3.10+
*   `uv` package manager (`pip install uv`)
*   NVIDIA GPU with CUDA and Triton support

### Installation

Install the project in editable mode with its dependencies:

```bash
uv pip install -e .
```

### Running Core Tests

The integration tests serve as primary validation and usage examples.

**1. Perturbation & Recovery Test:**
Validates the model's resilience, concept drift adaptation, and recovery from structural damage.

```bash
uv run pytest -s tests/integration/test_perturbation.py
```

**2. Supervised Learning & Generative Rollout Test:**
Validates the model's ability to learn a time-series and generate autonomous predictions.

```bash
uv run pytest -s tests/integration/test_sl.py
```

**3. Reinforcement Learning Test:**
Validates the actor-critic implementation in a standard Gym environment.

```bash
uv run pytest -s tests/integration/test_rl.py
```

### Running Examples

The `examples/` folder contains standalone scripts demonstrating AANS on various tasks.

**1. Reinforcement Learning (LunarLander):**
Train an AANS-based agent on the LunarLander environment with discrete or continuous action spaces.

```bash
uv run python -m examples.rl_lunarlander --num-blocks 32 --agent-type discrete --episodes 1000
```

Options:
- `--agent-type {discrete,continuous}`: Type of action space (required)
- `--episodes EPISODES`: Number of episodes to run
- `--visualize-episode`: Render the environment during training
- `--visualize-aans`: Enable live visualization of the AANS adjacency matrix
- `--visualize-traces`: Enable live visualization of internal traces
- `--scroll-window-size SCROLL_WINDOW_SIZE`: Number of steps in scrolling trace visualization
- `--num-blocks NUM_BLOCKS`: Number of AANS blocks (default: 32, giving 1024 neurons)
- `--value-lr VALUE_LR`: Learning rate for value head (default: 0.01)
- `--policy-lr POLICY_LR`: Learning rate for policy head (default: 1.0)
- `--gamma GAMMA`: Discount factor for TD learning (default: 0.99)
- `--no-tb`: Disable TensorBoard logging

**2. Supervised Learning (Time-Series Benchmarks):**
Train AANS on Mackey-Glass, Lorenz, or NARMA time-series prediction tasks.

```bash
uv run python -m examples.sl_narma10 --task narma --steps 10000
```

Options:
- `--task {mackey,lorenz,narma,all}`: Task to run (required)
- `--steps STEPS`: Number of training steps
- `--lr LR`: Learning rate
- `--num-blocks NUM_BLOCKS`: Number of AANS blocks
- `--neurons-per-block NEURONS_PER_BLOCK`: Neurons per block
- `--seed SEED`: Random seed
- `--scheduled-sampling-p SCHEDULED_SAMPLING_P`: Initial probability for teacher forcing
- `--sampling-decay-schedule {linear,exponential}`: Decay schedule for sampling
- `--sampling-jitter SAMPLING_JITTER`: Uniform noise added to sampling probability
- `--multistep-eval`: Enable multi-step-ahead evaluation for NARMA task

## II. Codebase Architecture

The project maps high-level theoretical concepts to specific, modular components. `index.html` explains the theory; this section maps that theory to the code.

| Concept (`index.html`)                     | Primary Code Module(s)                                                                      | Purpose                                                                                                         |
| ------------------------------------------ | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **High-Level Objective**                   | `src/triton/strat/` (`sl.py`, `rl.py`)                                                      | Defines the learning paradigm (Supervised, RL) and the primary error/reward signals driving the system.         |
| **State Dynamics (Inference)**             | `src/triton/machine.py` (`CoreMachine`)<br>`src/triton/bsr.py` (`BlockSparseRecurrentCore`) | Executes the recurrent state update (`forward` pass). The `bsr` module contains the low-level Triton kernel.    |
| **Continuous Plasticity (Learning)**       | `src/triton/weights.py` (`WeightUpdater`)                                                   | Implements all gradient-based synaptic weight update rules derived from the potential function `V`.             |
| **Discrete Plasticity (Phase Transition)** | `src/triton/topology.py` (`TopologyModifier`)                                               | Implements structural changes (growth/pruning of connections) based on threshold-crossing dynamics.             |
| **Integration & Orchestration**            | `src/triton/base.py` (`BaseModel`)<br>`src/triton/plasticity.py` (`PlasticityManager`)      | `BaseModel` integrates all components. `PlasticityManager` orchestrates `WeightUpdater` and `TopologyModifier`. |
| **System Configuration**                   | `src/triton/config.py`                                                                      | Centralized dataclass for all hyperparameters. This is the main entry point for configuring experiments.        |
| **Core State Representation**              | `src/triton/types.py` (`StateTuple`)                                                        | Defines the data structure for the network's transient state, ensuring a stateless `forward` method.            |
```
