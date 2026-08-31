# Grid-World Multi-Goal Navigation with DQN

A Deep Q-Network agent that learns to navigate a 10x10 grid with obstacles, visiting multiple goals in sequence, using a local vision window and previous-action memory as state features.

## What this is

- A robot starts at a position on a 10x10 grid and must visit a sequence of goal cells in order, avoiding obstacles.
- The agent is trained with vanilla DQN (experience replay + a periodically-synced target network), not Double DQN or Dueling DQN.
- Obstacle maps are randomly generated per episode and verified solvable (BFS reachability check across the full goal sequence) before training on them.

**Status:** trained and evaluated. Results are reported below, but were measured under a limited test setup — see Known limitations before drawing conclusions from them.

## Files

| File | Purpose |
|---|---|
| `environment.py` | Grid-world environment: state representation, step/reward logic, map generation, solvability check |
| `dqn.py` | Training loop: builds the Q-network and target network, runs episodic training with epsilon-greedy exploration, saves the best model |
| `test_env.py` | Loads a saved model and evaluates it greedily (epsilon=0) over a batch of test episodes |
| `simulation.py` | Pygame GUI: lets you manually place the start, goals, and obstacles on the grid, then watches the trained agent navigate the map it produces step-by-step, with a live panel showing position, reward, and per-action Q-values |
| `best_q_network.keras` | Best checkpoint saved during training (by 100-episode rolling average reward) |

## Environment details

**State (29-dim vector):**
- Normalized robot position (x, y) — 2 values
- Normalized current goal position (x, y) — 2 values
- 5x5 local obstacle view centered on the robot (out-of-bounds counts as obstacle) — 25 values
- One-hot of the previous action — 4 values

**Actions:** `UP`, `DOWN`, `RIGHT`, `LEFT`

**Reward shaping:**
- `-10` for hitting a wall or an obstacle (episode continues)
- `+100` for reaching a goal; if more goals remain, the next goal becomes active and the episode continues
- `-1` per step, `+3` if the step reduced Manhattan distance to the current goal, `-2` if it increased it
- `-5` if the action directly reverses the previous action (discourages oscillation)

**Episode ends** when all goals in the sequence are reached, or `max_steps` is exceeded.

## Model

Two-layer MLP, 64 units per layer, ReLU activations, linear output over 4 actions. Identical architecture for the online network and the target network. Target network weights are hard-copied from the online network every 500 training steps (not episodes).

## Training

```bash
python dqn.py
```

You'll be prompted for the number of obstacles to place on the grid. Key hyperparameters (edit directly in `dqn.py`):

| Param | Value |
|---|---|
| Learning rate | 0.0003 |
| Discount factor (gamma) | 0.99 |
| Epsilon start / min | 1.00 / 0.05 |
| Epsilon decay | x0.999 per episode |
| Episodes | 2000 |
| Replay buffer size | 10,000 |
| Batch size | 64 |
| Target sync interval | every 500 training steps |
| Max steps per episode | 300 |

Start position, goal sequence (3 goals), and the obstacle map are all randomized at the start of every episode. The best model (by 100-episode rolling average reward) is saved to `best_q_network.keras`.

## Evaluation

```bash
python test_env.py
```

Prompts for a fixed start position, goal position, and obstacle count, then runs the saved model greedily (no exploration) over 1000 test episodes and reports the success rate.

Note: the training script optimizes for a *randomized, multi-goal, random-start* setting, while the test script evaluates a *fixed single-goal* setting the network wasn't specifically trained to be evaluated on in isolation — keep this mismatch in mind when interpreting test results.

## Simulation / GUI

```bash
python simulation.py
```

Opens a map editor: click to place the start cell, up to 3 goals, and obstacles, then press Enter to load the trained model and watch it attempt the map you built, with a live side panel (progress, position, distance to goal, and the Q-value for each action at every step). Press R to reset the episode or Esc to return to the map editor.

Note: the GUI builds its obstacle grid directly from your clicks and does not run the BFS solvability check that `environment.py` uses during training — you can hand it a map with no valid path and it won't warn you.

## Results

<!-- Add training curves, test success rate output, and simulation screenshots/GIFs here. -->

## Requirements

```
tensorflow
numpy
tqdm
pygame
```

## Known limitations

- Reward shaping constants are hand-picked, not tuned or ablated.
- Vanilla DQN only — no Double DQN, Dueling DQN, or prioritized replay.
- Train-time and test-time task distributions differ (multi-goal/random-start vs. single-goal/fixed-start), so a single fixed-pair test success rate doesn't cleanly measure general performance — vary start/goal pairs across a test run for a meaningful number.
- Training reward/success rate is still fluctuating at the end of the run rather than clearly converged; "best" checkpoint is selected by a noisy rolling average, not a stable plateau.
- The GUI can produce unsolvable maps with no warning, since it skips the solvability check used during training.

## Possible extensions

- Log and plot reward/success-rate curves during training
- Add Double DQN to reduce Q-value overestimation
- Curriculum learning: start with 0 obstacles, increase difficulty as success rate improves
- Replace the hard target-network sync with Polyak averaging
- Run the solvability check in the GUI before starting a simulation

## Relevance to real-world navigation

This is a discrete, fully-observable, static 10x10 grid — it is a conceptual testbed, not a deployable navigation stack. The gap to a real robot is large: no continuous state/action space, no sensor noise, no localization uncertainty, no dynamic obstacles, no physical motion constraints (acceleration, turning radius, collision margins), and no sim-to-real transfer step. None of that is solved here.

What does carry over conceptually, and where this pattern shows up in industry:

- **Warehouse/fulfillment robotics** (e.g. AMRs in Amazon-style warehouses) — multi-goal sequencing (visit shelf A, then B, then C) and obstacle-aware pathing are the same core problem, just in continuous space with many more agents and dynamic obstacles.
- **Automated guided vehicles (AGVs) in manufacturing** — fixed or semi-fixed layouts with waypoint sequences resemble the grid + ordered-goals setup here, though industrial AGVs mostly still rely on deterministic path planning (A*, Dijkstra) rather than learned policies, precisely because determinism and verifiability matter more than learned generalization.
- **Drone/vehicle route planning with ordered waypoints** — the ordered multi-goal reward structure maps to delivery routing with fixed stop sequences.
- **RL-for-robotics research generally** — grid-world DQN is a standard first step before moving to continuous control (e.g. via SAC/PPO in a physics simulator like MuJoCo or Isaac Sim) precisely because it's cheap to iterate on reward shaping and architecture before paying the cost of simulating real dynamics.

If the goal is an actual industrial pitch, the honest next steps are: move to a continuous or higher-resolution state space, add dynamic/moving obstacles, benchmark against a classical planner (A*) to show the learned policy earns its complexity, and address the solvability and generalization gaps listed above. Right now this demonstrates you can implement DQN correctly, not that DQN is the right tool for this class of problem.
