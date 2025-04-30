# Occlusion: Mouse Navigation with Noisy Information Accumulation - A Reinforcement Learning Approach

## Instruction and files

1. Run the `generate information background.ipynb` notebook to generate the information maps.  
   These maps will be saved as `new info matrix hres.npy` files.

2. Run the `aperture rl ab.ipynb` notebook to train the agent.



## Problem Description

This problem models a mouse navigating through a 2D arena (`nx` columns, `ny` rows) to find the correct reward port based on **noisy, spatially-dependent information**. The mouse starts at position (`nx`//2, 0) (bottom center) and must navigate to one of two potential reward ports located at opposite sides of the top row: (`0`, `ny`-1) (left port) or (`nx`-1, `ny`-1) (right port). At the start of each episode, one port is randomly designated as correct (50% chance each), but the agent doesn't know which.

The central challenge is for the agent to **learn a policy** that efficiently gathers information to infer the correct port while minimizing movement costs and navigating towards the chosen port. Information gain is determined by the mouse's position (`x`, `y`) according to a predefined `info_matrix`, but this information is corrupted by **observation noise**, making the inference process inherently uncertain.

## Reinforcement Learning Environment (`MouseNavigationEnv`)

The problem is formulated as a Reinforcement Learning task using a custom `gymnasium` (or `gym`) environment.

### Observation Space (Agent's Input)

The agent receives observations from the environment at each timestep. The observation is a 5-dimensional vector:

1.  `x_norm`: Current x-position, normalized to `[0, 1]`.
2.  `y_norm`: Current y-position, normalized to `[0, 1]`.
3.  `vx_norm`: Current x-velocity, normalized to `[-1, 1]` based on `max_v`.
4.  `vy_norm`: Current y-velocity, normalized to `[-1, 1]` based on `max_v`.
5.  `belief_state`: A value in `[-1, 1]` representing the agent's current belief about the correct port, derived from the internally tracked `cumulative_info`. Calculated as `(success_prob(cumulative_info) * 2 - 1)` if the left port is correct, and `-(success_prob(-cumulative_info) * 2 - 1)` if the right port is correct. This effectively encodes both the strength and direction of the belief.

*(Note: While the environment internally tracks the raw `cumulative_info`, the agent only observes this processed `belief_state` value.)*

### Action Space

The agent chooses an action at each step, represented by discrete accelerations in the x and y directions.
* Action: `[ax_choice, ay_choice]` where each choice is an integer from 0 to 6.
* Mapping: This maps to accelerations `ax, ay` in the range `[-3, -2, -1, 0, 1, 2, 3]` for each axis.

### Constraints

* **Maximum Velocity:** The magnitude of the mouse's velocity vector (`vx`, `vy`) is capped at `max_v`. If acceleration would exceed this, the resulting velocity vector is scaled down.
* **Arena Boundaries:** The mouse cannot move outside the arena (0 ≤ `x` < `nx`, 0 ≤ `y` < `ny`). Position is clipped if a move would go out of bounds.
* **Port Approach:** The mouse can only approach the reward ports (`y = ny-1`) horizontally. If the mouse is about to reach the final row, its vertical velocity (`vy`) is forced to 0 for that step.
* **Maximum Steps:** Each episode terminates if the agent exceeds `max_steps`.

### Movement Costs, Information Gain, and Rewards

The agent learns by receiving rewards based on its actions and outcomes.

* **Step Costs:** Each movement incurs costs subtracted from the reward:
    * **Velocity Cost:** `sqrt(vx^2 + vy^2) * vcost` (penalizes high speeds).
    * **Acceleration Cost:** `sqrt(ax^2 + ay^2) * wcost` (penalizes rapid changes in velocity/high acceleration).
    * **Time Cost:** `timecost` (a constant penalty per step, encouraging efficiency).
* **Information Gain:**
    * At each step, potential information is available based on the agent's *previous* position (`old_x`, `old_y`) in the `info_matrix`.
    * This raw information is scaled by `info_scale`.
    * **Noise:** The actual information gain observed is stochastic due to `noise`. The probability of observing *no* information increases with the `noise` level and decreases with the amount of raw information available at that location (`noisy = max(0, (raw_info/max_info - U(0, noise)))`).
    * The observed `info_gain` is added to the internal `cumulative_info`. The sign is positive if the left port is correct and negative if the right port is correct.
* **Terminal Rewards/Penalties:**
    * **Reaching a Port:**
        * The `success_probability` is calculated based on the absolute value of the final `cumulative_info`: `success_prob = 0.5 * erfc(-sqrt(abs(cumulative_info) / 2))`.
        * If the **correct** port is chosen: `reward += success_prob * reward_magnitude`.
        * If the **incorrect** port is chosen: `reward -= success_prob * reward_magnitude * 5` (a significant penalty).
    * **Stopping:** If the agent stops (near-zero velocity) before reaching a port: `reward -= 0.5 * reward_magnitude`.
    * **Timeout:** If the agent reaches `max_steps`: `reward -= 0.25 * reward_magnitude`.

### Key Parameters

* `noise`: Controls the level of stochasticity in information gathering.
* `info_matrix`: The 2D array defining the base information value at each location.
* `info_scale`: A multiplier (0-1) applied to the raw information from `info_matrix`.
* `vcost`, `wcost`, `timecost`: Coefficients for the step costs.
* `max_v`: Maximum speed limit.
* `max_steps`: Maximum number of steps per episode.
* `reward_magnitude`: Scales the terminal rewards and penalties.

## Reinforcement Learning Goal

The objective is to train an RL agent (e.g., using Proximal Policy Optimization - `PPO`) to learn a policy (`π(action | observation)`) that maximizes the expected cumulative discounted reward. The agent must learn to navigate the environment, decide when and where to gather information (balancing exploration cost vs. information value), interpret the noisy information to form a belief, and commit to a port based on that belief, all while minimizing movement costs. The agent's performance is evaluated by its success rate in choosing the correct port and the total reward accumulated.