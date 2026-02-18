## SOM-RL: Problem & Setup Schematic

### 1. Problem definition

- **Domain**: Self-Organizing Map (SOM) over galaxies for redshift binning / cross-correlation calibration.
- **Grid**: `200 × 200` SOM nodes, optionally with multi-resolution clusters.
- **State**:
  - Each node is assigned to one of `K` photometric bins (currently `K = 11`).
  - Backend maintains fast correlation statistics (`dd, dr, rd, rr`) and derived clustering metrics.
- **Objective**:
  - Define an energy / objective \( E \) based on the xi matrix:
    - Backend computes \( \xi \) and then `energy_E()` as a scalar that increases when:
      - correlation mass is concentrated on the diagonal (good phot-z alignment),
      - off-diagonal leakage is reduced (less cross-contamination between bins).
  - **RL goal**: learn a policy that proposes node/bin changes that **increase `energy_E()`** under a simulated annealing-like acceptance rule.

### 2. Backend: `SharpZSOMBackend`

- **Responsibilities**:
  - Store SOM state:
    - `current_bins` over `N = H*W` nodes.
    - `node_gal_counts` (weighted counts per node).
  - Maintain correlation caches:
    - `DD_sum, DR_sum, RD_sum, RR_sum` over `(ref_bin, phot_bin)`.
  - Compute observables:
    - `xi_matrix()`
    - `trace_xi()`
    - `energy_E()` (used as the main scalar objective for RL).
  - Proposals:
    - `make_proposal_random_bin(node|xy, level, rng)`:
      - node/cluster resolution; proposes a new bin according to `proposal_bin_mode` (e.g. `"uniform_other"`, `"neighbor_pm1"`).
    - `make_proposal_set_bin(new_bin, node|xy, level)`:
      - pure set-bin proposal, with masks / constraints checked.
  - Apply / revert:
    - `apply_proposal(proposal) -> undo`
    - `revert(undo)`
  - Evaluation:
    - `evaluate_proposal_deltaE(proposal)`:
      - computes `E0`, applies proposal, computes `E1`, returns `dE = E1 - E0` and `undo`.
    - Leaves proposal **applied**; environment must call `revert` if it rejects.
  - Greedy helper:
    - `greedy_bin_for_node(node, level) -> (best_bin, best_dE)`:
      - scans all bins for a given node, evaluating `dE` for each, and returns the bin with maximal `dE`.

### 3. Scheduler: `som_rl/env/schedules.py`

- **Config**: `SchedulerConfig` controls:
  - `max_steps`: scheduler time horizon.
  - `level_min`, `level_max`, optional `level_list`: allowed levels.
  - Temperature schedule: `T_max`, `T_min`, `temp_mode`, `temp_evolve_with`.
  - Energy floor schedule: `min_abs_deltaE`, optional `max_abs_deltaE`, evolution mode (`steps`, `temp`, `level`).
  - Level control:
    - `level_fixed`, `freeze_level`, `reset_temp_on_level_up`.
  - Time advancement rule:
    - `advance_time_on`: `"step"`, `"accept"`, `"accepted_non_noop"`.
  - Gating:
    - `min_accepts_per_level` (hits-based gating).
    - Plateau gating parameters.

- **Base API**: `BaseScheduler`
  - `reset()`: initialize `step_idx`, `level`, internal flags.
  - `current_params() -> (level, T, floor)`: values for current env step.
  - `step_params(step_idx) -> (level, T, floor)`: same as above, for a given step index.
  - `update(accepted, dE, E, step_idx, is_noop)`: called once per env step after accept/reject.
  - `done() -> bool`: whether the scheduler/game is complete (used for env termination).

- **Concrete schedulers**:
  - `FixedScheduler`:
    - Monotonic schedule for `level` from `level_min` to `level_max` across steps.
    - `T` and `floor` from base interpolation.
  - `HitGatedScheduler`:
    - Hold `level` until `min_accepts_per_level` accepted (non-noop) moves are observed; then advance.
  - `PlateauGatedScheduler`:
    - Use a rolling window over accepted moves to detect plateau in `dE` and/or low accept rate; then advance level.
  - `LevelHitsScheduler`:
    - Uses `level_list` (e.g. `(30, 50, 80)`).
    - At each level, accumulate `min_accepts_per_level` accepted non-noop moves before advancing to the next level.
    - Once the final level also satisfies its quota, `done() == True`.
    - Scheduler `max_steps` is also honoured; if internal `step_idx` reaches `cfg.max_steps`, `done()` is set `True`.

### 4. Core environment: `SOMEnergyBaseEnv` (`env.py`)

- **Backbone responsibilities**:
  - Maintain cursor (`cursor_x, cursor_y`) and visitation stats:
    - `visit_count` per node.
    - `last_visit_step` per node.
  - Build observations (global or patch):
    - `bins` (int/float grid).
    - `empty` mask.
    - Optional `umatrix` grid.
    - Optional cluster features: `cluster_boundary`, `cluster_size`.
    - Optional `visit_count` and `recent` masks.
    - Scalar features: `[level_norm, T, E, steps_left_norm]`.
    - `cursor` coordinates.
  - Action decoding (delegated to subclasses).
  - Node targeting:
    - `_target_node_from_decoded(decoded)` → node index.
  - Scheduler integration:
    - `_sync_scheduler_params()` reads `(level, T, floor)` from scheduler.
    - Calls `scheduler.update(...)` after each step.
  - Accept/reject logic:
    - Build a proposal via backend:
      - random-bin or explicit set-bin depending on `new_bin`.
    - Evaluate via `evaluate_proposal_deltaE`.
    - Apply:
      - Energy floor gate: reject proposals with `dE < floor`.
      - Accept rule:
        - `accept_rule == "positive_only"`: accept if `dE > 0`.
        - `accept_rule == "sa"`: simulated annealing with probability `exp(dE / T)` for `dE <= 0`.
    - If rejected: call `revert(undo)` and record `is_noop` / `accepted=False`.
  - Reward computation (centralised `_compute_reward(out)`):
    - Two modes:
      - `"deltaE"`: based on `dE = E1 - E0` with optional `tanh` shaping:
        - `use_tanh_reward`: if True, use `tanh(dE / reward_tanh_temperature)`.
        - `reward_energy_scale`: overall multiplier.
      - `"rawE"`: use `E1` directly.
    - Penalties:
      - `reject_penalty`: applied when a non-noop proposal is rejected.
      - `noop_penalty`: applied when step is a true noop (e.g. invalid/empty nodes).
    - Generic shaping:
      - `step_penalty`: added every step.
      - `clip_reward`: optional clipping.
  - Termination / truncation:
    - `truncated = (step_idx >= cfg.max_steps)` from env side.
    - `terminated` set to `True` if `scheduler.done()` is `True`.

### 5. Archetype environments: `env_multi.py`

All archetypes inherit from `MultiBaseEnv`, which itself subclasses `SOMEnergyBaseEnv`, and thus share:

- Backend/scheduler contract.
- Accept/reject logic.
- Core reward shaping via `_compute_reward`.

#### 5.1 Bandit archetype: `EnvBandit`

- **Intent**: pure (or near-pure) multi-armed bandit over nodes, with no explicit spatial context.
- **Action space**:
  - `Discrete(H * W)` → node index.
- **Observation**:
  - `scalars`: `(4,)` → `[level_norm, temperature, E, steps_left_norm]`.
  - `cursor`: `(2,)` → `[cursor_x, cursor_y]`.
  - No SOM grids (no `bins`, no `umatrix`, etc.).
- **Dynamics**:
  - Action selects a node; bin selection uses:
    - random-bin via backend by default, or
    - greedy bin via `use_greedy_bin_for_node` + `greedy_bin_for_node`.
  - Accept/reject and reward exactly as in `SOMEnergyBaseEnv`.
- **Use case**:
  - Geometry-agnostic baseline: agent learns which node indices are profitable by trial and error without seeing local structure.

#### 5.2 Global-grid archetype: `EnvMinesweeper`

- **Intent**: "Minesweeper"-like full-grid view; agent sees entire SOM and picks one node per step.
- **Action space**:
  - `Discrete(H * W)` → node index.
- **Observation**:
  - `obs_mode="global"` enforced via config.
  - Grids:
    - `bins` (full `H×W`).
    - `empty` mask.
    - Optional `umatrix`.
    - Optional cluster features: `cluster_boundary`, `cluster_size`.
    - Optional `visit_count`, `recent`.
  - `scalars`: `(4,)`.
  - `cursor`: `(2,)`.
- **Dynamics**:
  - Each action chooses one node; bin selection and accept/reject as in base env.
  - Suitable for CNN-based policies that output a logit per node (global heatmap).

#### 5.3 Local patch + movement archetype: `EnvZelda`

- **Intent**: "Zelda"-like local exploration:
  - Agent navigates the SOM with local vision and a single "interact" button.
- **Action space**:
  - `Discrete(5)`:
    - `0`: move up.
    - `1`: move down.
    - `2`: move left.
    - `3`: move right.
    - `4`: interact (propose a bin change at current cursor node).
- **Observation**:
  - Patch mode enforced: `obs_mode="patch"`.
  - Local patch around cursor for:
    - `bins`.
    - `empty`.
    - Optional `umatrix`.
    - Optional `cluster_boundary`, `cluster_size`.
    - Optional `visit_count`, `recent`.
  - `scalars`: `(4,)`.
  - `cursor`: `(2,)`.
- **Dynamics**:
  - Movement actions:
    - Update `cursor_x, cursor_y` (torus or clipped).
    - Mark the new node as visited.
    - Call scheduler `update` with `accepted=False`, `is_noop=True`.
    - Use a synthetic `out` dict with `dE=0.0` for reward computation (only step_penalty or exploration bonuses would apply).
  - Interact action:
    - Convert `(cursor_x, cursor_y)` to node index.
    - Choose bin:
      - random-bin via backend or greedy bin if `use_greedy_bin_for_node=True`.
    - Call `_attempt_move(...)`, scheduler `update`, and `_compute_reward` as in base env.
    - Visit counts updated for all touched nodes (node or cluster).
- **Use case**:
  - Partial observability + navigation + interaction, similar to a console game agent.
  - Natural place to add intrinsic exploration rewards using `visit_count` / `last_visit_step`.

### 6. Policy archetypes: `som_rl/models.py`

All policies follow a simple interface:

- Base class: `SOMPolicyBase(nn.Module)`:
  - `forward(obs) -> (logits, value)`:
    - `logits`: `(B, action_dim)` over the discrete action space.
    - `value`: `(B, 1)` scalar value predictions.

#### 6.1 Bandit policy: `BanditMLPPolicy`

- **Target env**: `EnvBandit`.
- **Inputs**:
  - `obs["scalars"]`: `(B, 4)`.
  - `obs["cursor"]`: `(B, 2)`.
  - Concatenated to size 6.
- **Network**:
  - MLP:
    - `6 → 64 → 64` with ReLU.
  - Heads:
    - Policy: `64 → action_dim` (`H*W`).
    - Value: `64 → 1`.
- **Use case**:
  - Geometry-free bandit baseline; suitable for standard PPO / A2C with an MLP policy.

#### 6.2 Minesweeper policy: `MinesweeperCNNPolicy`

- **Target env**: `EnvMinesweeper`.
- **Inputs**:
  - `obs["grid"]`: `(B, C, H, W)` stacked spatial channels:
    - e.g. `bins`, `empty`, optional `umatrix`, cluster features, `visit_count`, `recent`.
  - `obs["scalars"]`: `(B, 4)`.
  - `obs["cursor"]`: `(B, 2)`.
- **Network**:
  - CNN over `grid` (Atari/Pokémon style):
    - Conv1: `C → 32`, `8×8`, stride 4, ReLU.
    - Conv2: `32 → 64`, `4×4`, stride 2, ReLU.
    - Conv3: `64 → 64`, `3×3`, stride 1, ReLU.
    - Flatten → `z_conv`.
  - Scalar encoder:
    - Concatenate `scalars` and `cursor` → `(6,)`.
    - MLP: `6 → 64 → 64` with ReLU → `z_scalar`.
  - Fusion:
    - Concat `z_conv` and `z_scalar`, pass through `fc` (e.g. to 256 dims).
  - Heads:
    - Policy: `hidden → action_dim` (`H*W` logit per node).
    - Value: `hidden → 1`.
- **Use case**:
  - Whole-grid policies that generate a spatial heatmap over nodes.

#### 6.3 Zelda policy: `ZeldaCNNPolicy`

- **Target env**: `EnvZelda`.
- **Inputs**:
  - `obs["patch"]`: `(B, C, ps, ps)` stacked local patch channels.
  - `obs["scalars"]`: `(B, 4)`.
  - `obs["cursor"]`: `(B, 2)`.
- **Network**:
  - CNN over patch:
    - Conv1: `C → 32`, `3×3`, stride 1, padding 1, ReLU.
    - Conv2: `32 → 64`, `3×3`, stride 1, padding 1, ReLU.
    - `adaptive_avg_pool2d` to `(3, 3)` followed by flatten → `z_conv`.
  - Scalar encoder:
    - Concatenate `scalars` and `cursor` → `(6,)`.
    - MLP: `6 → 64 → 64` with ReLU → `z_scalar`.
  - Fusion:
    - Concat `z_conv` and `z_scalar`, `fc` to `hidden_dim`.
  - Heads:
    - Policy: `hidden → 5` (up, down, left, right, interact).
    - Value: `hidden → 1`.
- **Extensions**:
  - For longer-horizon navigation, wrap this base CNN+MLP in an RNN/LSTM externally.
  - Natural candidate for intrinsic exploration bonus based on `visit_count` / `last_visit_step`.

---

This schematic is intended as a compact, implementation-level reference:

- Backends and schedulers define the **physics and curriculum** of the SOM energy optimisation problem.
- The three env archetypes define three **“games”** on top of that:
  - Bandit (no geometry).
  - Minesweeper (full grid).
  - Zelda (local patch + movement).
- The three policy archetypes (`BanditMLPPolicy`, `MinesweeperCNNPolicy`, `ZeldaCNNPolicy`) are tailored to the corresponding observation and action structures, while sharing the same underlying energy objective and reward shaping.

