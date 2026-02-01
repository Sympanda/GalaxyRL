# SOM Binning RL Config Reference (`config.jsonc`)

This document describes every config field, **allowed values**, and **what it does**, for the SOM backend (`SOMControl`) and Gymnasium environment (`SOMBinningEnv`).

---

## Full `config.jsonc` template (annotated)

```jsonc
{
  // =========================
  // Backend: SOMControl
  // =========================
  "backend": {
    // Fixed for this project
    "grid_shape": [200, 200],       // [H, W]; N = H*W nodes
    "num_bins": 11,                 // K=11 bins (0..10)

    // Numerical stability
    "eps": 1e-12,                   // added to denominators to avoid div-by-0

    // Normalization constants to match your notebook convention
    "dr_scale": 10.0,               // DR normalization multiplier
    "rd_scale": 10.0,               // RD normalization multiplier
    "rr_scale": 100.0,              // RR multiplier (= dr_scale * rd_scale)

    // No-op detection tolerance
    "noop_eps": 0.0                 // is_noop = (best_delta <= noop_eps)
  },

  // =========================
  // Environment: SOMBinningEnv
  // =========================
  "env": {
    "grid_shape": [200, 200],       // must match backend
    "torus": true,                  // wrap edges (true) vs clip edges (false)

    // ---------- Action interface ----------
    "action_mode": "teleport_node",

    // ---------- Observation interface ----------
    "obs_mode": "local_patch",
    "patch_size": 9,                // odd; used by local_patch and cursor_local_k
    "allow_act_every_step": true,   // for cursor modes: act on move steps

    // Include U-matrix if backend provides it
    "include_umatrix": true,

    // How bins appear in observation:
    // - "int": (ps,ps) int32 in [0..10] (or (H,W) for global)
    // - "onehot": (11,ps,ps) float32 0/1 (only sensible for local_patch)
    "bins_encoding": "int",

    // ---------- Reward ----------
    "reward_type": "delta",         // "delta" or "raw"
    "clip_reward": null,            // null or float clip, e.g. 1.0
    "no_op_penalty": 0.0,           // added if backend reports is_noop
    "invalid_action_penalty": -1.0, // penalty if action invalid (torus=false cases etc.)

    // ---------- Episode control ----------
    "max_steps": 2000,
    "terminate_on_no_improve": false,
    "patience": 200,                // only used if terminate_on_no_improve=true

    // =========================
    // Bonus channels & masking
    // =========================

    // Visit counts channel (helps avoid re-visiting the same nodes)
    "visit_counts": {
      "enabled": true,

      // "ever": cumulative counts; never decay
      // "recent": exponential decay (EMA) so old visits fade
      "mode": "recent",

      // Only used if mode="recent"
      "ema_alpha": 0.05,

      // Where to include in obs:
      // - "local_patch": add (ps,ps) float32 channel
      // - "global": add (H,W) float32 channel (not recommended)
      "obs_inclusion": "local_patch",

      // Scaling before putting in obs
      // - "none": raw
      // - "log1p": log(1+count)  (recommended)
      // - "sqrt": sqrt(count)
      "scale": "log1p"
    },

    // Cooldown masking/penalty (discourage rapid re-selection)
    "cooldown": {
      "enabled": true,

      // number of env steps a node remains in cooldown after an ACT
      "steps": 25,

      // If true: selecting a cooldown node is invalid (penalty, no act)
      // If false: still allowed, but penalized
      "hard_mask": false,

      // penalty applied on ACT if node in cooldown (only if hard_mask=false)
      "penalty": -0.05,

      // when cooldown is applied:
      // - "act": only when backend step_node is executed
      // - "any_select": whenever node is selected (movement or act)
      "apply_on": "act"
    }
  }
}
```

---

## Backend (`backend`) field reference

### `backend.grid_shape`
- **Type:** `[int, int]`  
- **Allowed:** `[200, 200]` for your project  
- **Meaning:** SOM grid height/width. Backend assumes **row-major** flattening: `idx = y*W + x`.

### `backend.num_bins`
- **Type:** `int`  
- **Allowed:** `11` (locked)  
- **Meaning:** Number of redshift bins, labels are integers `0..10`.

### `backend.eps`
- **Type:** `float`  
- **Allowed:** any small positive value (typ. `1e-12`)  
- **Meaning:** Adds stability to denominators such as `n_ref * n_phot` and avoids division by zero.

### `backend.dr_scale`, `backend.rd_scale`, `backend.rr_scale`
- **Type:** `float`  
- **Allowed:** your notebook convention uses `10.0`, `10.0`, `100.0`  
- **Meaning:** Applies your scaling convention so the proxy matches your manual normalization:
  - `DRn = DR_sum / (dr_scale * denom)`
  - `RDn = RD_sum / (rd_scale * denom)`
  - `RRn = RR_sum / (rr_scale * denom)`

### `backend.noop_eps`
- **Type:** `float`  
- **Allowed:** `0.0` (strict) or a small positive tolerance (e.g. `1e-10`–`1e-8`)  
- **Meaning:** Controls when a greedy action is treated as “no-op”:
  - `is_noop = (best_delta <= noop_eps)`
- **When to increase:** if you see tiny positive deltas caused by floating noise and want those treated as no-ops.

---

## Env (`env`) field reference

### `env.grid_shape`
- **Type:** `[int, int]`  
- **Allowed:** must match backend; `[200, 200]`  
- **Meaning:** Used for observation/action space shapes and reshaping `current_bins`.

### `env.torus`
- **Type:** `bool`  
- **Allowed:** `true` or `false`  
- **Meaning:**
  - `true`: the SOM acts like a torus; edges wrap for cursor moves + patch extraction.
  - `false`: edges clip; some actions can become invalid (and incur `invalid_action_penalty`).

---

## Action interface

### `env.action_mode`
- **Type:** `string`  
- **Allowed values:**
  - `"teleport_node"`: `action = node_idx` in `[0..39999]`.
  - `"teleport_xy"`: `action = (x,y)` in `[0..199]×[0..199]`. Sets cursor and acts there.
  - `"factor_xy"`: same as teleport_xy; conceptual factorization (useful for some agents).
  - `"cursor_move"`: `action ∈ {0..3}` (up/down/left/right); acts each move only if `allow_act_every_step=true`.
  - `"cursor_move_act"`: `action ∈ {0..4}` where `0..3` are moves and `4` is ACT at cursor.
  - `"cursor_local_k"`: `action ∈ {0..ps*ps-1}` selects a cell in the local patch and acts there.
- **Practical notes:**
  - `"teleport_node"` is simplest and usually fastest to learn with at first.
  - Cursor modes matter when you want locality, exploration constraints, or patch-based actions.

### `env.allow_act_every_step`
- **Type:** `bool`  
- **Allowed:** `true`/`false`  
- **Meaning:** In cursor-based modes:
  - `true`: moving also triggers backend `step_node`.
  - `false`: only explicit ACT triggers backend (relevant for `"cursor_move_act"`).

---

## Observation interface

### `env.obs_mode`
- **Type:** `string`  
- **Allowed:**
  - `"global_bins"`: observation includes entire `(H,W)` bins grid (and optionally global U-matrix).
  - `"local_patch"`: observation includes `(ps,ps)` patch around cursor (and optionally patch U-matrix).
- **Practical notes:**
  - `"global_bins"` is heavy (memory + compute), and “global one-hot” is especially wasteful.
  - `"local_patch"` is the default recommended.

### `env.patch_size`
- **Type:** `int` (must be odd)  
- **Allowed:** `5,7,9,11,...`  
- **Meaning:** patch radius is `r = ps//2`. Controls local observation and `cursor_local_k` action size.

### `env.include_umatrix`
- **Type:** `bool`  
- **Allowed:** `true`/`false`  
- **Meaning:** Adds U-matrix channel (if the backend provides it). For local_patch, it’s the U-matrix patch.

### `env.bins_encoding`
- **Type:** `string`  
- **Allowed:**
  - `"int"`: bins are ints `[0..10]`
    - local_patch → `(ps,ps)` int32
    - global_bins → `(H,W)` int32
  - `"onehot"`: bins are one-hot encoded
    - local_patch → `(11,ps,ps)` float32
    - **Not recommended for global_bins** (would be `(11,200,200)` = 440k floats per obs)
- **Tradeoff:**
  - One-hot can help CNNs learn categorical structure.
  - Int grids are smaller/faster and often sufficient.

---

## Reward and penalties

### `env.reward_type`
- **Type:** `string`  
- **Allowed:** `"delta"` or `"raw"`  
- **Meaning:**
  - `"delta"`: reward = change in trace from one backend greedy step (`delta_reward`).
  - `"raw"`: reward = current trace value (less stationary; can be harder to learn).

### `env.clip_reward`
- **Type:** `float` or `null`  
- **Allowed:** `null` or a positive float (e.g. `1.0`)  
- **Meaning:** clips reward to `[-clip, +clip]`. Helps stability if rare large deltas exist.

### `env.no_op_penalty`
- **Type:** `float`  
- **Allowed:** any float (typically `0` or small negative like `-0.01`)  
- **Meaning:** added when backend reports `is_noop=True` (node had no improving bin change).

### `env.invalid_action_penalty`
- **Type:** `float`  
- **Allowed:** typically negative (e.g. `-1.0`)  
- **Meaning:** returned when action is invalid (primarily when torus=false or local selection goes out of bounds).

---

## Episode control

### `env.max_steps`
- **Type:** `int`  
- **Meaning:** hard truncation length.

### `env.terminate_on_no_improve`
- **Type:** `bool`  
- **Meaning:** if enabled, terminate early when improvement stalls.

### `env.patience`
- **Type:** `int`  
- **Meaning:** number of steps without improvement before early termination (only used if `terminate_on_no_improve=true`).

---

## Bonus channels & masking

### `env.visit_counts`
Adds a **visitation-frequency** feature as an extra observation channel (usually local_patch).
This helps the policy avoid repeatedly sampling the same nodes.

Fields:
- `enabled` (`bool`)
- `mode` (`"ever"` | `"recent"`)
  - `"ever"`: counts only increase across the episode.
  - `"recent"`: applies exponential decay each step (or each select) so counts fade.
- `ema_alpha` (`float`, only for `"recent"`)
  - higher alpha → faster forgetting.
- `obs_inclusion` (`"local_patch"` | `"global"`)
- `scale` (`"none"` | `"log1p"` | `"sqrt"`)
  - use `"log1p"` by default.

### `env.cooldown`
Implements a **cooldown window** after a node is acted on to reduce thrashing/ping-pong.

Fields:
- `enabled` (`bool`)
- `steps` (`int`): cooldown length in env steps
- `hard_mask` (`bool`)
  - `true`: acting on cooldown nodes is **invalid** (penalized, no backend call)
  - `false`: backend call allowed but penalized by `penalty`
- `penalty` (`float`, only if `hard_mask=false`): negative reward added when acting during cooldown
- `apply_on` (`"act"` | `"any_select"`)
  - `"act"`: only when backend step occurs
  - `"any_select"`: anytime node is selected (cursor moves / teleports too)

---

## Recommended baseline config snippet

```jsonc
"env": {
  "action_mode": "teleport_node",
  "obs_mode": "local_patch",
  "patch_size": 9,
  "bins_encoding": "int",
  "reward_type": "delta",
  "no_op_penalty": -0.01,
  "cooldown": { "enabled": true, "steps": 25, "hard_mask": false, "penalty": -0.05, "apply_on": "act" },
  "visit_counts": { "enabled": true, "mode": "recent", "ema_alpha": 0.05, "obs_inclusion": "local_patch", "scale": "log1p" }
}
```

---

## Note on where `is_noop` belongs

- **Backend should compute** `best_bin`, `best_delta`, and `is_noop` (intrinsic to the greedy TreeCorr proxy).
- **Env should decide** how to use it (penalty, termination logic, cooldown updates, masking).

This keeps “proxy physics” in the backend and “training protocol” in the env.
