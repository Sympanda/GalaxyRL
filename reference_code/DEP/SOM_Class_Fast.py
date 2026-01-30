# SOM_Class_FastDD_true_opt.py
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

# ---------------- helpers ----------------
def _as_2d_bins_nodes(x, name, num_bins_expected=None, num_nodes_expected=None, dtype=np.float32):
    A = np.asarray(x).squeeze()
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeeze, got {A.shape}")
    if num_bins_expected is not None and A.shape[0] != num_bins_expected:
        raise ValueError(f"{name} shape[0]={A.shape[0]} != num_bins ({num_bins_expected})")
    if num_nodes_expected is not None and A.shape[1] != num_nodes_expected:
        raise ValueError(f"{name} shape[1]={A.shape[1]} != num_nodes ({num_nodes_expected})")
    return A.astype(dtype, copy=False)

def _coerce_dd(dd, num_bins, num_nodes):
    """
    Coerce dd to (num_bins, num_nodes) by squeezing/reducing/transposing as needed.
    If dd has an extra trailing axis, we sum over it (safe for pair-count style data).
    """
    A = np.asarray(dd)
    if A.ndim > 2:
        if A.shape[-1] == 1:
            A = np.squeeze(A, axis=-1)
        else:
            A = A.sum(axis=-1)
    A = np.squeeze(A)
    if A.ndim != 2:
        raise ValueError(f"dd must be 2D after squeeze/reduce, got {A.shape}")
    B, N = A.shape
    # Handle (nodes, bins) transposed case
    if B != num_bins and N == num_bins:
        A = A.T
        B, N = A.shape
    if B != num_bins or N != num_nodes:
        raise ValueError(f"dd shape {A.shape} != (num_bins={num_bins}, num_nodes={num_nodes})")
    return A.astype(np.float32, copy=False)

def bin_redshifts(z_array, zmin=0.0, zmax=2.2, num_bins=11):
    edges = np.linspace(zmin, zmax, num_bins + 1, dtype=np.float64)
    return np.clip(np.digitize(z_array, bins=edges) - 1, 0, num_bins - 1).astype(np.int32)

# ---------------- core controller ----------------
class SOMControlFastDD:
    """
    Fast SOM controller with an O(B) per-step DD-only proxy:

      Cache: DD_sum[:, j] = sum over nodes in photometric bin j of dd[:, node]
      Proxy: diag(norm(DD_sum)) - mean(offdiag(norm(DD_sum)))

    Optional: fast TreeCorr-style readout using cached sums (DR/RD/RR).
    """
    def __init__(self, node_indices_photo, initial_node_bins, dd, dr=None, rd=None, rr=None, num_bins=11):
        self.num_bins  = int(num_bins)
        self.current_bins = np.asarray(initial_node_bins, dtype=np.int32)
        self.num_nodes = int(self.current_bins.size)

        # per-photo galaxy -> node ids (photometric-only is fine)
        self.node_indices = np.asarray(node_indices_photo, dtype=np.int64)
        self.node_gal_counts = np.bincount(self.node_indices, minlength=self.num_nodes).astype(np.int64)

        # correlations
        self.dd = _coerce_dd(dd, self.num_bins, self.num_nodes)
        self.has_full_corr = (dr is not None and rd is not None and rr is not None)
        if self.has_full_corr:
            self.dr = _as_2d_bins_nodes(dr, "dr", self.num_bins, self.num_nodes)
            self.rd = _as_2d_bins_nodes(rd, "rd", self.num_bins, self.num_nodes)
            self.rr = _as_2d_bins_nodes(rr, "rr", self.num_bins, self.num_nodes)

        self.reset()

    def reset(self):
        # photometric bin counts (weighted by galaxies per node)
        self.galaxy_bin_counts = np.bincount(
            self.current_bins, weights=self.node_gal_counts, minlength=self.num_bins
        ).astype(np.float64)

        # cache DD_sum (ref_bin x phot_bin)
        self.DD_sum = np.zeros((self.num_bins, self.num_bins), dtype=np.float64)
        for n in range(self.num_nodes):
            j = int(self.current_bins[n])
            self.DD_sum[:, j] += self.dd[:, n]

        if self.has_full_corr:
            self.DR_sum = np.zeros_like(self.DD_sum)
            self.RD_sum = np.zeros_like(self.DD_sum)
            self.RR_sum = np.zeros_like(self.DD_sum)
            for n in range(self.num_nodes):
                j = int(self.current_bins[n])
                self.DR_sum[:, j] += self.dr[:, n]
                self.RD_sum[:, j] += self.rd[:, n]
                self.RR_sum[:, j] += self.rr[:, n]
            # proxy scale for ref bins from dd rows
            self.n_ref_proxy = self.dd.sum(axis=1).astype(np.float64) + 1e-12

    def set_node_bin(self, node_idx, new_bin):
        node = int(node_idx)
        newb = int(new_bin)
        oldb = int(self.current_bins[node])
        if newb == oldb:
            return

        # rank-1 updates (O(B))
        col_dd = self.dd[:, node]
        self.DD_sum[:, oldb] -= col_dd
        self.DD_sum[:, newb] += col_dd

        if self.has_full_corr:
            self.DR_sum[:, oldb] -= self.dr[:, node]; self.DR_sum[:, newb] += self.dr[:, node]
            self.RD_sum[:, oldb] -= self.rd[:, node]; self.RD_sum[:, newb] += self.rd[:, node]
            self.RR_sum[:, oldb] -= self.rr[:, node]; self.RR_sum[:, newb] += self.rr[:, node]

        # photometric counts
        w = int(self.node_gal_counts[node])
        self.galaxy_bin_counts[oldb] -= w
        self.galaxy_bin_counts[newb] += w

        self.current_bins[node] = newb

    # -------- rewards --------
    def proxy_reward_dd(self):
        M = self.DD_sum
        tot = float(M.sum())
        if tot <= 0.0:
            return 0.0
        N = M / tot
        diag = float(np.trace(N))
        off  = float(N.sum() - diag) / (self.num_bins * (self.num_bins - 1))
        return diag - off

    def total_reward_fast(self):
        if not self.has_full_corr:
            raise RuntimeError("Need dr/rd/rr for total_reward_fast.")
        n_phot = self.galaxy_bin_counts.astype(np.float64) + 1e-12
        n_ref  = self.n_ref_proxy.astype(np.float64) + 1e-12
        DDn = self.DD_sum / (n_ref[:, None] * n_phot[None, :])
        DRn = self.DR_sum / (10.0 * n_ref[:, None] * n_phot[None, :])
        RDn = self.RD_sum / (10.0 * n_ref[:, None] * n_phot[None, :])
        RRn = self.RR_sum / (10.0 * 10.0 * n_ref[:, None] * n_phot[None, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = np.divide(DDn - DRn - RDn + RRn, RRn, out=np.zeros_like(RRn), where=RRn > 0)
        return float(np.trace(xi))

# ---------------- main: load, compute optimal, run trace ----------------
if __name__ == "__main__":
    ROOT = "Benjamin Work/data"
    NPZ_PATH = os.path.join(ROOT, "correlations_dl", "npairs_compact.npz")
    INIT_BINS_PATH = os.path.join(ROOT, "nodes_labels_initial_fc_200_ext.npy")
    NODE_IDX_PATH  = os.path.join(ROOT, "node_indices_fc.npy")  # photometric only (len ~ accepted - reference)

    print("Loading correlations...")
    npz = np.load(NPZ_PATH)
    dd, dr, rd, rr = npz['dd'], npz['dr'], npz['rd'], npz['rr']
    print("  dd raw shape:", dd.shape)

    print("Loading node/galaxy arrays...")
    initial_node_bins = np.load(INIT_BINS_PATH)   # (num_nodes,)
    node_indices = np.load(NODE_IDX_PATH)         # photometric-only galaxy -> node
    num_bins = 11
    num_nodes = initial_node_bins.size
    print(f"  nodes: {num_nodes}, photo galaxies: {node_indices.size}")

    # --- Build controller ---
    print("Initializing controller (fast DD proxy)...")
    som = SOMControlFastDD(node_indices_photo=node_indices,
                           initial_node_bins=initial_node_bins,
                           dd=dd, dr=dr, rd=rd, rr=rr,
                           num_bins=num_bins)

    # --- Compute true-redshift-based optimal bins (your exact flow) ---
    print("\nComputing true-redshift-based optimal bins...")
    accept            = np.load(os.path.join(ROOT, "accept_fc.npy"))               # (N_full,) bool
    photometric_sample= np.load(os.path.join(ROOT, "photometric_sample_fc.npy"))   # (N_photo,) full idx  (not used directly here)
    reference_sample  = np.load(os.path.join(ROOT, "reference_sample_fc.npy"))     # (N_ref,) full idx

    # Load full catalogue redshifts and reduce to "accepted" rows
    true_redshift = pd.read_parquet(os.path.join(ROOT, "full_catalogue.parquet"), columns=['redshift'])
    # Drop last row (as per your code)
    true_redshift.drop(index=true_redshift.index[-1], axis=0, inplace=True)
    # Keep only accepted galaxies
    true_redshift.drop(index=np.arange(len(accept))[accept == False], axis=0, inplace=True)
    true_redshift.reset_index(inplace=True, drop=True)

    # Remove reference galaxies to get photometric-only array aligned with node_indices
    true_redshift_phot = true_redshift.drop(index=reference_sample, inplace=False)['redshift'].to_numpy()

    # Bin true redshifts to 11 bins
    true_bins_phot = bin_redshifts(true_redshift_phot, zmin=0.0, zmax=2.2, num_bins=num_bins)

    # Compute per-node mean true redshift over photometric galaxies
    print("  Aggregating per-node means (photometric only)...")
    nodes_mean_true = np.zeros(num_nodes, dtype=np.float64)
    nodes_size      = np.zeros(num_nodes, dtype=np.float64)
    # node_indices is per-photometric-galaxy
    for i in range(node_indices.size):
        idx = int(node_indices[i])
        nodes_mean_true[idx] += float(true_redshift_phot[i])
        nodes_size[idx]      += 1.0

    # Protect against empty nodes
    nodes_mean_true = np.divide(nodes_mean_true, nodes_size,
                                out=np.zeros_like(nodes_mean_true),
                                where=nodes_size > 0.0)

    optimal_node_bins = bin_redshifts(nodes_mean_true, zmin=0.0, zmax=2.2, num_bins=num_bins)
    print("  Done. (Computed optimal bin per node from true photo-z means)")

    # --- Baselines ---
    t0 = time.time(); pr = som.proxy_reward_dd(); t1 = time.time()
    print(f"\nFast DD proxy (initial): {pr:.6f}  (t = {t1 - t0:.4f}s)")

    try:
        t0 = time.time(); tr = som.total_reward_fast(); t1 = time.time()
        print(f"Fast TreeCorr-style trace (initial): {tr:.6f}  (t = {t1 - t0:.4f}s)")
    except RuntimeError as e:
        print("TreeCorr fast readout skipped:", e)

    # --- Toward / Away traces using your optimal bins ---
    # ==================== Toward & Away (effective steps), plot + checkpoint tables ====================
    import numpy as np
    import matplotlib.pyplot as plt
    # If your controller class is SOMControlFastDD, keep as-is; otherwise change here:
    ControllerClass = som.__class__

    rng = np.random.default_rng(2025)
    HORIZON = 2500
    CHECKS  = set(range(0, HORIZON + 1, 500))

    def run_series(direction="toward"):
        """Return (steps_eff, proxy_vals, tc_fast_vals, records) for a given direction."""
        # Start from the SAME initial state for both runs
        som_run = ControllerClass(
            node_indices_photo=node_indices,
            initial_node_bins=initial_node_bins,
            dd=dd, dr=dr, rd=rd, rr=rr,
            num_bins=num_bins
        )

        steps_eff   = [0]
        proxy_vals  = [float(som_run.proxy_reward_dd())]
        tc_fast_vals= [float(som_run.total_reward_fast())]
        records = [(0, proxy_vals[0], tc_fast_vals[0])]

        eff = 0
        while eff < HORIZON:
            n = int(rng.integers(0, num_nodes))
            cur = int(som_run.current_bins[n])
            opt = int(optimal_node_bins[n])

            if not (0 <= opt < num_bins):
                continue

            if direction == "toward":
                new_bin = opt
                if new_bin == cur:
                    continue
            else:  # "away"
                # choose any bin that's neither current nor optimal
                candidates = [b for b in range(num_bins) if b != cur and b != opt]
                if not candidates:
                    continue
                new_bin = int(rng.choice(candidates))

            som_run.set_node_bin(n, new_bin)
            eff += 1

            steps_eff.append(eff)
            proxy_vals.append(float(som_run.proxy_reward_dd()))
            tc_fast_vals.append(float(som_run.total_reward_fast()))

            if eff in CHECKS:
                records.append((eff, proxy_vals[-1], tc_fast_vals[-1]))

        return np.array(steps_eff), np.array(proxy_vals), np.array(tc_fast_vals), records

    # Run both directions
    steps_t, proxy_t, tc_t, rec_t = run_series("toward")
    steps_a, proxy_a, tc_a, rec_a = run_series("away")

    # Plot Δ from each run's own start; x<0 for toward, x>0 for away (so you see both on one axis)
    plt.figure(figsize=(10,5))
    plt.plot(-steps_t, tc_t  - tc_t[0],  label="tc_fast Δ (toward)", alpha=0.9)
    plt.plot(-steps_t, proxy_t - proxy_t[0], label="proxy Δ (toward)", alpha=0.9)
    plt.plot( steps_a, tc_a  - tc_a[0],  label="tc_fast Δ (away)",   alpha=0.8, linestyle="--")
    plt.plot( steps_a, proxy_a - proxy_a[0], label="proxy Δ (away)",   alpha=0.8, linestyle="--")
    plt.axvline(0, color="gray", linestyle=":")
    plt.xlabel("Effective steps (negative = toward, positive = away)")
    plt.ylabel("Reward change from each run's start")
    plt.title("Fast TreeCorr vs Proxy: Toward vs Away (effective steps)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("reward_series_toward_away_effective.png", dpi=150)
    plt.show()

    # Print checkpoint tables (every 500 effective steps) for both runs
    def print_table(records, title):
        print(f"\n{title}")
        print(f"{'steps':>6}  {'fast_proxy':>12}  {'tc_fast':>12}  {'Δfast':>10}  {'Δtc_fast':>10}")
        prev_fp, prev_tc = records[0][1], records[0][2]
        for s, fp, tc in records:
            print(f"{s:6d}  {fp:12.6f}  {tc:12.6f}  {fp-prev_fp:10.6f}  {tc-prev_tc:10.6f}")
            prev_fp, prev_tc = fp, tc

    print_table(rec_t, "Checkpoint comparison (TOWARD optimal)")
    print_table(rec_a, "Checkpoint comparison (AWAY from optimal)")



    # --- Micro-benchmark of inner loop ---
    N = 1000
    change_times = np.empty(N, dtype=np.float64)
    proxy_times  = np.empty(N, dtype=np.float64)
    for k in range(N):
        n  = int(rng.integers(0, num_nodes))
        ob = int(som.current_bins[n])
        nb = int((ob + 1 + rng.integers(0, num_bins - 1)) % num_bins)
        t0 = time.time(); som.set_node_bin(n, nb); t1 = time.time()
        change_times[k] = t1 - t0
        t0 = time.time(); _ = som.proxy_reward_dd(); t1 = time.time()
        proxy_times[k] = t1 - t0
    print("\n=== 1000-step benchmark ===")
    print(f"Avg bin-change time: {change_times.mean():.6f}s (median {np.median(change_times):.6f}s)")
    print(f"Avg proxy eval time: {proxy_times.mean():.6f}s (median {np.median(proxy_times):.6f}s)")


    # === Full TreeCorr vs Fast Proxy: checkpoint table every 500 steps (to 2500) ===
    print("\n=== Full TreeCorr vs Fast Proxy: checkpoint table every 500 steps (to 2500) ===")
    # --- OPTIONAL: load TreeCorr objects (for checkpoint evaluation only) ---
    import pickle, os
    import treecorr

    DD_TC_PKL = os.path.join(ROOT, "correlations_dl", "dd.pkl")
    DR_TC_PKL = os.path.join(ROOT, "correlations_dl", "dr.pkl")
    RD_TC_PKL = os.path.join(ROOT, "correlations_dl", "rd.pkl")
    RR_TC_PKL = os.path.join(ROOT, "correlations_dl", "rr.pkl")

    if all(os.path.exists(p) for p in [DD_TC_PKL, DR_TC_PKL, RD_TC_PKL, RR_TC_PKL]):
        with open(DD_TC_PKL, "rb") as f: dd_tc = pickle.load(f)
        with open(DR_TC_PKL, "rb") as f: dr_tc = pickle.load(f)
        with open(RD_TC_PKL, "rb") as f: rd_tc = pickle.load(f)
        with open(RR_TC_PKL, "rb") as f: rr_tc = pickle.load(f)
        has_tc_objects = True
        # Optional sanity: shapes should be (num_bins, num_nodes)
        print("[TC] Loaded TreeCorr columns:", type(dd_tc), "example entry type:", type(dd_tc[0,0]))
    else:
        has_tc_objects = False
        print("[TC] Pickled TreeCorr columns not found; will skip TC-object checkpoints.")

    # --- TreeCorr-object trace using __iadd__ into fresh accumulators ---
    def _calc_xi_scalar(dd_tot, rr_tot, dr_tot, rd_tot, nbins=1):
        """Call calculateXi compatibly across TreeCorr versions and return a scalar xi."""
        out = dd_tot.calculateXi(rr=rr_tot, dr=dr_tot, rd=rd_tot)
        # out can be (xi, varxi) or (xi, varxi, weight) or even just xi in some contexts
        if isinstance(out, tuple):
            xi_vals = out[0]
        else:
            xi_vals = out
        # nbins=1 -> xi_vals is length-1 array; otherwise take mean or choose index
        if np.ndim(xi_vals) == 0:
            return float(xi_vals)
        elif xi_vals.size == 1:
            return float(xi_vals[0])
        else:
            # if you ever run nbins>1, choose how to summarize; mean is a safe fallback
            return float(np.mean(xi_vals))

    def treecorr_trace_from_columns(current_bins, dd_cols, dr_cols, rd_cols, rr_cols,
                                    num_bins=11, min_sep=0.01, max_sep=0.1, sep_units='degrees', nbins=1):
        """
        current_bins: (num_nodes,) ints 0..num_bins-1
        dd_cols/dr_cols/rd_cols/rr_cols: arrays shaped (num_bins, num_nodes) of TreeCorr NNCorrelation columns
        Returns (trace_xi, xi_matrix)
        """
        bins_nodes = [np.where(current_bins == i)[0] for i in range(num_bins)]
        xi_bins = np.zeros((num_bins, num_bins), dtype=np.float64)

        for j in range(num_bins):      # reference bin (row)
            for i in range(num_bins):  # photometric bin (col)
                dd_tot = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units)
                dr_tot = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units)
                rd_tot = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units)
                rr_tot = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units)

                for k in bins_nodes[i]:
                    dd_tot.__iadd__(dd_cols[j, k])
                    dr_tot.__iadd__(dr_cols[j, k])
                    rd_tot.__iadd__(rd_cols[j, k])
                    rr_tot.__iadd__(rr_cols[j, k])

                dd_tot.finalize(); dr_tot.finalize(); rd_tot.finalize(); rr_tot.finalize()
                xi_bins[i, j] = _calc_xi_scalar(dd_tot, rr_tot, dr_tot, rd_tot, nbins=nbins)

        return float(np.trace(xi_bins)), xi_bins

    # --- Checkpoints every 500 effective steps (toward optimal), up to 2500 ---
    checkpoints = set(range(0, 2500 + 1, 500))
    records = []  # (steps, fast_proxy, tc_fast_trace, tc_object_trace_or_nan)

    # initial
    fp0 = float(som.proxy_reward_dd())
    tc_fast0 = float(som.total_reward_fast())  # uses cached sums from your class
    if has_tc_objects:
        tc_obj0, _ = treecorr_trace_from_columns(som.current_bins, dd_tc, dr_tc, rd_tc, rr_tc, num_bins=num_bins)
    else:
        tc_obj0 = float("nan")
    records.append((0, fp0, tc_fast0, tc_obj0))

    rng = np.random.default_rng(42)
    effective = 0
    while effective < 2500:
        node = int(rng.integers(0, num_nodes))
        new_bin = int(optimal_node_bins[node])
        if not (0 <= new_bin < num_bins):  # guard
            continue
        if new_bin == int(som.current_bins[node]):
            continue

        som.set_node_bin(node, new_bin)
        effective += 1

        if effective in checkpoints:
            fp = float(som.proxy_reward_dd())
            tc_fast = float(som.total_reward_fast())
            if has_tc_objects:
                tc_obj, _ = treecorr_trace_from_columns(som.current_bins, dd_tc, dr_tc, rd_tc, rr_tc, num_bins=num_bins)
            else:
                tc_obj = float("nan")
            records.append((effective, fp, tc_fast, tc_obj))

    # --- Print comparison table ---
    print("\nCheckpoint comparison (toward optimal)")
    print(f"{'steps':>6}  {'fast_proxy':>12}  {'tc_fast':>12}  {'tc_object':>12}  {'Δfast':>10}  {'Δtc_fast':>10}  {'Δtc_obj':>10}")
    prev_fp, prev_tcf, prev_tco = records[0][1], records[0][2], records[0][3]
    for s, fp, tcf, tco in records:
        d_fp  = fp  - prev_fp
        d_tcf = tcf - prev_tcf if np.isfinite(tcf) and np.isfinite(prev_tcf) else float("nan")
        d_tco = tco - prev_tco if np.isfinite(tco) and np.isfinite(prev_tco) else float("nan")
        print(f"{s:6d}  {fp:12.6f}  {tcf:12.6f}  {tco:12.6f}  {d_fp:10.6f}  {d_tcf:10.6f}  {d_tco:10.6f}")
        prev_fp, prev_tcf, prev_tco = fp, tcf, tco


    # ==================== Incremental ±1 tests: Toward vs Away (effective steps) ====================
    import numpy as np
    import matplotlib.pyplot as plt

    ControllerClass = som.__class__  # use whichever class you already constructed

    HORIZON = 2500
    CHECKS  = set(range(0, HORIZON + 1, 500))

    def make_controller():
        return ControllerClass(
            node_indices_photo=node_indices,
            initial_node_bins=initial_node_bins,
            dd=dd, dr=dr, rd=rd, rr=rr,
            num_bins=num_bins
        )

    def step_toward(cur, opt):
        """Return next bin (±1) moving one step closer to opt; None if already at opt."""
        if cur == opt:
            return None
        return cur + (1 if opt > cur else -1)

    def step_away(cur, opt):
        """Return next bin (±1) moving one step farther from opt; None if cannot move farther within [0,B-1]."""
        if cur == opt:
            return None
        # move in opposite direction to step_toward
        newb = cur + (-1 if opt > cur else 1)
        if 0 <= newb < num_bins:
            return newb
        return None  # at edge; can't move farther away

    def run_incremental(direction="toward", seed=2025):
        """Return (steps_eff, proxy_vals, tc_fast_vals, records, som_run)."""
        som_run = make_controller()
        rng = np.random.default_rng(seed)

        steps_eff    = [0]
        proxy_vals   = [float(som_run.proxy_reward_dd())]
        tc_fast_vals = [float(som_run.total_reward_fast())]
        records = [(0, proxy_vals[0], tc_fast_vals[0])]  # checkpoint rows

        eff = 0
        while eff < HORIZON:
            n   = int(rng.integers(0, num_nodes))
            cur = int(som_run.current_bins[n])
            opt = int(optimal_node_bins[n])
            if not (0 <= opt < num_bins):
                continue

            if direction == "toward":
                newb = step_toward(cur, opt)
            else:
                newb = step_away(cur, opt)

            if newb is None or newb == cur:
                continue  # no effective move for this node; pick another

            som_run.set_node_bin(n, int(newb))
            eff += 1

            steps_eff.append(eff)
            proxy_vals.append(float(som_run.proxy_reward_dd()))
            tc_fast_vals.append(float(som_run.total_reward_fast()))

            if eff in CHECKS:
                records.append((eff, proxy_vals[-1], tc_fast_vals[-1]))

        return (np.array(steps_eff),
                np.array(proxy_vals),
                np.array(tc_fast_vals),
                records,
                som_run)

    # ---- Run both directions with ±1 moves ----
    steps_t, proxy_t, tc_t, rec_t, som_t = run_incremental("toward", seed=11)
    steps_a, proxy_a, tc_a, rec_a, som_a = run_incremental("away",   seed=12)

    # ---- Noise stats (Δ per effective step) ----
    def print_noise_stats(name, series):
        d = np.diff(series)
        print(f"[{name}] mean Δ per step: {d.mean():.6e}  (+/−/0 = {(d>0).sum()}/{(d<0).sum()}/{(d==0).sum()})")

    print_noise_stats("tc_fast (toward)", tc_t)
    print_noise_stats("proxy  (toward)",  proxy_t)
    print_noise_stats("tc_fast (away)",   tc_a)
    print_noise_stats("proxy  (away)",    proxy_a)

    # ---- Checkpoint tables (every 500 effective steps) ----
    def print_table(records, title):
        print(f"\n{title}")
        print(f"{'steps':>6}  {'fast_proxy':>12}  {'tc_fast':>12}  {'Δfast':>10}  {'Δtc_fast':>10}")
        prev_fp, prev_tc = records[0][1], records[0][2]
        for s, fp, tc in records:
            print(f"{s:6d}  {fp:12.6f}  {tc:12.6f}  {fp-prev_fp:10.6f}  {tc-prev_tc:10.6f}")
            prev_fp, prev_tc = fp, tc

    print_table(rec_t, "Checkpoint comparison (TOWARD optimal, ±1 steps)")
    print_table(rec_a, "Checkpoint comparison (AWAY from optimal, ±1 steps)")

    # ---- Plot Δ from each run's own start; x<0 = toward, x>0 = away ----
    plt.figure(figsize=(10,5))
    plt.plot(-steps_t, tc_t - tc_t[0],     label="tc_fast Δ (toward, ±1)", alpha=0.9)
    plt.plot(-steps_t, proxy_t - proxy_t[0], label="proxy Δ (toward, ±1)",   alpha=0.9)
    plt.plot( steps_a, tc_a - tc_a[0],     label="tc_fast Δ (away, ±1)",   alpha=0.8, linestyle="--")
    plt.plot( steps_a, proxy_a - proxy_a[0], label="proxy Δ (away, ±1)",     alpha=0.8, linestyle="--")
    plt.axvline(0, color="gray", linestyle=":")
    plt.xlabel("Effective steps (negative = toward, positive = away)")
    plt.ylabel("Reward change from each run's start")
    plt.title("Incremental ±1 moves: Fast TreeCorr vs Proxy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("reward_series_incremental_pm1.png", dpi=150)
    plt.show()
