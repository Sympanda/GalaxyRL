# som_fast_tc_only.py
import os, time, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def _as_2d_bins_nodes(x, name, B=None, N=None, dtype=np.float32):
    A = np.asarray(x)
    if A.ndim > 2:
        if A.shape[-1] == 1:
            A = A[..., 0]
        else:
            A = A.sum(axis=-1)
    A = np.squeeze(A)
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {A.shape}")
    if B is not None and N is not None:
        r, c = A.shape
        if r != B and c == B:
            A = A.T
            r, c = A.shape
        if r != B or c != N:
            raise ValueError(f"{name} has shape {A.shape}, expected ({B},{N})")
    return A.astype(dtype, copy=False)

def bin_redshifts(z_array, zmin=0.0, zmax=2.2, num_bins=11):
    edges = np.linspace(zmin, zmax, num_bins + 1, dtype=np.float64)
    return np.clip(np.digitize(z_array, bins=edges) - 1, 0, num_bins - 1).astype(np.int32)

# ----------------- core: Fast TreeCorr only -----------------
class SOMFastTC:
    """
    Minimal SOM controller for fast TreeCorr-style reward.

    Keeps cached per-photometric-bin column sums:
      DD_sum[:, j] = sum_n in phot-bin j  dd[:, n]
    likewise for DR_sum, RD_sum, RR_sum.

    Methods:
      - set_node_bin(node, new_bin): O(B) rank-1 cache update
      - total_reward_fast(): Tr( (DD-DR-RD+RR)/RR ) with proper normalizations
    """
    def __init__(self, node_indices_photo, initial_node_bins, dd, dr, rd, rr, num_bins=11):
        self.num_bins  = int(num_bins)
        self.current_bins = np.asarray(initial_node_bins, dtype=np.int32)
        self.num_nodes = int(self.current_bins.size)

        self.node_indices = np.asarray(node_indices_photo, dtype=np.int64)
        self.node_gal_counts = np.bincount(self.node_indices, minlength=self.num_nodes).astype(np.int64)

        self.dd = _as_2d_bins_nodes(dd, "dd", self.num_bins, self.num_nodes, np.float32)
        self.dr = _as_2d_bins_nodes(dr, "dr", self.num_bins, self.num_nodes, np.float32)
        self.rd = _as_2d_bins_nodes(rd, "rd", self.num_bins, self.num_nodes, np.float32)
        self.rr = _as_2d_bins_nodes(rr, "rr", self.num_bins, self.num_nodes, np.float32)

        self.reset()

    def reset(self):
        # photometric counts (weighted by galaxies per node)
        self.galaxy_bin_counts = np.bincount(
            self.current_bins, weights=self.node_gal_counts, minlength=self.num_bins
        ).astype(np.float64)

        # cached sums per photometric bin
        B = self.num_bins
        self.DD_sum = np.zeros((B, B), dtype=np.float64)
        self.DR_sum = np.zeros((B, B), dtype=np.float64)
        self.RD_sum = np.zeros((B, B), dtype=np.float64)
        self.RR_sum = np.zeros((B, B), dtype=np.float64)
        for n in range(self.num_nodes):
            j = int(self.current_bins[n])
            self.DD_sum[:, j] += self.dd[:, n]
            self.DR_sum[:, j] += self.dr[:, n]
            self.RD_sum[:, j] += self.rd[:, n]
            self.RR_sum[:, j] += self.rr[:, n]

        # proxy for per-reference-bin "count" scale from dd mass (used only in normalization)
        self.n_ref_proxy = self.dd.sum(axis=1).astype(np.float64) + 1e-12

    def set_node_bin(self, node_idx, new_bin):
        node = int(node_idx)
        newb = int(new_bin)
        oldb = int(self.current_bins[node])
        if newb == oldb:
            return

        # rank-1 updates (O(B))
        self.DD_sum[:, oldb] -= self.dd[:, node]; self.DD_sum[:, newb] += self.dd[:, node]
        self.DR_sum[:, oldb] -= self.dr[:, node]; self.DR_sum[:, newb] += self.dr[:, node]
        self.RD_sum[:, oldb] -= self.rd[:, node]; self.RD_sum[:, newb] += self.rd[:, node]
        self.RR_sum[:, oldb] -= self.rr[:, node]; self.RR_sum[:, newb] += self.rr[:, node]

        # counts per photometric bin
        w = int(self.node_gal_counts[node])
        self.galaxy_bin_counts[oldb] -= w
        self.galaxy_bin_counts[newb] += w

        self.current_bins[node] = newb

    def total_reward_fast(self):
        n_phot = self.galaxy_bin_counts.astype(np.float64) + 1e-12     # (phot_bin,)
        n_ref  = self.n_ref_proxy.astype(np.float64) + 1e-12            # (ref_bin,)

        DDn = self.DD_sum / (n_ref[:, None] * n_phot[None, :])
        DRn = self.DR_sum / (10.0 * n_ref[:, None] * n_phot[None, :])
        RDn = self.RD_sum / (10.0 * n_ref[:, None] * n_phot[None, :])
        RRn = self.RR_sum / (10.0 * 10.0 * n_ref[:, None] * n_phot[None, :])

        with np.errstate(divide='ignore', invalid='ignore'):
            xi = np.divide(DDn - DRn - RDn + RRn, RRn, out=np.zeros_like(RRn), where=RRn > 0)
        return float(np.trace(xi))

# ----------------- optional: real TreeCorr from pickled columns -----------------
def _tc_calc_scalar(dd_tot, rr_tot, dr_tot, rd_tot, nbins=1):
    out = dd_tot.calculateXi(rr=rr_tot, dr=dr_tot, rd=rd_tot)
    if isinstance(out, tuple):
        xi_vals = out[0]
    else:
        xi_vals = out
    if np.ndim(xi_vals) == 0:
        return float(xi_vals)
    if np.size(xi_vals) == 1:
        return float(xi_vals.ravel()[0])
    return float(np.mean(xi_vals))

def treecorr_trace_from_columns(current_bins, dd_cols, dr_cols, rd_cols, rr_cols,
                                num_bins=11, min_sep=0.01, max_sep=0.1, sep_units='degrees', nbins=1):
    import treecorr
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
            xi_bins[i, j] = _tc_calc_scalar(dd_tot, rr_tot, dr_tot, rd_tot, nbins=nbins)

    return float(np.trace(xi_bins)), xi_bins

# ----------------- main -----------------
if __name__ == "__main__":
    ROOT = "Benjamin Work/data"
    NPZ = os.path.join(ROOT, "correlations_dl", "npairs_compact.npz")
    INIT = os.path.join(ROOT, "nodes_labels_initial_fc_200_ext.npy")
    NODE = os.path.join(ROOT, "node_indices_fc.npy")  # photometric-only node ids
    num_bins = 11

    print("[load] correlations...")
    npz = np.load(NPZ)
    dd, dr, rd, rr = npz['dd'], npz['dr'], npz['rd'], npz['rr']

    print("[load] labels / indices...")
    initial_node_bins = np.load(INIT)   # (num_nodes,)
    node_indices = np.load(NODE)        # per-photometric galaxy node id
    num_nodes = initial_node_bins.size

    # Build controller
    som = SOMFastTC(node_indices_photo=node_indices,
                    initial_node_bins=initial_node_bins,
                    dd=dd, dr=dr, rd=rd, rr=rr,
                    num_bins=num_bins)

    # ----- true-z optimal bins (exactly your flow) -----
    print("[optimal] computing per-node optimal bins from true-z means...")
    accept            = np.load(os.path.join(ROOT, "accept_fc.npy"))
    reference_sample  = np.load(os.path.join(ROOT, "reference_sample_fc.npy"))

    true_redshift = pd.read_parquet(os.path.join(ROOT, "full_catalogue.parquet"), columns=['redshift'])
    true_redshift.drop(index=true_redshift.index[-1], axis=0, inplace=True)
    true_redshift.drop(index=np.arange(len(accept))[accept == False], axis=0, inplace=True)
    true_redshift.reset_index(inplace=True, drop=True)

    true_redshift_phot = true_redshift.drop(index=reference_sample, inplace=False)['redshift'].to_numpy()

    # per-node mean true-z across photometric galaxies
    nodes_mean_true = np.zeros(num_nodes, dtype=np.float64)
    nodes_size      = np.zeros(num_nodes, dtype=np.float64)
    for i in range(node_indices.size):
        idx = int(node_indices[i])
        nodes_mean_true[idx] += float(true_redshift_phot[i])
        nodes_size[idx]      += 1.0
    nodes_mean_true = np.divide(nodes_mean_true, nodes_size,
                                out=np.zeros_like(nodes_mean_true), where=nodes_size>0)

    optimal_node_bins = bin_redshifts(nodes_mean_true, zmin=0.0, zmax=2.2, num_bins=num_bins)

    # ----- optional: load pickled TreeCorr columns -----
    DD_PKL = os.path.join(ROOT, "correlations_dl", "dd.pkl")
    DR_PKL = os.path.join(ROOT, "correlations_dl", "dr.pkl")
    RD_PKL = os.path.join(ROOT, "correlations_dl", "rd.pkl")
    RR_PKL = os.path.join(ROOT, "correlations_dl", "rr.pkl")
    if all(os.path.exists(p) for p in [DD_PKL, DR_PKL, RD_PKL, RR_PKL]):
        with open(DD_PKL, "rb") as f: dd_tc = pickle.load(f)
        with open(DR_PKL, "rb") as f: dr_tc = pickle.load(f)
        with open(RD_PKL, "rb") as f: rd_tc = pickle.load(f)
        with open(RR_PKL, "rb") as f: rr_tc = pickle.load(f)
        have_tc = True
        print("[tc] Loaded pickled TreeCorr columns.")
    else:
        have_tc = False
        print("[tc] Pickled TreeCorr columns not found; real-TC comparisons will be skipped.")

    # ----- helper to run a series of effective steps -----
    RNG = np.random.default_rng(42)
    HORIZON = 2500
    CHECKS  = set(range(0, HORIZON + 1, 500))

    def run_jump_series(direction="toward"):
        # fresh instance per run
        s = SOMFastTC(node_indices, initial_node_bins, dd, dr, rd, rr, num_bins=num_bins)
        steps = [0]; fast = [s.total_reward_fast()]
        records = [(0, fast[0], None)]
        if have_tc:
            tc0, _ = treecorr_trace_from_columns(s.current_bins, dd_tc, dr_tc, rd_tc, rr_tc, num_bins=num_bins)
            records[0] = (0, fast[0], tc0)

        eff = 0
        while eff < HORIZON:
            n = int(RNG.integers(0, num_nodes))
            cur = int(s.current_bins[n])
            opt = int(optimal_node_bins[n])
            if not (0 <= opt < num_bins):
                continue

            if direction == "toward":
                newb = opt
                if newb == cur: 
                    continue
            else:  # away
                cands = [b for b in range(num_bins) if b != cur and b != opt]
                if not cands:
                    continue
                newb = int(RNG.choice(cands))

            s.set_node_bin(n, newb)
            eff += 1
            steps.append(eff); fast.append(s.total_reward_fast())

            if eff in CHECKS:
                if have_tc:
                    tc_trace, _ = treecorr_trace_from_columns(s.current_bins, dd_tc, dr_tc, rd_tc, rr_tc, num_bins=num_bins)
                    records.append((eff, fast[-1], tc_trace))
                else:
                    records.append((eff, fast[-1], None))
        return np.array(steps), np.array(fast), records

    def step_pm1_toward(cur, opt):
        if cur == opt: return None
        return cur + (1 if opt > cur else -1)

    def step_pm1_away(cur, opt):
        if cur == opt: return None
        newb = cur + (-1 if opt > cur else 1)
        return newb if 0 <= newb < num_bins else None

    def run_pm1_series(direction="toward"):
        s = SOMFastTC(node_indices, initial_node_bins, dd, dr, rd, rr, num_bins=num_bins)
        steps = [0]; fast = [s.total_reward_fast()]
        records = [(0, fast[0])]
        eff = 0
        while eff < HORIZON:
            n = int(RNG.integers(0, num_nodes))
            cur = int(s.current_bins[n])
            opt = int(optimal_node_bins[n])
            if not (0 <= opt < num_bins): 
                continue
            newb = step_pm1_toward(cur, opt) if direction=="toward" else step_pm1_away(cur, opt)
            if newb is None or newb == cur: 
                continue
            s.set_node_bin(n, newb)
            eff += 1
            steps.append(eff); fast.append(s.total_reward_fast())
            if eff in CHECKS:
                records.append((eff, fast[-1]))
        return np.array(steps), np.array(fast), records

    # ----- RUN: jump to optimal (toward/away) -----
    print("\n[JUMP] Toward optimal...")
    st_t, ft_t, rec_t = run_jump_series("toward")
    print("steps  tc_fast     ", "tc_object" if have_tc else "")
    prev_f, prev_tc = rec_t[0][1], rec_t[0][2]
    for (s, f, tco) in rec_t:
        df = f - prev_f; prev_f = f
        if have_tc:
            dt = tco - prev_tc if prev_tc is not None else 0.0
            prev_tc = tco
            print(f"{s:5d} {f:10.6f}  {tco:10.6f}  Δf={df:+.6f}  Δtc={dt:+.6f}")
        else:
            print(f"{s:5d} {f:10.6f}  Δf={df:+.6f}")

    print("\n[JUMP] Away from optimal...")
    st_a, ft_a, rec_a = run_jump_series("away")
    print("steps  tc_fast     ", "tc_object" if have_tc else "")
    prev_f, prev_tc = rec_a[0][1], rec_a[0][2]
    for (s, f, tco) in rec_a:
        df = f - prev_f; prev_f = f
        if have_tc:
            dt = tco - prev_tc if prev_tc is not None else 0.0
            prev_tc = tco
            print(f"{s:5d} {f:10.6f}  {tco:10.6f}  Δf={df:+.6f}  Δtc={dt:+.6f}")
        else:
            print(f"{s:5d} {f:10.6f}  Δf={df:+.6f}")


    # ----- RUN: ±1 step (toward/away) -----
    print("\n[±1] Toward optimal...")
    sp_t, fp_t, recp_t = run_pm1_series("toward")
    print("steps  tc_fast     ", "tc_object" if have_tc else "")
    prev_f, prev_tc = recp_t[0][1], recp_t[0][2]
    for (s, f, tco) in recp_t:
        df = f - prev_f; prev_f = f
        if have_tc:
            dt = tco - prev_tc if prev_tc is not None else 0.0
            prev_tc = tco
            print(f"{s:5d} {f:10.6f}  {tco:10.6f}  Δf={df:+.6f}  Δtc={dt:+.6f}")
        else:
            print(f"{s:5d} {f:10.6f}  Δf={df:+.6f}")
    print("[±1] Away from optimal...")
    sp_a, fp_a, recp_a = run_pm1_series("away")
    print("steps  tc_fast     ", "tc_object" if have_tc else "")
    prev_f, prev_tc = recp_a[0][1], recp_a[0][2]
    for (s, f, tco) in recp_a:
        df = f - prev_f; prev_f = f
        if have_tc:
            dt = tco - prev_tc if prev_tc is not None else 0.0
            prev_tc = tco
            print(f"{s:5d} {f:10.6f}  {tco:10.6f}  Δf={df:+.6f}  Δtc={dt:+.6f}")
        else:
            print(f"{s:5d} {f:10.6f}  Δf={df:+.6f}")

    # ----- PLOTS: jump vs ±1 (use Δ from each run's start) -----
    plt.figure(figsize=(10,5))
    plt.plot(-st_t, ft_t - ft_t[0], label="tc_fast Δ (jump → toward)", alpha=0.9)
    plt.plot( st_a, ft_a - ft_a[0], label="tc_fast Δ (jump → away)",   alpha=0.9, linestyle="--")
    plt.plot(-sp_t, fp_t - fp_t[0], label="tc_fast Δ (±1 → toward)",   alpha=0.8)
    plt.plot( sp_a, fp_a - fp_a[0], label="tc_fast Δ (±1 → away)",     alpha=0.8, linestyle="--")
    plt.axvline(0, color='gray', linestyle=':')
    plt.xlabel("Effective steps (negative=toward, positive=away)")
    plt.ylabel("Δ tc_fast from run start")
    plt.title("Fast TreeCorr: jump-to-target vs ±1 increments")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("fast_tc_jump_vs_pm1.png", dpi=150); plt.show()

    # ----- TIMING: change, fast-TC, real-TC (once) -----
    print("\n[TIMING] change/proxy-fastTC/realTC(once)")
    s_bench = SOMFastTC(node_indices, initial_node_bins, dd, dr, rd, rr, num_bins=num_bins)
    rngb = np.random.default_rng(7)
    N = 1000
    change_t = np.empty(N); fast_t = np.empty(N)

    # warm-up
    _ = s_bench.total_reward_fast()

    def rand_effective_move():
        while True:
            n  = int(rngb.integers(0, num_nodes))
            ob = int(s_bench.current_bins[n])
            nb = int((ob + 1 + rngb.integers(0, num_bins-1)) % num_bins)
            if nb != ob:
                return n, nb

    for k in range(N):
        n, nb = rand_effective_move()
        t0 = time.time(); s_bench.set_node_bin(n, nb); change_t[k] = time.time() - t0
        t0 = time.time(); _ = s_bench.total_reward_fast(); fast_t[k] = time.time() - t0

    print(f"Avg bin-change time: {change_t.mean():.6f}s (median {np.median(change_t):.6f}s)")
    print(f"Avg fast-TC time:    {fast_t.mean():.6f}s (median {np.median(fast_t):.6f}s)")

    if have_tc:
        t0 = time.time()
        tc_trace_once, _ = treecorr_trace_from_columns(s_bench.current_bins, dd_tc, dr_tc, rd_tc, rr_tc, num_bins=num_bins)
        real_tc_dt = time.time() - t0
        print(f"Real TreeCorr (from columns) once: {real_tc_dt:.3f}s  (trace={tc_trace_once:.6f})")
    else:
        print("Real TreeCorr columns not available; skipped.")
