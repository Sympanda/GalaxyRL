import numpy as np
import copy
import matplotlib.pyplot as plt

def _as_2d_bins_nodes(x, name, num_bins_expected=None, num_nodes_expected=None, dtype=np.float32):
    x = np.asarray(x).squeeze()
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeeze, got {x.shape}")
    if num_bins_expected is not None and x.shape[0] != num_bins_expected:
        raise ValueError(f"{name} shape[0]={x.shape[0]} != num_bins ({num_bins_expected})")
    if num_nodes_expected is not None and x.shape[1] != num_nodes_expected:
        raise ValueError(f"{name} shape[1]={{x.shape[1]}} != num_nodes ({num_nodes_expected})")
    return x.astype(dtype, copy=False)

class SOMControl:
    def __init__(self, node_indices, reference_labels, initial_node_bins, dd, dr, rd, rr, num_bins=11, keep_per_galaxy_labels=True):
        # Strict dtypes
        self.node_indices     = np.asarray(node_indices,     dtype=np.int32)  # len = n_galaxies
        self.reference_labels = np.asarray(reference_labels, dtype=np.int32)  # per-node label
        self.assigned_bins    = np.asarray(initial_node_bins, dtype=np.int32) # per-node bin
        self.num_nodes        = int(self.assigned_bins.size)
        self.num_bins         = int(num_bins)
        self.keep_per_galaxy  = bool(keep_per_galaxy_labels)

        # Immutable counts
        self.node_gal_counts = np.bincount(self.node_indices, minlength=self.num_nodes).astype(np.int32)
        self.ref_bin_counts  = np.bincount(self.reference_labels[:self.num_nodes], minlength=self.num_bins).astype(np.int32)

        # Correlation arrays -> (num_bins, num_nodes), float32
        self.dd = _as_2d_bins_nodes(dd, "dd", num_bins_expected=self.num_bins, num_nodes_expected=self.num_nodes, dtype=np.float32)
        self.dr = _as_2d_bins_nodes(dr, "dr", num_bins_expected=self.num_bins, num_nodes_expected=self.num_nodes, dtype=np.float32)
        self.rd = _as_2d_bins_nodes(rd, "rd", num_bins_expected=self.num_bins, num_nodes_expected=self.num_nodes, dtype=np.float32)
        self.rr = _as_2d_bins_nodes(rr, "rr", num_bins_expected=self.num_bins, num_nodes_expected=self.num_nodes, dtype=np.float32)

        self.reset()

    # ---------- core state ----------
    def reset(self):
        self.current_bins = self.assigned_bins.copy()  # (nodes,)

        # nodes_mask[b, n] == True iff node n assigned to bin b
        self.nodes_mask = np.zeros((self.num_bins, self.num_nodes), dtype=bool)
        self.nodes_mask[self.current_bins, np.arange(self.num_nodes, dtype=np.int32)] = True

        # total galaxies per bin
        self.galaxy_bin_counts = np.bincount(self.current_bins, weights=self.node_gal_counts,
                                             minlength=self.num_bins).astype(np.int64)

        # counts per node × bin: a single spike per row equal to node_gal_counts
        self.photo_counts = np.zeros((self.num_nodes, self.num_bins), dtype=np.int32)
        self.photo_counts[np.arange(self.num_nodes, dtype=np.int32), self.current_bins] = self.node_gal_counts

        # spec per-node one-hot
        self.spec_counts = np.zeros((self.num_nodes, self.num_bins), dtype=np.int32)
        nodes = np.arange(min(self.num_nodes, self.reference_labels.size), dtype=np.int32)
        self.spec_counts[nodes, self.reference_labels[nodes]] += 1

        # Optional per-galaxy labels (skip if not needed for the env state)
        if self.keep_per_galaxy:
            self.photo_labels = self.current_bins[self.node_indices]
            self.galaxy_bin   = self.photo_labels.copy()

    # ---------- updates ----------
    def set_node_bin(self, node_idx, new_bin):
        node_idx = int(node_idx)
        new_bin  = int(new_bin)

        old_bin = int(self.current_bins[node_idx])
        if old_bin == new_bin:
            return

        cnt = int(self.node_gal_counts[node_idx])  # galaxies at this node

        # update aggregates
        self.current_bins[node_idx] = new_bin
        self.nodes_mask[old_bin, node_idx] = False
        self.nodes_mask[new_bin, node_idx] = True
        self.galaxy_bin_counts[old_bin] -= cnt
        self.galaxy_bin_counts[new_bin] += cnt
        self.photo_counts[node_idx, old_bin] -= cnt
        self.photo_counts[node_idx, new_bin] += cnt

        if self.keep_per_galaxy:
            mask = (self.node_indices == node_idx)
            if mask.any():
                self.galaxy_bin[mask] = new_bin
                self.photo_labels = self.galaxy_bin  # view ok

    def get_node_bin(self, node_idx):
        return int(self.current_bins[int(node_idx)])

    def get_galaxy_bins(self):
        if not self.keep_per_galaxy:
            raise RuntimeError("Per-galaxy labels disabled (keep_per_galaxy=False).")
        return self.photo_labels

    # ---------- rewards ----------
    def compute_manual_reward_fast(self):
        """
        Vectorized column-sums via boolean masks; avoids matmul shape pitfalls.
        dd/dr/rd/rr: shape (num_bins, num_nodes)
        """
        xi = np.zeros((self.num_bins, self.num_bins), dtype=np.float32)
        n_ref = self.ref_bin_counts.astype(np.float32)  # (j,)

        for i in range(self.num_bins):
            n_phot = float(self.galaxy_bin_counts[i])
            if n_phot == 0:
                continue

            mask_i = self.nodes_mask[i]  # (num_nodes,) bool

            # column-sums for selected nodes -> (num_bins,)
            DD = self.dd[:, mask_i].sum(axis=1)
            DR = self.dr[:, mask_i].sum(axis=1)
            RD = self.rd[:, mask_i].sum(axis=1)
            RR = self.rr[:, mask_i].sum(axis=1)

            denom_common = n_phot * n_ref
            DDn = np.divide(DD, denom_common, out=np.zeros_like(DD, dtype=np.float32), where=denom_common > 0)
            DRn = np.divide(DR, (n_phot * 10.0 * n_ref), out=np.zeros_like(DR, dtype=np.float32), where=n_ref > 0)
            RDn = np.divide(RD, (10.0 * n_phot * n_ref), out=np.zeros_like(RD, dtype=np.float32), where=n_ref > 0)
            RRn = np.divide(RR, (10.0 * n_phot * 10.0 * n_ref), out=np.zeros_like(RR, dtype=np.float32), where=n_ref > 0)

            with np.errstate(divide='ignore', invalid='ignore'):
                val = np.divide(DDn - DRn - RDn + RRn, RRn, out=np.zeros_like(RRn), where=RRn > 0)
            xi[i, :] = val

        return xi

    def total_reward(self):
        return float(np.trace(self.compute_manual_reward_fast()))

    def proxy_reward(self):
        # overlap = (bins×nodes) @ ((nodes×bins) * node_gal_counts)
        weights = self.node_gal_counts.astype(np.float32)[:, None]  # (nodes,1)
        spec_w  = self.spec_counts * weights                         # (nodes,bins)
        overlap = self.nodes_mask @ spec_w                           # (bins,bins)

        total = overlap.sum()
        if total <= 0:
            return 0.0
        norm = overlap / total
        diag = float(np.trace(norm))
        off  = float(norm.sum() - diag) / (self.num_bins * (self.num_bins - 1))
        return diag - off

    # ---------- utils ----------
    def clone(self):
        return copy.deepcopy(self)

    def plot_som(self, som_size=(200, 200), title="SOM Bin Assignment", cmap='tab10'):
        grid = self.current_bins.reshape(som_size)
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap=cmap, origin='lower')
        plt.title(title); plt.xlabel("SOM X"); plt.ylabel("SOM Y")
        cbar = plt.colorbar(); cbar.set_label("Redshift Bin")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    import time

    print("Loading precomputed npairs arrays...")
    npz = np.load("Benjamin Work/data/correlations_dl/npairs_compact.npz")
    dd = npz['dd']; dr = npz['dr']; rd = npz['rd']; rr = npz['rr']
    print("Done.")

    node_indices      = np.load("Benjamin Work/data/node_indices_fc.npy")
    reference_labels  = np.load("Benjamin Work/data/reference_labels_fc_fc.npy")
    initial_node_bins = np.load("Benjamin Work/data/nodes_labels_initial_fc_200_ext.npy")

    print("Initializing SOMControl...")
    som = SOMControl(node_indices, reference_labels, initial_node_bins, dd, dr, rd, rr, keep_per_galaxy_labels=True)

    # Baselines
    print("Calculating initial total reward...")
    t0 = time.time(); tr = som.total_reward(); t1 = time.time()
    print("Total reward (TreeCorr-style):", tr)
    print("Time taken:", t1 - t0)

    t0 = time.time(); pr = som.proxy_reward(); t1 = time.time()
    print("Proxy reward:", pr)
    print("Time taken:", t1 - t0)

    # Single change sanity
    print("\nChanging one node's bin...")
    test_node = 10000
    old_bin = som.get_node_bin(test_node)
    new_bin = int((old_bin + 1) % som.num_bins)
    t0 = time.time(); som.set_node_bin(test_node, new_bin); t1 = time.time()
    print(f"Changed node {test_node} from bin {old_bin} to bin {new_bin}")
    print("Time to update bin:", t1 - t0)

    t0 = time.time(); pr = som.proxy_reward(); t1 = time.time()
    print("Proxy reward after update:", pr)
    print("Time taken:", t1 - t0)

    # ---------- 1000-step RL-style benchmark ----------
    start = time.time()
    rng = np.random.default_rng(42)
    N = 1000
    change_times = np.empty(N, dtype=np.float64)
    proxy_times  = np.empty(N, dtype=np.float64)
    total_times  = np.empty(N, dtype=np.float64)

    for k in range(N):
        n  = int(rng.integers(0, som.num_nodes))
        ob = som.get_node_bin(n)
        nb = int((ob + 1 + rng.integers(0, som.num_bins - 1)) % som.num_bins)

        t0 = time.time(); som.set_node_bin(n, nb); t1 = time.time()
        change_times[k] = t1 - t0

        t0 = time.time(); _ = som.proxy_reward(); t1 = time.time()
        proxy_times[k] = t1 - t0

        t0 = time.time(); _ = som.total_reward(); t1 = time.time()
        total_times[k] = t1 - t0

    print("\n=== 1000-step benchmark ===")
    print(f"Avg bin-change time:   {change_times.mean():.6f}s (median {np.median(change_times):.6f}s)")
    print(f"Avg proxy reward time: {proxy_times.mean():.6f}s (median {np.median(proxy_times):.6f}s)")
    print(f"Avg total reward time: {total_times.mean():.6f}s (median {np.median(total_times):.6f}s)")
    print(f"Total time for 1000 steps: {time.time() - start:.2f}s")

    ###### Plots for correlating the rewards with optimal steps ######
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import time
    from collections import defaultdict

    print("\n\n=== Reward Trace Toward/Away from Optimal Binning ===")

    # === Step 1: Load correlation arrays ===
    print("Loading precomputed npairs arrays...")
    npz = np.load("Benjamin Work/data/correlations_dl/npairs_compact.npz")
    dd, dr, rd, rr = npz['dd'], npz['dr'], npz['rd'], npz['rr']
    print("Done.")

    # === Step 2: Load node/galaxy info ===
    print("Loading node/galaxy info...")
    node_indices = np.load("Benjamin Work/data/node_indices_fc.npy")
    initial_node_bins = np.load("Benjamin Work/data/nodes_labels_initial_fc_200_ext.npy")
    reference_labels = np.load("Benjamin Work/data/reference_labels_fc_fc.npy")
    print("Done.")

    # === Step 3: Load true redshifts ===
    print("Loading true redshifts...")
    accept = np.load('Benjamin Work/data/accept_fc.npy')
    photometric_sample = np.load('Benjamin Work/data/photometric_sample_fc.npy')
    reference_sample = np.load('Benjamin Work/data/reference_sample_fc.npy')

    true_redshift = pd.read_parquet('Benjamin Work/data/full_catalogue.parquet', columns=['redshift'])
    true_redshift.drop(index=true_redshift.index[-1], axis=0, inplace=True)
    true_redshift.drop(index=np.arange(len(accept))[accept == False], axis=0, inplace=True)
    true_redshift.reset_index(inplace=True, drop=True)

    # Get true redshifts for photometric sample (non-reference galaxies)
    true_redshift_phot = true_redshift.drop(index=reference_sample, inplace=False)
    true_redshift_phot = true_redshift_phot['redshift'].to_numpy()
    print("Done.")

    # === Step 4: Bin the true redshifts into 11 bins ===
    print("Binning true redshifts...")
    def bin_redshifts(z_array, zmin=0.0, zmax=2.2, num_bins=11):
        edges = np.linspace(zmin, zmax, num_bins + 1)
        return np.clip(np.digitize(z_array, bins=edges) - 1, 0, num_bins - 1)

    true_bins_phot = bin_redshifts(true_redshift_phot)
    print("Done.")

    # === Step 5: Assign optimal bin per SOM node ===
    print("Computing optimal bin per node...")
    num_nodes = len(initial_node_bins)
    num_bins = 11
    print("Computing optimal bin per node (fast)...")
    nodes_mean_true = np.zeros(num_nodes)
    nodes_size = np.zeros(num_nodes)
    for i in range(len(node_indices)):
        idx = node_indices[i]
        nodes_mean_true[idx] += true_redshift_phot[i]
        nodes_size[idx] += 1
    nodes_mean_true = np.divide(
        nodes_mean_true, nodes_size, out=np.zeros_like(nodes_mean_true), where=nodes_size > 0
    )
    optimal_node_bins = bin_redshifts(nodes_mean_true, zmin=0.0, zmax=2.2, num_bins=num_bins)
    print("Computed optimal bin per node.")

    # === Step 6: Create SOMControl instance ===
    print("Initializing SOMControl...")
    som = SOMControl(node_indices, reference_labels, initial_node_bins, dd, dr, rd, rr)

    # === Step 7: Record initial reward ===
    print("Calculating initial rewards...")
    init_treecorr = som.total_reward()
    init_proxy = som.proxy_reward()
    print(f"Initial TreeCorr-style reward: {init_treecorr:.4f}")
    print(f"Initial Proxy reward:         {init_proxy:.4f}")

    # === Step 8: Perform improvement and worsening steps ===
    X = 1000  # steps toward optimal
    Y = 1000  # steps away from optimal

    trace_steps = []
    proxy_scores = []
    treecorr_scores = []

    # Function to update bin and record reward
    def take_step(node, new_bin):
        som.set_node_bin(node, new_bin)
        proxy = som.proxy_reward()
        treecorr = som.total_reward()
        return proxy, treecorr

    # === TOWARD OPTIMAL BINNING ===
    print(f"\nTaking {X} steps TOWARD optimal...")
    rng = np.random.default_rng(42)
    for i in range(X):
        node = rng.integers(0, num_nodes)
        new_bin = optimal_node_bins[node]
        if new_bin == -1 or new_bin == som.get_node_bin(node):
            continue
        proxy, treecorr = take_step(node, new_bin)
        trace_steps.append(i - X)  # Negative steps = improving
        proxy_scores.append(proxy)
        treecorr_scores.append(treecorr)

    # === AWAY FROM OPTIMAL BINNING ===
    print(f"\nTaking {Y} steps AWAY from optimal...")
    for i in range(Y):
        node = rng.integers(0, num_nodes)
        current = som.get_node_bin(node)
        optimal = optimal_node_bins[node]
        if optimal == -1:
            continue
        # Choose wrong bin (not optimal)
        candidates = [b for b in range(num_bins) if b != optimal and b != current]
        if not candidates:
            continue
        bad_bin = rng.choice(candidates)
        proxy, treecorr = take_step(node, bad_bin)
        trace_steps.append(i + 1)  # Positive steps = worsening
        proxy_scores.append(proxy)
        treecorr_scores.append(treecorr)

    # Print final rewards after X steps
    print(f"Final TreeCorr-style reward after {X} steps: {treecorr_scores[-1]:.4f}")
    print(f"Final Proxy reward after {X} steps:         {proxy_scores[-1]:.4f}")

    # Normalise scores
    proxy_scores = np.array(proxy_scores)
    treecorr_scores = np.array(treecorr_scores)

# === Step 9: Dual-y plot of reward traces ===
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left y-axis for Proxy Reward
color1 = 'tab:blue'
ax1.set_xlabel("Step (Negative = Toward Optimal, Positive = Away)")
ax1.set_ylabel("Proxy Reward", color=color1)
ax1.plot(trace_steps, proxy_scores, color=color1, label="Proxy Reward", alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color1)

# Right y-axis for TreeCorr-style Reward
ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel("TreeCorr-Style Reward", color=color2)
ax2.plot(trace_steps, treecorr_scores, color=color2, label="TreeCorr-Style Reward", alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color2)

# Add vertical start line
ax1.axvline(x=0, color='gray', linestyle='--', label="Start Point")

# Title and layout
plt.title("Reward Evolution While Moving Toward/Away from Optimal")
fig.tight_layout()
fig.savefig("reward_trace_dual_axis.png", dpi=150)
plt.show()