import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

K = 11
H = W = 200
N = H * W

def bin_redshifts(z_array, zmin=0.0, zmax=2.2, num_bins=11):
    edges = np.linspace(zmin, zmax, num_bins + 1, dtype=np.float64)
    return np.clip(np.digitize(z_array, bins=edges) - 1, 0, num_bins - 1).astype(np.int32)

def get_perfect_bins():
    # Load needed arrays (paths as per your dataset)
    accept = np.load("data/accept_fc.npy")
    reference_sample = np.load("data/reference_sample_fc.npy")
    node_indices_photo = np.load("data/node_indices_fc.npy").astype(np.int64)   # photometric galaxies -> node id

    # True redshifts (filter accept, remove reference)
    true_redshift = pd.read_parquet("data/full_catalogue.parquet", columns=["redshift"])
    true_redshift.drop(index=true_redshift.index[-1], axis=0, inplace=True)
    true_redshift.drop(index=np.arange(len(accept))[accept == False], axis=0, inplace=True)
    true_redshift.reset_index(inplace=True, drop=True)

    true_redshift_phot = true_redshift.drop(index=reference_sample, inplace=False)["redshift"].to_numpy()

    # Per-node mean true-z
    nodes_sum = np.zeros(N, dtype=np.float64)
    nodes_cnt = np.zeros(N, dtype=np.float64)

    for i in range(node_indices_photo.size):
        n = int(node_indices_photo[i])
        nodes_sum[n] += float(true_redshift_phot[i])
        nodes_cnt[n] += 1.0

    nodes_mean_true = np.divide(nodes_sum, nodes_cnt, out=np.zeros_like(nodes_sum), where=nodes_cnt > 0)

    perfect = bin_redshifts(nodes_mean_true, zmin=0.0, zmax=2.2, num_bins=K)
    return perfect


def to_grid(bins_1d, H=200, W=200):
    return np.asarray(bins_1d, dtype=np.int32).reshape(H, W)

def plot_bins_triptych(start_bins, end_bins, perfect_bins, H=200, W=200, K=11, title=""):
    start = to_grid(start_bins, H, W)
    end   = to_grid(end_bins, H, W)
    perf  = to_grid(perfect_bins, H, W)

    # diagnostics
    mismatch_end = (end != perf).astype(np.int32)
    mismatch_start = (start != perf).astype(np.int32)

    fig, axs = plt.subplots(2, 3, figsize=(14, 9))

    axs[0,0].imshow(start, vmin=0, vmax=K-1)
    axs[0,0].set_title("Start bins")

    axs[0,1].imshow(end, vmin=0, vmax=K-1)
    axs[0,1].set_title("End bins")

    axs[0,2].imshow(perf, vmin=0, vmax=K-1)
    axs[0,2].set_title("Perfect bins (diagnostic)")

    axs[1,0].imshow(mismatch_start, vmin=0, vmax=1)
    axs[1,0].set_title("Start != perfect")

    axs[1,1].imshow(mismatch_end, vmin=0, vmax=1)
    axs[1,1].set_title("End != perfect")

    axs[1,2].imshow(mismatch_end - mismatch_start, vmin=-1, vmax=1)
    axs[1,2].set_title("Mismatch change (end-start)")

    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    # scalar summaries
    start_match = float((start == perf).mean())
    end_match   = float((end == perf).mean())
    print(f"Match to perfect: start={start_match:.3f}, end={end_match:.3f}, Δ={end_match-start_match:+.3f}")
