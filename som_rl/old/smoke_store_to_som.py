import numpy as np

from som_rl.datastore.correlation_store import CorrelationStore, CorrelationPaths
from som_rl.old.som_state import SOMState


def main():
    paths = CorrelationPaths(
        dd="data/correlations_dl/dd.pkl",
        dr="data/correlations_dl/dr.pkl",
        rd="data/correlations_dl/rd.pkl",
        rr="data/correlations_dl/rr.pkl",
        node_indices_photo="data/processed/node_indices_fc.npy",
        initial_node_bins="data/processed/nodes_labels_initial_fc_200_ext.npy"
    )

    store = CorrelationStore(paths, memmap=True)
    data = store.load()

    som = SOMState(
        node_indices_photo=data["node_indices_photo"],
        initial_node_bins=data["initial_node_bins"],
        dd=data["dd"], dr=data["dr"], rd=data["rd"], rr=data["rr"],
        num_bins=11,
        grid_shape=None,               # inferred if index_to_xy not passed
        seed=0,
    )

    r0 = som.total_reward_fast()
    print("initial reward:", r0)

    # Single move sanity check (choose a node and a new bin)
    node = 0
    old_bin = int(som.current_bins[node])
    new_bin = (old_bin + 1) % som.num_bins

    # Use your fast delta estimator if present
    if hasattr(som, "estimate_delta_trace"):
        delta = float(som.estimate_delta_trace(node, new_bin))
        print("estimated delta:", delta)

    # Apply move (use whatever API your SOM.py exposes)
    if hasattr(som, "set_node_bin_idx"):
        som.set_node_bin_idx(node, new_bin)
    else:
        # fallback: grid-based setter if that’s what exists
        x, y = som.index_to_xy(node)
        som.set_node_bin(x, y, new_bin)

    r1 = som.total_reward_fast()
    print("after move reward:", r1, "delta:", (r1 - r0))


if __name__ == "__main__":
    main()
