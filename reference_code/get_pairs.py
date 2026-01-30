import pickle
import numpy as np

def extract_npairs(path, num_bins, num_nodes):
    with open(path, 'rb') as f:
        corr = pickle.load(f)
    return np.array([[corr[i, j].npairs for j in range(num_nodes)] for i in range(num_bins)])

num_bins = 11
num_nodes = 200 * 200

dd_npairs = extract_npairs('Benjamin Work/data/correlations_dl/dd.pkl', num_bins, num_nodes)
dr_npairs = extract_npairs('Benjamin Work/data/correlations_dl/dr.pkl', num_bins, num_nodes)
rd_npairs = extract_npairs('Benjamin Work/data/correlations_dl/rd.pkl', num_bins, num_nodes)
rr_npairs = extract_npairs('Benjamin Work/data/correlations_dl/rr.pkl', num_bins, num_nodes)

np.savez_compressed('Benjamin Work/data/correlations_dl/npairs_compact.npz',
                    dd=dd_npairs, dr=dr_npairs, rd=rd_npairs, rr=rr_npairs)