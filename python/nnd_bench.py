from pynndescent import NNDescent
import argparse
import time
import sys
from util import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HNSW build script')
    parser.add_argument('--space', type=str, help='distance space')
    parser.add_argument('--M', type=int, help='out degree upper bound')
    parser.add_argument('--max_elements', type=int, help='num elements to build index')
    parser.add_argument('--feat_path', type=str, help='feature path')
    parser.add_argument('--index_path', type=str, help='index save path')
    parser.add_argument('--query_path', type=str, help='path to query')
    parser.add_argument('--truth_path', type=str, help='path to ground truth')
    args = parser.parse_args()
    print(f"{args=}")
    
    t = time.time()
    feat = load_data(args.feat_path, args.max_elements, as_float=True)
    query = load_data(args.query_path, sys.maxsize)
    labels, distance = load_ground_truth(args.truth_path)
    nnd_graph = np.load(args.index_path).reshape(-1, args.M)
        
    num_elements = feat.shape[0]
    dim = feat.shape[1]
    ids = np.arange(num_elements)
    num_query = query.shape[0]
    
    print(f"load feat in {time.time() - t} secs")

    metric = None
    if args.space == "l2":
        metric = "euclidean"
    elif args.space == "cosine" or "ip":
        metric = "cosine"
        
    t = time.time()
    index = NNDescent(data=feat, metric=metric, init_graph=nnd_graph, n_neighbors=nnd_graph.shape[1], low_memory=False)
    index.prepare()

    print("START BENCHMARK")
    for k in [1, 10, 100]:
        for epsilon in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            t = time.time()
            plabels, pdistance = index.query(query_data = query, k = k, epsilon=epsilon)
            total_matched = 0
            for i in range(num_query):
                gt_label = labels[i][:k]
                p_label = plabels[i]
                for pred in p_label:
                    if pred in gt_label:
                        total_matched += 1
            recall = total_matched / (num_query * k)
            print(f"{k=} {epsilon=} {recall=}", flush=True)
    print("END BENCHMARK")