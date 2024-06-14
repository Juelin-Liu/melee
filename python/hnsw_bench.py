import hnswlib
import argparse
import time
import sys
from util import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HNSW benchmark script')
    parser.add_argument('--space', type=str, help="distance space")
    parser.add_argument('--index_path', type=str, help='index path')
    parser.add_argument('--query_path', type=str, help='query path')
    parser.add_argument('--truth_path', type=str, help='truth path')
    args = parser.parse_args()    
    t = time.time()
    query = load_data(args.query_path, sys.maxsize)
    num_query = query.shape[0]
    dim = query.shape[1]    
    labels, distance = load_ground_truth(args.truth_path)
    p = hnswlib.Index(space = args.space, dim = dim)
    p.load_index(args.index_path)
    # print(f"load data in {time.time() - t} secs")
    print("START BENCHMARK")
    for k in [1, 10, 100]:
        for ef in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
            if (ef < k):
                continue
            p.set_ef(ef)
            t = time.time()
            plabels, pdistance = p.knn_query(query, k = k)
            total_matched = 0
            for i in range(num_query):
                gt_label = labels[i][:k]
                p_label = plabels[i]
                for pred in p_label:
                    if pred in gt_label:
                        total_matched += 1
            recall = total_matched / (num_query * k)
            print(f"{k=} {ef=} {recall=}", flush=True)
    print("END BENCHMARK")
