import argparse
import time
from util import *
import sys
import faiss

# space = 'l2'
# feat_path = "/data/juelin/project/melee/data/datasets/bigann/base.1B.u8bin"
# query_path = '/data/juelin/project/melee/data/datasets/bigann/query.10k.u8bin'
# truth_path = '/data/juelin/project/melee/data/datasets/gt/GT_10M/bigann-10M'
# index_path = '/data/juelin/project/melee/data/nnd/bigann/bigann_10M_M16.npy'
# max_elements= 10000000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HNSW build script')
    parser.add_argument('--space', type=str, help='distance space')
    # parser.add_argument('--ef_construction', type=int, help='ef construction')
    parser.add_argument('--M', type=int, help='out degree upper bound')
    parser.add_argument('--max_elements', type=int, help='num elements to build index')
    parser.add_argument('--feat_path', type=str, help='feature path')
    parser.add_argument('--index_path', type=str, help='index save path')
    args = parser.parse_args()

    t = time.time()

    feat = load_data(args.feat_path, args.max_elements, as_float=True)
    query = load_data(args.query_path, sys.maxsize)
    labels, distance = load_ground_truth(args.truth_path)

    nnd_graph = np.load(args.index_path)

    print(f"load feat in {time.time() - t} secs")

    num_elements = feat.shape[0]
    dim = feat.shape[1]
    ids = np.arange(num_elements)
    num_query = query.shape[0]

    t = time.time()
    metric = None
    if space == "l2":
        metric = faiss.METRIC_L2
    elif space == "cosine" or "ip":
        metric = faiss.METRIC_Lp

    index = faiss.IndexNNDescentFlat(feat.shape[1], 32, metric)
    index.add(feat)
    print(f"train feat in {time.time() - t} secs")

    print("START BENCHMARK")
    for k in [1, 10, 100]:
        for search_L in [1, 10, 20, 40, 80, 160, 320, 640, 960]:
            if search_L < k:
                continue
            index.nndescent.search_L = search_L
            t = time.time()
            pdistance, plabels = index.search(x = query, k = k)
            search_time = time.time() - t
            qps = int(num_query / search_time)
            
            total_matched = 0
            for i in range(num_query):
                gt_label = labels[i][:k]
                p_label = plabels[i]
                for pred in p_label:
                    if pred in gt_label:
                        total_matched += 1
            recall = total_matched / (num_query * k)
            print(f"{k=} {recall=} {search_L=} {qps=}", flush=True)
    print("END BENCHMARK")