import hnswlib
import argparse
import time
from util import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HNSW build script')
    parser.add_argument('--space', type=str, help='distance space')
    parser.add_argument('--ef_construction', type=int, help='ef construction')
    parser.add_argument('--M', type=int, help='out degree upper bound')
    parser.add_argument('--max_elements', type=int, help='num elements to build index')
    parser.add_argument('--feat_path', type=str, help='feature path')
    parser.add_argument('--index_path', type=str, help='index save path')
    args = parser.parse_args()
    
    print(f"{args=}")
    
    t = time.time()
    feat = load_data(args.feat_path, args.max_elements)
    print(f"load feat in {time.time() - t} secs")
    
    num_elements = feat.shape[0]
    dim = feat.shape[1]
    ids = np.arange(num_elements)
    
    t = time.time()
    p = hnswlib.Index(space = args.space, dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = args.ef_construction, M = args.M)
    p.add_items(feat, ids)
    
    print(f"build index in {time.time() - t} secs")
    p.save_index(args.index_path)
    print("saving index to ", args.index_path)