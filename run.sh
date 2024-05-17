#!/bin/bash
CUR_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

BIN=$CUR_DIR/build/main
DATADIR=$CUR_DIR/data/datasets/bigann
feat_path=${DATADIR}/learn.100M.u8bin
index_out=${DATADIR}/learn_M16_ef200.index

# Build Index
$BIN \
  --space "l2uint8" \
  --M 16 \
  --ef_construction 200 \
  --ef 100 \
  --num_threads 24 \
  --k 10 \
  --feat_path ${feat_path} \
  --index_out ${index_out}

# # Build Index
# $BIN \
#   --space l2uint8 \
#   --M 6 \
#   --ef_construction 100 \
#   --ef 100 \
#   --num_threads 24 \
#   --k 10 \
#   --feat_path ${feat_path} \
#   --index_path "" \
#   --query_path "" \
#   --truth_path "" \
#   --index_out ${index_out}
# --space "l2u8int" --M 16 --ef_construction 200 --ef 100 --num_threads 24 --k 10 --feat_path /data/juelin/project/melee/data/datasets/bigann/learn.100M.u8bin --index_out /data/juelin/project/melee/data/datasets/bigann/hnsw_index.bin
