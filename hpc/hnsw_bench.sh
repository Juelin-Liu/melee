#!/bin/bash

CUR_DIR="$(dirname "$(readlink -f "$0")")"

data_name=bigann
query_file=query.10k.u8bin
space=l2
ef_construction=500

RUNNER=$CUR_DIR/hnsw_bench_runner.sh
WORK_DIR=$(realpath "$CUR_DIR/../")
DATA_DIR=$WORK_DIR/data/datasets/${data_name}
GRAPH_DIR=$WORK_DIR/data/hnsw/${data_name}
LOG_DIR=$WORK_DIR/data/hnsw/bench_logs/${data_name}
query_path=$DATA_DIR/${query_file}
mkdir -p $GRAPH_DIR
mkdir -p $LOG_DIR


for M in 16 32 48 64; do
    for max_elements_str in 10M 100M; do
        truth_path=${WORK_DIR}/data/datasets/gt/GT_${max_elements_str}/${data_name}-${max_elements_str}
        jobname=${data_name}_${max_elements_str}_M${M}_efcon${ef_construction}
        index_path=$GRAPH_DIR/${jobname}.bin
        bench_log=$LOG_DIR/${jobname}.log
        $RUNNER \
        --work_dir=${WORK_DIR} \
        --space=$space \
        --query_path=${query_path} \
        --truth_path=${truth_path} \
        --index_path=${index_path} > $bench_log

    done
done
