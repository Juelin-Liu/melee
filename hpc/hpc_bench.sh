#!/bin/bash

CUR_DIR="$(dirname "$(readlink -f "$0")")"

data_name=deep
query_file=query.10k.fbin
space=l2
ef_construction=500

RUNNER=$CUR_DIR/hpc_bench_runner.sh
WORK_DIR=$(realpath "$CUR_DIR/../")
DATA_DIR=$WORK_DIR/data/datasets/${data_name}
GRAPH_DIR=$WORK_DIR/data/graphs/${data_name}
LOG_DIR=$WORK_DIR/data/bench_logs/${data_name}
query_path=$DATA_DIR/${query_file}
mkdir -p $GRAPH_DIR
mkdir -p $LOG_DIR

maxtime=2:00:00
partition=defq # 2 days

for M in 16 32 48 64; do
    for max_elements_str in 10M; do
        truth_path=${WORK_DIR}/data/datasets/gt/GT_${max_elements_str}/${data_name}-${max_elements_str}
        jobname=${data_name}_${max_elements_str}_M${M}_efcon${ef_construction}
        index_path=$GRAPH_DIR/${jobname}.hnsw
        bench_log=$LOG_DIR/${jobname}.log
            sbatch --partition=${partition} --time=${maxtime} --job-name=${jobname} --output=${bench_log} --exclusive \
                $RUNNER \
                --work_dir=${WORK_DIR} \
                --space=$space \
                --query_path=${query_path} \
                --truth_path=${truth_path} \
                --index_path=${index_path}

    done
done
