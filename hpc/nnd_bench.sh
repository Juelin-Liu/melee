#!/bin/bash

CUR_DIR="$(dirname "$(readlink -f "$0")")"

data_name=bigann
data_file=base.1B.u8bin
query_file=query.10k.u8bin
space=l2
RUNNER=$CUR_DIR/nnd_bench_runner.sh
WORK_DIR=$(realpath "$CUR_DIR/../")
DATA_DIR=$WORK_DIR/data/datasets/${data_name}
GRAPH_DIR=$WORK_DIR/data/nnd/${data_name}
LOG_DIR=$WORK_DIR/data/nnd/bench_logs/${data_name}
feat_path=$DATA_DIR/${data_file}
query_path=$DATA_DIR/${query_file}

mkdir -p $GRAPH_DIR
mkdir -p $LOG_DIR

for M in 16; do
    for max_elements_str in 10M; do
        if [ "$max_elements_str" = "1M" ]; then
            max_elements=1000000
        elif [ "$max_elements_str" = "10M" ]; then
            max_elements=10000000
        elif [ "$max_elements_str" = "100M" ]; then
            max_elements=100000000
        else
            echo "max_elements_str is not equal to 1M, 10M, or 100M."
            exit 1
        fi
        truth_path=${WORK_DIR}/data/datasets/gt/GT_${max_elements_str}/${data_name}-${max_elements_str}

        jobname=${data_name}_${max_elements_str}_M${M}
        index_path=$GRAPH_DIR/${jobname}.npy
        build_log=$LOG_DIR/${jobname}.log
        $RUNNER \
        --work_dir=${WORK_DIR} \
        --space=$space \
        --max_elements=${max_elements} \
        --feat_path=${feat_path} \
        --query_path=${query_path} \
        --truth_path=${truth_path} \
        --index_path=${index_path} > ${build_log}
    done
done
