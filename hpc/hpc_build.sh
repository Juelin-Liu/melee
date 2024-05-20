#!/bin/bash

CUR_DIR="$(dirname "$(readlink -f "$0")")"

data_name=bigann
data_file=base.1B.u8bin
space=l2
ef_construction=500

RUNNER=$CUR_DIR/hpc_build_runner.sh
WORK_DIR=$(realpath "$CUR_DIR/../")
DATA_DIR=$WORK_DIR/data/datasets/${data_name}
GRAPH_DIR=$WORK_DIR/data/graphs/${data_name}
LOG_DIR=$WORK_DIR/data/build_logs/${data_name}
feat_path=$DATA_DIR/${data_file}

mkdir -p $GRAPH_DIR
mkdir -p $LOG_DIR

maxtime=8:00:00
partition=defq # 2 days

for M in 16 32 48 64; do
    for max_elements_str in 10M 100M; do
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

        jobname=${data_name}_${max_elements_str}_M${M}_efcon${ef_construction}
        index_path=$GRAPH_DIR/${jobname}.hnsw
        build_log=$LOG_DIR/${jobname}.log

        sbatch --partition=${partition} --time=${maxtime} --job-name=${jobname} --output=${build_log} --exclusive \
            $RUNNER \
            --work_dir=${WORK_DIR} \
            --space=$space \
            --M=$M \
            --ef_construction=${ef_construction} \
            --max_elements=${max_elements} \
            --feat_path=${feat_path} \
            --index_path=${index_path}

    done
done
