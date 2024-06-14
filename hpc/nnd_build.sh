#!/bin/bash

CUR_DIR="$(dirname "$(readlink -f "$0")")"

data_name=bigann
data_file=base.1B.u8bin
space=l2

RUNNER=$CUR_DIR/nnd_build_runner.sh
WORK_DIR=$(realpath "$CUR_DIR/../")
DATA_DIR=$WORK_DIR/data/datasets/${data_name}
GRAPH_DIR=$WORK_DIR/data/nnd/${data_name}
LOG_DIR=$WORK_DIR/data/nnd/build_logs/${data_name}
feat_path=$DATA_DIR/${data_file}

mkdir -p $GRAPH_DIR
mkdir -p $LOG_DIR

for M in 16 32 48 64; do
    for max_elements_str in 10M 100M; do
        if [ "$max_elements_str" = "1M" ]; then
            max_elements=1000000
        elif [ "$max_elements_str" = "10M" ]; then
            max_elements=10000000
        elif [ "$max_elements_str" = "100M" ]; then
            max_elements=100000000
        elif [ "$max_elements_str" = "1B" ]; then
            max_elements=1000000000
        else
            echo "max_elements_str is not equal to 1M, 10M, or 100M."
            exit 1
        fi

        jobname=${data_name}_${max_elements_str}_M${M}
        index_path=$GRAPH_DIR/${jobname}.npy
        build_log=$LOG_DIR/${jobname}.log
        
        $RUNNER \
            --work_dir=${WORK_DIR} \
            --space=$space \
            --M=$M \
            --max_elements=${max_elements} \
            --feat_path=${feat_path} \
            --index_path=${index_path} > ${build_log}
    done
done
