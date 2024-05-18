#!/bin/bash

CUR_DIR="$(dirname "$(readlink -f "$0")")"
RUNNER=$CUR_DIR/hpc_runner.sh
WORK_DIR=$(realpath "$CUR_DIR/../")

DATA_DIR=$WORK_DIR/data/datasets
GRAPH_DIR=$WORK_DIR/data/graphs
LOG_DIR=$WORK_DIR/data/logs

data_name=deep
data_file=base.1B.u8bin
query_file=query.10k.u8bin

mkdir -p $GRAPH_DIR
mkdir -p $LOG_DIR

space=l2

num_threads=56
feat_path=$DATA_DIR/${data_name}/${data_file}
query_path=$DATA_DIR/${data_name}/${query_file}
maxtime=1-18:00:00
partition=longq # 21 days
# partition=defq # 12 hours
k=10

for M in 16 32; do
    for ef_construction in 100 500; do
        for max_elements_str in 10M 100M; do
            # Check if max_elements_str is equal to 1M, 10M, or 100M
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
            

            jobname=${data_name}_${max_elements_str}_M${M}_ef${ef_construction}
            index_out=$GRAPH_DIR/${jobname}.index
            build_log=$LOG_DIR/${jobname}.log
            truth_path=$WORK_DIR/data/gt/GT_${max_elements_str}/${data_name}-${max_elements_str}

            sbatch --partition=${partition} --time=${maxtime} --job-name=${jobname} --output=${build_log} --exclusive \
                $RUNNER \
                --work_dir=${WORK_DIR} \
                --space=$space \
                --M=$M \
                --k=$k \
                --ef_construction=${ef_construction} \
                --ef=${ef_construction} \
                --max_elements=${max_elements} \
                --num_threads=${num_threads} \
                --feat_path=${feat_path} \
                --index_out=${index_out} \
                --index_path="" \
                --query_path="" \
                --truth_path=""
        done
    done
done
