#!/bin/bash

CUR_DIR="$(dirname "$(readlink -f "$0")")"
RUNNER=$CUR_DIR/hpc_runner.sh
WORK_DIR=$(realpath "$CUR_DIR/../")

DATA_DIR=$WORK_DIR/data/datasets
GRAPH_DIR=$WORK_DIR/data/graphs
LOG_DIR=$WORK_DIR/data/bench

data_name=deep
data_file=base.1B.fbin
query_file=query.10k.fbin
space=l2

mkdir -p $GRAPH_DIR
mkdir -p $LOG_DIR

num_threads=56
feat_path=$DATA_DIR/${data_name}/${data_file}
query_path=$DATA_DIR/${data_name}/${query_file}
maxtime=1-18:00:00
partition=longq # 21 days
# partition=defq # 12 hours
k=10

for M in 16; do
    for ef_construction in 500; do
        for max_elements_str in 10M; do
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
            truth_path=$WORK_DIR/data/datasets/gt/GT_${max_elements_str}/${data_name}-${max_elements_str}
            
            for ef in 100 200 300; do
                build_job=${data_name}_${max_elements_str}_M${M}_ef${ef_construction}
                bench_job=${build_job}_ef${ef}_k${k}

                index_path=$GRAPH_DIR/${build_job}.index
                bench_log=$LOG_DIR/${bench_job}.log

                sbatch --partition=${partition} --time=${maxtime} --job-name=${bench_job} --output=${bench_log} --exclusive \
                    $RUNNER \
                    --work_dir=${WORK_DIR} \
                    --space=$space \
                    --M=$M \
                    --k=$k \
                    --ef_construction=${ef_construction} \
                    --ef=${ef} \
                    --max_elements=${max_elements} \
                    --num_threads=${num_threads} \
                    --feat_path="" \
                    --index_out="" \
                    --index_path="$index_path" \
                    --query_path="$query_path" \
                    --truth_path="$truth_path"
            done

        done
    done
done
