#!/bin/bash
START=$(date +%s.%N)

for i in "$@"; do
    case $i in
    --work_dir=*)
        work_dir="${i#*=}"
        shift
        ;;
    --space=*)
        space="${i#*=}"
        shift
        ;;
    --max_elements=*)
        max_elements="${i#*=}"
        shift
        ;;
    --index_path=*)
        index_path="${i#*=}"
        shift
        ;;
    --feat_path=*)
        feat_path="${i#*=}"
        shift
        ;;
    --query_path=*)
        query_path="${i#*=}"
        shift
        ;;
    --truth_path=*)
        truth_path="${i#*=}"
        shift
        ;;
    esac
done

echo "work_dir         = ${work_dir}"
echo "space            = ${space}"
echo "max_elements     = ${max_elements}"
echo "query_path       = ${query_path}"
echo "feat_path        = ${feat_path}"
echo "truth_apth       = ${truth_path}"
echo "index_path       = ${index_path}"
echo "script           = ${work_dir}/python/nnd_bench.py"
echo ""

python3 ${work_dir}/python/nnd_bench.py \
    --space $space \
    --feat_path ${feat_path} \
    --max_elements ${max_elements} \
    --query_path ${query_path} \
    --truth_path ${truth_path} \
    --index_path ${index_path}

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

printf "Job is done in %0.2f secs!" $DIFF
