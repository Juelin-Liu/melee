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
    --M=*)
        M="${i#*=}"
        shift
        ;;
    --feat_path=*)
        feat_path="${i#*=}"
        shift
        ;;
    --index_path=*)
        index_path="${i#*=}"
        shift
        ;;
    --max_elements=*)
        max_elements="${i#*=}"
        shift
        ;;
    esac
done

echo "work_dir         = ${work_dir}"
echo "space            = ${space}"
echo "M                = ${M}"
echo "feat_path        = ${feat_path}"
echo "index_path       = ${index_path}"
echo "script           = ${work_dir}/build/nnd_build"
echo ""

# python3 ${work_dir}/python/nnd_build.py \

${work_dir}/build/nnd_build \
--space $space \
--M $M \
--max_elements ${max_elements} \
--feat_path ${feat_path} \
--index_path ${index_path} \

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

printf "Job is done in %0.2f secs!" $DIFF