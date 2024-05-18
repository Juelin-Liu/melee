#!/bin/bash
START=$(date +%s.%N)

for i in "$@"; do
    case $i in
    --space=*)
        space="${i#*=}"
        shift
        ;;
    --work_dir=*)
        work_dir="${i#*=}"
        shift
        ;;
    --k=*)
        k="${i#*=}"
        shift
        ;;
    --num_threads=*)
        num_threads="${i#*=}"
        shift
        ;;
    --M=*)
        M="${i#*=}"
        shift
        ;;
    --ef_construction=*)
        ef_construction="${i#*=}"
        shift
        ;;
    --ef=*)
        ef="${i#*=}"
        shift
        ;;
    --index_out=*)
        index_out="${i#*=}"
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
    --query_path=*)
        query_path="${i#*=}"
        shift
        ;;
    --truth_path=*)
        truth_path="${i#*=}"
        shift
        ;;
    --max_elements=*)
        max_elements="${i#*=}"
        shift
        ;;
    -* | --*)
        echo "Unknown option $i"
        exit 1
        ;;
    *) ;;
    esac
done

echo "work_dir         = ${work_dir}"
echo "space            = ${space}"
echo "M                = ${M}"
echo "k                = ${k}"
echo "ef_construction  = ${ef_construction}"
echo "ef               = ${ef}"
echo "num_threads      = ${num_threads}"
echo "feat_path        = ${feat_path}"
echo "index_out        = ${index_out}"
echo "index_path       = ${index_path}"
echo "query_path       = ${query_path}"
echo "truth_path       = ${truth_path}"
echo "binary           = ${work_dir}/build/main"
echo ""

${work_dir}/build/main \
--space $space \
--M $M \
--k $k \
--ef_construction ${ef_construction} \
--ef ${ef} \
--max_elements ${max_elements} \
--num_threads ${num_threads} \
--feat_path ${feat_path} \
--index_out ${index_out} \
--index_path ${index_path} \
--query_path ${query_path} \
--truth_path ${truth_path}


END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

printf "Job is done in %0.2f secs!" $DIFF