#!/bin/bash
CUR_DIR="$(dirname "$(readlink -f "$0")")"
WORK_DIR=$(realpath $CUR_DIR/..)

RUNNER=$WORK_DIR/build/dist_bench
LOG_PATH=$WORK_DIR/logs/dist_bench.log
mkdir -p $WORK_DIR/logs

N=100000

echo "START BENCHMARK" > ${LOG_PATH}
for T in 1 12 24; do
    for K in 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144; do
        numactl --cpunodebind=0 --membind=0 ${RUNNER} --K ${K} --T ${T} --N ${N} >> ${LOG_PATH}
    done

    for K in 96 192 384 768 1536 3072 6144 12288 24576 49152 98304 196608 393216; do
        numactl --cpunodebind=0 --membind=0 ${RUNNER} --K ${K} --T ${T} --N ${N} >> ${LOG_PATH}
    done
done
echo "END BENCHMARK" >> ${LOG_PATH}
