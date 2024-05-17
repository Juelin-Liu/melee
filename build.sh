#!/bin/bash
CUR_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

mkdir -p ${CUR_DIR}/build

cmake -B ${CUR_DIR}/build -DCMAKE_BUILD_TYPE=Release

cmake --build ${CUR_DIR}/build -j
