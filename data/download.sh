#!/bin/bash
script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
dataset_dir=$script_dir/datasets
mkdir -p $dataset_dir


# Download azcopy
# mkdir -p $script_dir/azcopy && pushd $script_dir/azcopy
# wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1
# mkdir -p $dataset_dir/msspacev && pushd $dataset_dir/msspacev
# azcopy=$script_dir/azcopy/azcopy
# $azcopy copy https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/spacev1b_base.i8bin $dataset_dir/msspacev/base.i8bin       
# $azcopy copy https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/spacev100m_base.i8bin $dataset_dir/msspacev/learn.i8bin 
# $azcopy copy https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/query.i8bin $dataset_dir/msspacev/query.i8bin 
# $azcopy copy https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/public_query_gt100.bin $dataset_dir/msspacev/ground_truth.bin 

# BIGANN/SIFT 1B
mkdir -p $dataset_dir/bigann && pushd $dataset_dir/bigann
aria2c -x 16 -j 16 -c -o base.1B.u8bin       https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
# aria2c -x 16 -j 16 -c -o learn.100M.u8bin      https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/learn.100M.u8bin
# aria2c -x 16 -j 16 -c -o query.10k.u8bin      https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin

# SpaceV
# git clone --recurse-submodules https://github.com/microsoft/SPTAG

# # Yandex Deep
# mkdir -p $dataset_dir/deep && pushd $dataset_dir/deep
# aria2c -x 16 -j 16 -c -o base.1B.fbin        https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
# aria2c -x 16 -j 16 -c -o learn.350M.fbin       https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/learn.350M.fbin
# aria2c -x 16 -j 16 -c -o query.10k.fbin       https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin

# # Yandex Text-to-image
# mkdir -p $dataset_dir/text2image && pushd $dataset_dir/text2image
# aria2c -x 16 -j 16 -c -o base.1B.fbin        https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
# aria2c -x 16 -j 16 -c -o learn.50M.fbin       https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
# aria2c -x 16 -j 16 -c -o query.100K.fbin       https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin

# Ground Truth
# mkdir -p $dataset_dir/gt && pushd $dataset_dir/gt
# aria2c -x 16 -j 16 -c https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M_v2.tgz
# aria2c -x 16 -j 16 -c https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M_v2.tgz
# aria2c -x 16 -j 16 -c https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_1B_v2.tgz

# tar -xvzf GT_10M_v2.tgz
# tar -xvzf GT_100M_v2.tgz
# tar -xvzf GT_1B_v2.tgz
# rm *.tgz