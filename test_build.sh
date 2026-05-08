#!/bin/bash

BLOCK_SIZE=${1}

echo "nvcc -std=c++11 -Xcompiler="-fPIC" -I/usr/local/cuda-10.0/samples/common/inc/ -DBLOCK_SIZE=${BLOCK_SIZE} CudaUtil.cu -c"
nvcc -std=c++11 -Xcompiler="-fPIC" -I/usr/local/cuda-10.0/samples/common/inc/ -DBLOCK_SIZE=${BLOCK_SIZE} CudaUtil.cu -c

rm GRAFPlaneSearchGPUROI.cpython-37m-x86_64-linux-gnu.so
rm GRAF.cpp
python setup.py build_ext --inplace
L=3
M=2
FNAME=test/tree
python test_build.py ${L} ${M} ${FNAME}

