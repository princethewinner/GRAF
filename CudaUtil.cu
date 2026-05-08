/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cuda_util.cu
 * Author: aashi & prashant
 *
 * Created on 15 September, 2019, 9:04 PM
 */

#include "CudaUtil.cuh"
#include "MatmulKernel.cuh"
#include "declaration.hpp"
#include "Matrix.hpp"

template Matrix<numeric> toGPU(const Matrix<numeric>);
template Matrix<int> toGPU(const Matrix<int>);

template void freeGPU(const Matrix<numeric>);
template void freeGPU(const Matrix<int>);

template void matmulGPU(const Matrix<numeric>, const Matrix<numeric>, Matrix<int>, const Matrix<numeric>);
template void matmulGPU(const Matrix<numeric>, const Matrix<numeric>, Matrix<numeric>, const Matrix<numeric>);
