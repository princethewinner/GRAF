/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   cuda_util.cuh
 * Author: aashi & prashant
 *
 * Created on 15 September, 2019, 9:05 PM
 */

#ifndef CUDAUTIL_CUH
#define CUDAUTIL_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cmath>
#include "Matrix.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

template<typename T> Matrix<T> toGPU(const Matrix<T> mat){

    Matrix<T> d_mat;

    d_mat.height = mat.height;
    d_mat.width = mat.width;
    d_mat.stride = mat.width;

    int size = mat.height * mat.width * sizeof(T);

    checkCudaErrors(cudaMalloc(&d_mat.elements, size));
    checkCudaErrors(cudaMemcpy(d_mat.elements, mat.elements, size, cudaMemcpyHostToDevice));

    return d_mat;

}


template<typename T> void freeGPU(const Matrix<T> d_mat){
    cudaFree(d_mat.elements);
}


template<typename T, typename D> void matmulGPU(const Matrix<T> d_A_mat, const Matrix<T> B_mat, Matrix<D> C_mat, const Matrix<T> D_mat){

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((int)(ceil((float)B_mat.width/dimBlock.x)), (int)(ceil((float)d_A_mat.height/dimBlock.y)));

    Matrix<T> d_B_mat = toGPU<T>(B_mat);
    Matrix<T> d_D_mat = toGPU<T>(D_mat);

    Matrix<D> d_C_mat;
    d_C_mat.height = C_mat.height;
    d_C_mat.width = C_mat.width;
    d_C_mat.stride = C_mat.width;

    int size = C_mat.height * C_mat.width * sizeof(D);
    checkCudaErrors(cudaMalloc(&d_C_mat.elements, size));

    MatMulKernel<<< dimGrid, dimBlock >>>(d_A_mat, d_B_mat, d_C_mat, d_D_mat);

    checkCudaErrors(cudaMemcpy(C_mat.elements, d_C_mat.elements, size, cudaMemcpyDeviceToHost));

    cudaFree(d_C_mat.elements);
    freeGPU<T>(d_B_mat);
    freeGPU<T>(d_D_mat);

}


#endif /* CUDAUTIL_CUH */

