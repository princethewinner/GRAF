/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   matmul_kernel.cuh
 * Author: aashi & prashant
 *
 * Created on 15 September, 2019, 9:06 PM
 */

#ifndef MATMULKERNEL_CUH
#define MATMULKERNEL_CUH

#include <cuda_runtime.h>
#include "Matrix.hpp"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cmath>
#include <type_traits>
//#include <iostream>
//#include <cstdio>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif


template<typename T> __device__ T GetElement(const Matrix<T> A, int row, int col){
    return A.elements[row * A.stride + col];
}

template<typename T> __device__ void SetElement(Matrix<T> A, int row, int col, T value){
    A.elements[row * A.stride + col] = value;
}

template<typename T> __device__ Matrix<T> GetSubMatrix(Matrix<T> A, int row, int col){
    Matrix<T> Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

template<typename T, typename R> __global__ void MatMulKernel(Matrix<T> A, Matrix<T> B, Matrix<R> C, Matrix<T> D){

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix<R> Csub = GetSubMatrix<R>(C, blockRow, blockCol);

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    int Row = blockRow * BLOCK_SIZE + row;
    int Col = blockCol * BLOCK_SIZE + col;

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    T Cvalue = 0;
    T AddValue = 0;

    if(Col < B.width){
        AddValue = GetElement<T>(D, 0, Col);
    }

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for(int m = 0; m < (int)(ceil((float)A.width / BLOCK_SIZE)); ++m){

        // Get sub-matrix Asub of A
        Matrix<T> Asub = GetSubMatrix<T>(A, blockRow, m);

        // Get sub-matrix Asub of B
        Matrix<T> Bsub = GetSubMatrix<T>(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix

        if( m * BLOCK_SIZE + col < A.width && Row < A.height){
            As[row][col] = GetElement<T>(Asub, row, col);
        }
        else{
            As[row][col] = 0.0;
        }

        if( m * BLOCK_SIZE + row < B.height && Col < B.width){
            Bs[row][col] = GetElement<T>(Bsub, row, col);
        }
        else{
            Bs[row][col] = 0.0;
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation

        __syncthreads();

        // Multiply Asub and Bsub together
        if(Row < A.height && Col < B.width){
            for(int e = 0; e < BLOCK_SIZE; ++e){
                Cvalue += As[row][e] * Bs[e][col];
            }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element

    if(Row < A.height && Col < B.width){
        R bit;
        if(std::is_integral<R>::value)
            bit = (Cvalue + AddValue > 0) ? 1 : 0;
        else bit = Cvalue + AddValue;
        SetElement<R>(Csub, row, col, bit);
        //printf("Block id : %d, %d\tThread id %d %d\tCvalue %lf\n", blockCol, blockRow, row, col, Cvalue);
    }
}



#endif /* MATMULKERNEL_CUH */

