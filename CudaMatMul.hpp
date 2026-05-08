/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cuda_matmul.hpp
 * Author: aashi & prashant
 *
 * Created on 15 September, 2019, 9:02 PM
 */

#ifndef CUDAMATMUL_HPP
#define CUDAMATMUL_HPP


#include <vector>
#include <iostream>
#include "Matrix.hpp"
#include <memory>
#include "utils.hpp"

//@TODO: Subspacing compatibility

template<typename T> Matrix<T> toGPU(const Matrix<T>);
template<typename T> void freeGPU(const Matrix<T>);
template<typename T, typename D> void matmulGPU(const Matrix<T>, const Matrix<T>, Matrix<D>, const Matrix<T>);

template<typename T, typename D>
class CudaMatmul{

    private: Matrix<T> A_matrix;
    private: Matrix<T> d_A_matrix;
    private: int block_size;
    private: bool cleared;
    
    public: CudaMatmul();
    public: CudaMatmul(const std::vector<std::vector<T>>&, const std::vector<int>&);
    public: void init(const std::vector<std::vector<T>>&, const std::vector<int>&);
    public: void clear();
    public: ~CudaMatmul();

    public: D* matmul(const std::vector<std::vector<T>>&, const std::vector<T>&);

};

template<typename T, typename D>
CudaMatmul<T, D>::CudaMatmul(){}

template<typename T, typename D>
CudaMatmul<T, D>::CudaMatmul(const std::vector<std::vector<T>>& vec, const std::vector<int>& subspace_ind){
    convertToRowMajor<T>(vec, this->A_matrix, subspace_ind);
    this->d_A_matrix = toGPU<T>(this->A_matrix);
    this->cleared = false;
}

template<typename T, typename D>
CudaMatmul<T, D>::~CudaMatmul(){
    if(!this->cleared){
        delete[] this->A_matrix.elements;
        freeGPU<T>(this->d_A_matrix);
    }
}

template<typename T, typename D>
void CudaMatmul<T, D>::init(const std::vector<std::vector<T>>& vec, const std::vector<int>& subspace_ind){
    convertToRowMajor<T>(vec, this->A_matrix, subspace_ind);
    this->d_A_matrix = toGPU<T>(this->A_matrix);    
    this->cleared = false;
}

template<typename T, typename D>
void CudaMatmul<T, D>::clear(){
    if(!this->cleared){
        delete[] this->A_matrix.elements;
        freeGPU<T>(this->d_A_matrix);        
        this->cleared = true;
    }
}

template<typename T, typename D>
D* CudaMatmul<T, D>::matmul(const std::vector<std::vector<T>>& vec, const std::vector<T>& add_vec){

    Matrix<T> B_matrix;
    Matrix<D> C_matrix;
    Matrix<T> D_matrix;

    convertToColMajor<T>(vec, B_matrix);
    convertToRowMajor<T>(add_vec, D_matrix);

    C_matrix.height = this->A_matrix.height;
    C_matrix.width = B_matrix.width;
    C_matrix.stride = 1;
    C_matrix.elements = new D[C_matrix.height * C_matrix.width];

    matmulGPU<T, D>(this->d_A_matrix, B_matrix, C_matrix, D_matrix);

    delete[] B_matrix.elements;
    delete[] D_matrix.elements;

    return C_matrix.elements;
}


#endif /* CUDAMATMUL_HPP */

