/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   utils.hpp
 * Author: aashi & prashant
 *
 * Created on 4 October, 2019, 5:15 PM
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include "declaration.hpp"
#include "Matrix.hpp"
#include <vector>
#include <iostream>
#include <limits>
#include <type_traits>
#include <typeinfo>
#include <random>
#include <algorithm>
#include <iomanip>

template<typename T>
void printVec(const std::vector<T>& vec){

    for(size_t i=0; i<vec.size(); i++){
        std::cout << std::setprecision(8) << vec.at(i) << " ";
    }
    std::cout << std::endl;

}

template<typename T>
void printVec(const std::vector<std::vector<T>>& vec){

    for(size_t i=0; i<vec.size(); i++){
        printVec<T>(vec.at(i));
    }

}

template<typename T>
void printMatrix(const Matrix<T>& mat){

    int n_elements = mat.height * mat.width;
    for(int i=0; i<n_elements; i++){
        std::cout << mat.elements[i] << " ";
    }
    std::cout << std::endl;

}

template<typename T>
void convertToRowMajor(const std::vector<std::vector<T>>& vec, Matrix<T>& mat){

    //Assuming it as matrix
    size_t n_rows = vec.size();
    size_t n_cols = vec.at(0).size();

    size_t n_elements = n_rows * n_cols;

    mat.height = n_rows;
    mat.width = n_cols;
    mat.stride = 1;
    mat.elements = new T[n_elements];

    size_t count = 0;

    for(size_t i=0; i<n_rows; i++){
        for(size_t j=0; j<n_cols; j++){
            mat.elements[count] = vec.at(i).at(j);
            count++;
        }
    }

}

template<typename T>
void convertToRowMajor(const std::vector<std::vector<T>>& vec, Matrix<T>& mat, const std::vector<int>& subspace_ind){

    //Assuming it as matrix
    size_t n_rows = vec.size();
    size_t n_cols = subspace_ind.size();

    size_t n_elements = n_rows * n_cols;

    mat.height = n_rows;
    mat.width = n_cols;
    mat.stride = 1;
    mat.elements = new T[n_elements];

    size_t count = 0;

    for(size_t i=0; i<n_rows; i++){
        for(size_t j=0; j<n_cols; j++){
            mat.elements[count] = vec.at(i).at(subspace_ind.at(j));
            count++;
        }
    }

}

template<typename T>
void convertToColMajor(const std::vector<std::vector<T>>& vec, Matrix<T>& mat){


    size_t n_rows = vec.size();
    size_t n_cols = vec.at(0).size();

    size_t n_elements = n_rows * n_cols;

    mat.height = n_cols;
    mat.width = n_rows;
    mat.stride = 1;
    mat.elements = new T[n_elements];

    size_t count = 0;

    for(size_t i=0; i<n_cols; i++){
        for(size_t j=0; j<n_rows; j++){
            mat.elements[count] = vec.at(j).at(i);
            count++;
        }
    }

}

template<typename T>
void convertToRowMajor(const std::vector<T>& vec, Matrix<T>& mat){

    size_t n_rows = 1;
    size_t n_cols = vec.size();

    size_t n_elements = n_rows * n_cols;

    mat.height = n_rows;
    mat.width = n_cols;
    mat.stride = 1;
    mat.elements = new T[n_elements];

    for(size_t i=0; i<n_cols; i++){
        mat.elements[i] = vec.at(i);
    }

}

namespace utils{
    
    template<typename T>
    T max(T a, T b){return (a > b)? a : b;}
    
    template<typename T, typename RNG>
    std::vector<T> randomSample(std::vector<T> vec, int n, RNG rng){
        
        std::vector<T> shuffled_vec(n, 0);
        std::shuffle(vec.begin(), vec.end(), rng);
        
        for(int i=0; i<n; i++)shuffled_vec.at(i) = vec.at(i);
        
        return shuffled_vec;
    }
    
    template<typename T>
    void normalize(std::vector<T>& vec){
        //static_assert(std::is_floating_point<T>::value, "Required floating point");
        
        size_t size = vec.size();
        T sq_sum = 0;
        T sum_sqrt = 0;
        
        for(size_t i=0; i < size; i++){
            sq_sum += (vec.at(i) * vec.at(i));
        }
        
        sum_sqrt = std::sqrt(sq_sum);
        
        for(size_t i=0; i<size; i++){
            vec.at(i) = vec.at(i)/sum_sqrt;
        }
        
    }
    
    template<typename T, typename D>
    D* matmul(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B, const std::vector<T>& C, const int_vec1d& subspace){
        
        size_t A_row = A.size(), A_col = subspace.size();
        size_t B_row = B.size();
        
        D* multiplied_matrix = new D[A_row * B_row];
        
        // Assuming A_col == B_col and B_row == C_count
        
        T temp = 0;
        for(size_t i=0; i<A_row; i++){
            for(size_t k=0; k<B_row; k++){
                temp = 0;
                for(size_t j=0; j<A_col; j++){
                    temp += (A.at(i).at(subspace.at(j)) * B.at(k).at(j));
                }
                
                if(std::is_floating_point<D>::value){
                    multiplied_matrix[i*B_row + k] = temp + C.at(k);
                }
                else{
                    multiplied_matrix[i*B_row + k] = (temp + C.at(k) > 0)? 1 : 0;
                }
            }
        }
        return multiplied_matrix;
    }
    
    
}



#endif /* UTILS_HPP */

