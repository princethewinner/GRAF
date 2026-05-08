/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   matrix.hpp
 * Author: aashi & prashant
 *
 * Created on 15 September, 2019, 9:03 PM
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

template<typename T>
struct Matrix {
    int width;
    int height;
    int stride;
    T* elements;
};

#endif /* MATRIX_HPP */

