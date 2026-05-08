/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   declaration.hpp
 * Author: aashi & prashant
 *
 * Created on 4 October, 2019, 5:04 PM
 */

#ifndef DECLARATION_HPP
#define DECLARATION_HPP

#include <vector>
#include <map>
#include <random>

typedef float numeric;

typedef unsigned int uint;
typedef std::vector<int> int_vec1d;
typedef std::vector< std::vector<int> > int_vec2d;
typedef std::vector<std::vector<std::vector<int>>> int_vec3d;

typedef std::vector<numeric> numeric_vec1d;
typedef std::vector<std::vector<numeric> > numeric_vec2d;
typedef std::vector<std::vector<std::vector<numeric>>> numeric_vec3d;

typedef std::map<int, int> int_int_map;

typedef std::mt19937 default_rng;

typedef std::uniform_real_distribution<numeric> numeric_uni_dist;
typedef std::normal_distribution<numeric> numeric_normal_dist;
typedef std::uniform_int_distribution<uint> uint_uni_dist;


#endif /* DECLARATION_HPP */

