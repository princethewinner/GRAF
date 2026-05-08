/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   GenericHash.hpp
 * Author: aashi & prashant
 *
 * Created on 4 October, 2019, 5:10 PM
 */

#ifndef GENERICHASH_HPP
#define GENERICHASH_HPP

#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include "GenericParams.hpp"
#include "GenericProp.hpp"
#include "declaration.hpp"

class GenericHash{
    
    public: GenericHash(){};
    public: virtual ~GenericHash() = default;
    public: virtual void fit(const numeric_vec2d&, const int_vec1d&) = 0;
    public: virtual void predictProba(const numeric_vec2d&, numeric_vec2d&, bool) = 0;
    public: virtual void predictLabel(const numeric_vec2d&, int_vec1d&, bool, int_vec2d&) = 0;
    public: virtual std::unique_ptr<GenericParams> getParams() = 0;
    public: virtual std::unique_ptr<GenericProp> getProperties() = 0;
    public: virtual void dumpModel(std::string, int, ...) = 0;
    
};


#endif /* GENERICHASH_HPP */

