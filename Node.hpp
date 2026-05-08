/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Node.hpp
 * Author: aashi & prashant
 *
 * Created on 4 October, 2019, 9:41 PM
 */

#ifndef NODE_HPP
#define NODE_HPP

#include "declaration.hpp"
#include <limits>

class Node{
    
    public: numeric_vec1d probs;
    public: int label, height;
    public: bool leaf;
    
    public: Node(){leaf=false;}
    
};

class Region{
    
    public: numeric_vec1d min, max, mean;
    public: numeric roi, impurity;
    public: int_vec1d region_samples, fs;
    public: Node node;
    public: Region * left, * right;
    
    public: Region(){this->left=NULL; this->right=NULL;}
    public: void reserve(int M, int n_samples, int c, bool overwrite){
        
        this->min.resize(M, std::numeric_limits<numeric>::max());
        this->max.resize(M, -1 * std::numeric_limits<numeric>::max());
        this->mean.resize(M, 0);
        this->fs.resize(M, 0);
        this->node.probs.resize(c, 0);

        if(overwrite) this->region_samples.reserve(n_samples);
        else this->region_samples.reserve(512);
        
    }
};

class PartView{
    
    public: std::vector<Region*> indexes_0;
    public: std::vector<Region*> indexes_1;
    public: int_vec1d parent_ind;
    //public: int_vec1d splitted;
    
    public: PartView(){}
    public: void reserve(int n){
        this->indexes_0.reserve(n);
        this->indexes_1.reserve(n);
        this->parent_ind.reserve(n);
        //this->splitted.reserve(n);
    }
    
};

#endif /* NODE_HPP */

