/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   GuidedRandomForest.cpp
 * Author: aashi & prashant
 *
 * Created on 4 October, 2019, 5:02 PM
 */

#include "GuidedRandomForest.hpp"
#include "HashTree.hpp"
#include "utils.hpp"
#include "HashParams.hpp"
#include "HashProp.hpp"
#include "declaration.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <memory>

GuidedRandomForest::GuidedRandomForest(){
    this->debug = __DEBUG_HASH;
}

GuidedRandomForest::GuidedRandomForest(int L, int M, int n_samples_to_split, int trial_count, bool extend_plane, int seed, int verbose){
    this->L = L;
    this->M = M;
    this->n_samples_to_split = n_samples_to_split;
    this->trial_count = trial_count;
    this->extend_plane = extend_plane;
    this->seed = seed;
    this->verbose = verbose;
    
    this->debug = __DEBUG_HASH;
}

GuidedRandomForest::~GuidedRandomForest(){
    for(int i=0; i<this->L; i++){
        delete this->trees.at(i);
    }
}

void GuidedRandomForest::__setup(){
    
    uint s;
    default_rng rng(this->seed);

    this->trees.resize(this->L);
    
    for(int i=0; i<this->L; i++){
        s = uint_uni_dist()(rng);
        this->trees.at(i) = new HashTree<default_rng>(this->M, this->C, this->n_samples_to_split, this->trial_count, this->extend_plane, 
                this->balance_factor, s, this->verbose);
    }
    
}

void GuidedRandomForest::__processLabels(const int_vec1d& y_orig, int_vec1d& y_processed){
    
    int curr_label;
    int n_labels = 0;

    int_vec1d::iterator itr;

    for(size_t i=0; i<y_orig.size(); i++){
        curr_label = y_orig.at(i);

        itr = std::find(this->label_map.begin(), this->label_map.end(), curr_label);
        if(itr == this->label_map.end()){
            this->label_map.resize(n_labels + 1);
            this->balance_factor.resize(n_labels + 1, 0);
            this->label_map.at(n_labels) = curr_label;
            this->inv_label_map[curr_label] = n_labels;
            n_labels++;
        }

        y_processed.at(i) = this->inv_label_map.at(y_orig.at(i));
        this->balance_factor.at(y_processed.at(i)) += 1;
    }

    this->C = n_labels;
    
    for(int i=0; i<this->C; i++){
        this->balance_factor.at(i) = static_cast<numeric>(y_orig.size())/this->balance_factor.at(i);
    }
    
    if(this->debug){
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Balance Factor " << std::endl;
        printVec<numeric>(this->balance_factor);
    }
    
}

void GuidedRandomForest::fit(const numeric_vec2d& X, const int_vec1d& y){
    
    int_vec1d y_processed(y.size(), 0);
    this->__processLabels(y, y_processed);
    this->__setup();
    for(int i=0; i<this->L; i++){
        if(this->debug){
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Growing tree " << i+1 << std::endl;
        }
        this->trees.at(i)->fit(X, y_processed);
        
        if(this->debug) std::cout << std::endl << std::endl;
        
    }    
}

int_vec1d GuidedRandomForest::predict(const numeric_vec2d& X){
    
    size_t _samples = X.size();
    int_vec1d predicted_labels(_samples);

    
    numeric_vec2d probs(_samples, numeric_vec1d(this->C, 0));
    for(int i=0; i<this->L; i++){
        
        if(this->debug){
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Testing tree " << i+1 << std::endl;
        }
        
        this->trees.at(i)->predictProba(X, probs, true);
    }

    if(this->debug){
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "]" << "Prediction " << std::endl;
        printVec<numeric>(probs);
    }

    for(size_t i=0; i<_samples; i++){
        int lb = std::distance(probs.at(i).begin(), std::max_element(probs.at(i).begin(), probs.at(i).end()));
        predicted_labels.at(i) = this->label_map.at(lb);
    }

    return predicted_labels;
    
}

void GuidedRandomForest::saveTrees(const std::string fname){
}

numeric_vec3d GuidedRandomForest::getWeights(){
}

numeric_vec2d GuidedRandomForest::getBiases(){
}

int_vec2d GuidedRandomForest::getDims(){
}

int_vec1d GuidedRandomForest::getTreeHeights(){
}

int_vec1d GuidedRandomForest::getNodeCount(){
}

int_vec1d GuidedRandomForest::getLeafNodeCount(){
}

int_vec1d GuidedRandomForest::getLabelMap(){
}