/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   GuidedRandomForest.hpp
 * Author: aashi & prashant
 *
 * Created on 4 October, 2019, 5:02 PM
 */

#ifndef GUIDEDRANDOMFOREST_HPP
#define GUIDEDRANDOMFOREST_HPP

#include "GenericHash.hpp"
#include "declaration.hpp"

#include <vector>
#include <map>

class GuidedRandomForest{
    
    public: int L;
    public: int M;
    public: int trial_count;    
    public: bool extend_plane;
    public: int n_samples_to_split;
    public: int seed;
    public: int verbose;

    private: bool debug;
    private: int C;
    private: int_vec1d label_map;
    private: int_int_map inv_label_map;
    private: numeric_vec1d balance_factor;
    private: std::vector<GenericHash *> trees;

    
    public: GuidedRandomForest();
    public: GuidedRandomForest(int L, int M, int n_samples_to_split, int trial_count, bool extend_plane, int seed, int verbose);
    public: ~GuidedRandomForest();
    public: void fit(const numeric_vec2d& X, const int_vec1d& y);
    public: int_vec1d predict(const numeric_vec2d& X);
    public: void saveTrees(const std::string fname);
    public: numeric_vec3d getWeights();
    public: numeric_vec2d getBiases();
    public: int_vec2d getDims();
    public: int_vec1d getTreeHeights();
    public: int_vec1d getNodeCount();
    public: int_vec1d getLeafNodeCount();
    public: int_vec1d getLabelMap();
    
    private: void __setup();
    private: void __processLabels(const int_vec1d& y_orig, int_vec1d& y_processed);

    
};

#endif /* GUIDEDRANDOMFOREST_HPP */

