/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: aashi & prashant
 *
 * Created on 4 October, 2019, 4:42 PM
 */

#include <cstdlib>

#include "declaration.hpp"
#include "GuidedRandomForest.hpp"
#include "utils.hpp"

/*
 * 
 */
int main( int argc, char** argv ) {

    int L = 3;
    int M = 2;
    int n_samples_to_split = 2;
    int trial_count = M;
    bool extend_plane = true;
    int seed = 0;
    int verbose = 0;
    
    int train_samples = 10;
    int test_samples = 5;
    int n_features = 3;
    numeric_vec2d train_data(train_samples, numeric_vec1d(n_features));
    int_vec1d train_labels(train_samples);
    numeric_vec2d test_data(test_samples, numeric_vec1d(n_features));
    int_vec1d test_labels(test_samples);
    
    //Intializing train data
    int count = 0;
    for(int i=0; i<train_samples; i++){
        for(int j=0; j<n_features; j++){
            train_data.at(i).at(j) = count;
            count++;
        }
    }
    for(int i=0; i<train_samples; i++){
        if(i%2==0) train_labels.at(i) = 0;
        if(i%2==1) train_labels.at(i) = 1 ;
    }
    
    //Printing train data
    //std::cout << std::endl;
    //std::cout << "Train data" << std::endl;
    //printVec<numeric>(train_data);
    
    //std::cout << std::endl;
    //std::cout << "Train label" << std::endl;
    //printVec<int>(train_labels);
    
    //Initializing test data
    for(int i=0; i<test_samples; i++){
        for(int j=0; j<n_features; j++){
            test_data.at(i).at(j) = count;
            count++;
        }
    }
    
    for(int i=0; i<test_samples; i++){
        if(i%2==0) test_labels.at(i) = 0;
        if(i%2==1) test_labels.at(i) = 1;
    }
    
    //Printing test data
    //std::cout << std::endl;
    //std::cout << "Test data" << std::endl;
    //printVec<numeric>(test_data);
    
    //std::cout << std::endl;
    //std::cout << "Test label" << std::endl;
    //printVec<int>(test_labels);
    
    //GuidedRandomForest graf(L, M, n_samples_to_split, trial_count, extend_plane, seed, verbose);
    //graf.fit(train_data, train_labels);
    //int_vec1d train_predictions =  graf.predict(train_data);
    //int_vec1d test_predictions = graf.predict(test_data);
    
    //std::cout << std::endl;
    //std::cout << "Train prediction" << std::endl;
    //printVec<int>(train_predictions);
    
    //std::cout << std::endl;
    //std::cout << "Test prediction" << std::endl;
    //printVec<int>(test_predictions);
    
    //[[0.5488135  0.71518937 0.60276338]
    //[0.54488318 0.4236548  0.64589411]
    //[0.43758721 0.891773   0.96366276]
    //[0.38344152 0.79172504 0.52889492]
    //[0.56804456 0.92559664 0.07103606]
    //[0.0871293  0.0202184  0.83261985]
    //[0.77815675 0.87001215 0.97861834]
    //[0.79915856 0.46147936 0.78052918]
    //[0.11827443 0.63992102 0.14335329]
    //[0.94466892 0.52184832 0.41466194]]
    
    numeric_vec2d train_data2;
    numeric_vec1d temp1 = {0.5488135,  0.71518937, 0.60276338};
    numeric_vec1d temp2 = {0.54488318, 0.4236548,  0.64589411};
    numeric_vec1d temp3 = {0.43758721, 0.891773,   0.96366276};
    numeric_vec1d temp4 = {0.38344152, 0.79172504, 0.52889492};
    numeric_vec1d temp5 = {0.56804456, 0.92559664, 0.07103606};
    numeric_vec1d temp6 = {0.0871293,  0.0202184,  0.83261985};
    numeric_vec1d temp7 = {0.77815675, 0.87001215, 0.97861834};
    numeric_vec1d temp8 = {0.79915856, 0.46147936, 0.78052918};
    numeric_vec1d temp9 = {0.11827443, 0.63992102, 0.14335329};
    numeric_vec1d temp10 = {0.94466892, 0.52184832, 0.41466194};
    
    train_data2.push_back(temp1);
    train_data2.push_back(temp2);
    train_data2.push_back(temp3);
    train_data2.push_back(temp4);
    train_data2.push_back(temp5);
    train_data2.push_back(temp6);
    train_data2.push_back(temp7);
    train_data2.push_back(temp8);
    train_data2.push_back(temp9);
    train_data2.push_back(temp10);
    
    int_vec1d train_labels2 = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    
    numeric_vec2d test_data2;
    numeric_vec1d test_temp1 = {0.26455561, 0.77423369, 0.45615033};
    numeric_vec1d test_temp2 = {0.56843395, 0.0187898, 0.6176355};
    numeric_vec1d test_temp3 = {0.61209572, 0.616934, 0.94374808};
    numeric_vec1d test_temp4 = {0.6818203, 0.3595079, 0.43703195};
    numeric_vec1d test_temp5 = {0.6976312, 0.06022547, 0.66676672};
    
    test_data2.push_back(test_temp1);
    test_data2.push_back(test_temp2);
    test_data2.push_back(test_temp3);
    test_data2.push_back(test_temp4);
    test_data2.push_back(test_temp5);
    
    int_vec1d test_labels2 = {0, 0, 1, 1, 1};
    
    //Printing second data
    std::cout << "Train data" << std::endl;
    printVec<numeric>(train_data2);
    std::cout << std::endl;
    
    std::cout << "Train label" << std::endl;
    printVec<int>(train_labels2);
    std::cout << std::endl;
    
    std::cout << "Test data" << std::endl;
    printVec<numeric>(test_data2);
    std::cout << std::endl;
    
    std::cout << "Test label" << std::endl;
    printVec<int>(test_labels2);
    std::cout << std::endl;
    
    GuidedRandomForest graf2(3, 2, n_samples_to_split, 2, extend_plane, seed, verbose);
    graf2.fit(train_data2, train_labels2);
    
    std::cout << std::endl << std::endl << std::endl;
    int_vec1d train_predictions =  graf2.predict(train_data2);
    std::cout << "Train prediction" << std::endl;
    printVec<int>(train_predictions);

    std::cout << std::endl << std::endl << std::endl;
    int_vec1d test_predictions = graf2.predict(test_data2);
    std::cout << "Test predictions" << std::endl;
    printVec<int>(test_predictions);
    
    return 0;
}

