/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   HashTree.hpp
 * Author: aashi & prashant
 *
 * Created on 4 October, 2019, 5:17 PM
 */

#ifndef HASHTREE_HPP
#define HASHTREE_HPP

#include "GenericHash.hpp"
#include "Node.hpp"
#include "utils.hpp"
#include "CudaMatMul.hpp"
#include <vector>
#include <cmath>
#include <limits>

#define EPS 1e-10

template<typename RNG>
class HashTree : public GenericHash{
    
    public: int M, n_samples_to_split, trial_count, seed, verbose;
    public: bool extend_plane;
    
    private: bool debug;
    private: int C, n_samples, n_dims, most_impure_region_ind;
    //private: int_vec1d dims, left_child, right_child, impure_regions;
    private: int_vec1d dims, impure_regions;
    private: std::vector<Region*> regions;
    private: numeric_vec2d weights;
    private: numeric_vec1d biases, balance_factor;
    private: numeric max_impurity;
    private: CudaMatmul<numeric, int> cuda_matmul;
    
    
    public: HashTree();
    public: HashTree(int, int, int, int, bool, numeric_vec1d, int, int);
                                    //Order M, C, n_samples_to_split, trial_count, extend_plane, balance_factor, seed, verbose
    public: ~HashTree();
    
    public: void fit(const numeric_vec2d&, const int_vec1d&)                                    override;
    public: void predictProba(const numeric_vec2d&, numeric_vec2d&, bool)                       override;
    public: void predictLabel(const numeric_vec2d&, int_vec1d&, bool, int_vec2d&)               override;
    public: std::unique_ptr<GenericParams> getParams()                                          override;
    public: std::unique_ptr<GenericProp> getProperties()                                        override;
    public: void dumpModel(std::string, int, ...)                                               override;
    
    
    private: void __hash(const numeric_vec2d&, const int_vec1d&, RNG&);
    private: void __splitRegionGlobally(const numeric_vec2d&, const int_vec1d&, Region*, RNG&);
    private: void __chooseBestPlane(const numeric_vec2d&, const int_vec1d&, const numeric_vec2d&, const numeric_vec1d&);
    //private: void __fillRegions(Region*, int, const numeric_vec2d&, const int_vec1d&);
    private: void __fillRegions(Region*, Region*, const numeric_vec2d&, const int_vec1d&);
    private: void __createRegions(PartView&, int_vec1d&, const numeric_vec2d&, const int_vec1d&);
    private: numeric __computeGiniImpurity(const numeric_vec1d&, int);
    private: numeric __computeGiniImpurity(Region *, const int_vec1d&);
    
};


template<typename RNG> HashTree<RNG>::HashTree(){
    this->C = 0;
    this->debug = __DEBUG_TREE;
}

template<typename RNG> HashTree<RNG>::HashTree(int M, int C, int n_samples_to_split, int trial_count, bool extend_plane, numeric_vec1d balance_factor, int seed, int verbose){
    this->M = M;
    this->C = C;
    this->n_samples_to_split = n_samples_to_split;
    this->trial_count = trial_count;
    this->extend_plane = extend_plane;
    this->balance_factor = balance_factor;
    this->seed = seed;
    this->verbose = verbose;
    
    this->max_impurity = 0;
    this->debug = __DEBUG_TREE;
}

template<typename RNG> HashTree<RNG>::~HashTree(){
    
    for(size_t i=0; i<this->regions.size(); i++){
        delete this->regions.at(i);
    }
    
}

template<typename RNG> numeric HashTree<RNG>::__computeGiniImpurity(const numeric_vec1d& region_content, int samples){

    numeric metric, sum = 0;
    for(int i=0; i<this->C; i++){
        sum += (region_content.at(i) * region_content.at(i));
    }

    metric = (1 - sum) * samples;

    return metric;

}

template<typename RNG> void HashTree<RNG>::__fillRegions(Region* region, Region* parent_region, const numeric_vec2d& X, const int_vec1d& y){
    
    //Region * parent_region = this->regions.at(parent_ind);
    
    if(region->impurity == 0){
        parent_region->node.height = this->weights.size() - 1;
        region->node.label = std::distance(region->node.probs.begin(), std::max_element(region->node.probs.begin(), region->node.probs.end()));
        region->node.leaf = true;
        
        if(this->debug){
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Probabilities as leaf node ";
            printVec<numeric>(region->node.probs);
            std::cout << "For samples ";
            printVec<int>(region->region_samples);
            
        }
        
        return;
    }
    
    int ns = region->region_samples.size();
    
    for(int i=0; i<this->M; i++){
        if(parent_region->fs.at(i)==0){
            
            numeric min_v = std::numeric_limits<numeric>::max();
            numeric max_v = -1 * std::numeric_limits<numeric>::max();
            numeric mean_v = 0;
            
            region->fs.push_back(0);
            for(int j=0; j<ns; j++){
                
                int ind = region->region_samples.at(j);
                numeric v = X.at(ind).at(this->dims.at(i));
                
                if(v < min_v) min_v = v;
                if(v > max_v) max_v = v;
                mean_v += v;
            }
            region->min.push_back(min_v);
            region->max.push_back(max_v);
            region->mean.push_back(mean_v/ns);
            
        }
        else{
            region->fs.push_back(1);
            region->min.push_back(parent_region->min.at(i));
            region->max.push_back(parent_region->max.at(i));
            region->mean.push_back(parent_region->mean.at(i));
        }
    }
    
    int tracker = 0;
    numeric roi_from_min = 0, roi_from_max = 0;
    for(int i=0; i<this->M; i++){
        if(region->fs.at(i)){
            tracker++;
        }
        else{
            if(std::fabs(region->min.at(i) - region->max.at(i)) < EPS){
                tracker++;
                region->fs.at(i) = 1;
            }
        }
        
        numeric v1 = region->min.at(i) - region->mean.at(i);
        numeric v2 = region->max.at(i) - region->mean.at(i);
        
        roi_from_min += (v1 * v1);
        roi_from_max += (v2 * v2);
        
    }
    
    if(roi_from_min > roi_from_max) region->roi = std::sqrt(roi_from_min);
    else region->roi = std::sqrt(roi_from_max);
    
    if(tracker == this->M) region->impurity = 0;
    
    parent_region->node.height = this->weights.size() - 1;
    
    if(region->impurity == 0 || ns < this->n_samples_to_split){
        region->node.label = std::distance(region->node.probs.begin(), std::max_element(region->node.probs.begin(), region->node.probs.end()));
        region->node.leaf = true;
        
        if(this->debug){
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Probabilities as leaf node ";
            printVec<numeric>(region->node.probs);
            std::cout << "For samples ";
            printVec<int>(region->region_samples);
        }
        
    }
    
}

template<typename RNG> void HashTree<RNG>::__createRegions(PartView& part_view, int_vec1d& intact_regions, const numeric_vec2d& X, const int_vec1d& y){
    
    int new_regions = part_view.indexes_0.size();
    
    for(int i=0; i<new_regions; i++){
        
        Region * parent_region = this->regions.at(part_view.parent_ind.at(i));
        Region * region_0 = part_view.indexes_0.at(i), * region_1 = part_view.indexes_1.at(i);
        
        parent_region->left = region_0;
        parent_region->right = region_1;
        
        this->__fillRegions(region_0, parent_region, X, y);
        this->__fillRegions(region_1, parent_region, X, y);

        this->regions.push_back(region_0);
        this->regions.push_back(region_1);

        int r0_ind = this->regions.size() - 2;
        int r1_ind = this->regions.size() - 1;

        //this->left_child.at(part_view.parent_ind.at(i)) = r0_ind;
        //this->right_child.at(part_view.parent_ind.at(i)) = r1_ind;
        

        if(this->max_impurity < region_0->impurity){
            this->max_impurity = region_0->impurity;
            this->most_impure_region_ind = r0_ind;
        }

        if(this->max_impurity < region_1->impurity){
            this->max_impurity = region_1->impurity;
            this->most_impure_region_ind = r1_ind;
        }

        if(!region_0->node.leaf){
            intact_regions.push_back(r0_ind);
        }
        if(!region_1->node.leaf){
            intact_regions.push_back(r1_ind);
        }
        
    }
    
    //this->left_child.resize(this->regions.size(), -1);
    //this->right_child.resize(this->regions.size(), -1);
    
    this->impure_regions = intact_regions;
    
}

template<typename RNG> numeric HashTree<RNG>::__computeGiniImpurity(Region * region, const int_vec1d& y){
    
    int ns = region->region_samples.size();
    region->node.probs.resize(this->C, 0);
    for(int i=0; i<ns; i++){
        int ind = region->region_samples.at(i);
        region->node.probs.at(y.at(ind)) += 1;
    }
    
    numeric balance = 0;
    for(int i=0; i<this->C; i++){
        region->node.probs.at(i) /= ns;
        region->node.probs.at(i) *= this->balance_factor.at(i);
        balance += region->node.probs.at(i);
    }
    
    for(int i=0; i<this->C; i++){
        region->node.probs.at(i) /= balance;
    }
    
    region->impurity = this->__computeGiniImpurity(region->node.probs, ns);
    return region->impurity;
    
}

template<typename RNG> void HashTree<RNG>::__chooseBestPlane(const numeric_vec2d& X, const int_vec1d& y, const numeric_vec2d& ws, const numeric_vec1d& bs){
    
    numeric_vec1d impurities(this->trial_count, 0);
    numeric_vec1d max_impurities(this->trial_count, 0);
    int_vec1d _most_impure_region_ind(this->trial_count, -1);
    int_vec2d intact_regions(this->trial_count, int_vec1d());
    std::vector<PartView> splitted_regions(this->trial_count);
    
    int * bits = this->cuda_matmul.matmul(ws, bs);
    
    for(int i=0; i<this->trial_count; i++){
        
        if(this->debug){
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Running trial " << i+1 << std::endl;
        }
        
        const numeric_vec1d& w = ws.at(i);
        numeric b = bs.at(i);
        
        splitted_regions.at(i).reserve(this->impure_regions.size());
        //int splitted_region_count = -1;
        
        for(size_t j=0; j<this->impure_regions.size(); j++){
            Region* region = this->regions.at(this->impure_regions.at(j));
            
            numeric dist = 0;
            numeric norm = 0;
            for(int k=0; k<this->M; k++){
                dist += (region->mean.at(k) * w.at(k));
                norm += (w.at(k) * w.at(k));
            }

            dist += b;
            dist = std::fabs(dist)/std::sqrt(norm);
            
            if(this->debug){
                std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Region roi " << region->roi << std::endl;
                std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "plane to mean dist " << dist << std::endl;
            }
            
            if(region->roi > dist){
                
                //splitted_region_count++;
                //splitted_regions.at(i).indexes_0.push_back(new Region());
                //splitted_regions.at(i).indexes_1.push_back(new Region());
                //splitted_regions.at(i).parent_ind.push_back(this->impure_regions.at(j));
                
                Region* r_0 = new Region();
                Region* r_1 = new Region();
                
                for(size_t k=0; k<region->region_samples.size(); k++){
                    int ind = region->region_samples.at(k);
                    
                    //numeric dot = 0;
                    //for(int l=0; l<this->M; l++){
                    //    dot += (X.at(ind).at(this->dims.at(l)) * w.at(l));
                    //}
                    //dot += b;
                    int dot = bits[ind * this->trial_count + i];
                    
                    if(dot == 0){
                        //splitted_regions.at(i).indexes_0.at(splitted_region_count).push_back(ind);
                        //splitted_regions.at(i).indexes_0.at(splitted_region_count)->region_samples.push_back(ind);
                        r_0->region_samples.push_back(ind);
                    }
                    else{
                        //splitted_regions.at(i).indexes_1.at(splitted_region_count).push_back(ind);
                        //splitted_regions.at(i).indexes_1.at(splitted_region_count)->region_samples.push_back(ind);
                        r_1->region_samples.push_back(ind);
                    }
                    
                }
                
                //if(splitted_regions.at(i).indexes_0.at(splitted_region_count)->region_samples.size() > 0 && splitted_regions.at(i).indexes_1.at(splitted_region_count)->region_samples.size() > 0){
                //    splitted_regions.at(i).splitted.push_back(1);
                //    impurities.at(i) += this->__computeGiniImpurity(splitted_regions.at(i).indexes_0.at(splitted_region_count), y);
                //    impurities.at(i) += this->__computeGiniImpurity(splitted_regions.at(i).indexes_1.at(splitted_region_count), y);
                //}
                
                if(this->debug){
                    std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Splitted indexes with 0" << std::endl;
                    printVec<int>(r_0->region_samples);
                    std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Splitted indexes with 1" << std::endl;
                    printVec<int>(r_1->region_samples);
                }
                
                if(r_0->region_samples.size() > 0 && r_1->region_samples.size() > 0){
                    //splitted_regions.at(i).splitted.push_back(1);
                    impurities.at(i) += this->__computeGiniImpurity(r_0, y);
                    impurities.at(i) += this->__computeGiniImpurity(r_1, y);
                    splitted_regions.at(i).indexes_0.push_back(r_0);
                    splitted_regions.at(i).indexes_1.push_back(r_1);
                    splitted_regions.at(i).parent_ind.push_back(this->impure_regions.at(j));
                    
                }
                else{
                    delete r_0;
                    delete r_1;
                    //splitted_regions.at(i).splitted.push_back(0);
                    intact_regions.at(i).push_back(this->impure_regions.at(j));
                    impurities.at(i) += region->impurity;
                    
                    if(max_impurities.at(i) < region->impurity){
                        max_impurities.at(i) = region->impurity;
                        _most_impure_region_ind.at(i) = this->impure_regions.at(j);
                    }
                    
                }
            }
            else{
                intact_regions.at(i).push_back(this->impure_regions.at(j));
                impurities.at(i) += region->impurity;
                
                if(max_impurities.at(i) < region->impurity){
                    max_impurities.at(i) = region->impurity;
                    _most_impure_region_ind.at(i) = this->impure_regions.at(j);
                }
                
            }
            
        }
        
    }
    
    if(this->debug){
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Computed Impurities " << std::endl;
        printVec<numeric>(impurities);
    }
    
    int best_plane_ind = std::distance(impurities.begin(), std::min_element(impurities.begin(), impurities.end()));
    
    if(this->debug){
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Selected weights " << std::endl;
        printVec<numeric>(ws.at(best_plane_ind));
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Selected bias " << std::endl;
        std::cout << bs.at(best_plane_ind) << std::endl;
    }
    
    this->weights.push_back(ws.at(best_plane_ind));
    this->biases.push_back(bs.at(best_plane_ind));
    
    this->max_impurity = max_impurities.at(best_plane_ind);
    this->most_impure_region_ind = _most_impure_region_ind.at(best_plane_ind);
    this->__createRegions(splitted_regions.at(best_plane_ind), intact_regions.at(best_plane_ind), X, y);
    
    for(int i=0; i<this->trial_count; i++){
        if(i != best_plane_ind){
            for(size_t j=0; j<splitted_regions.at(i).indexes_0.size(); j++){
                delete splitted_regions.at(i).indexes_0.at(j);
                delete splitted_regions.at(i).indexes_1.at(j);
            }
        }
    }
    
    delete[] bits;
    
}

template<typename RNG> void HashTree<RNG>::__splitRegionGlobally(const numeric_vec2d& X, const int_vec1d& y, Region* region, RNG& rng){
    
    numeric_vec2d ws(this->trial_count, numeric_vec1d(this->M));
    numeric_vec1d bs(this->trial_count, 0);
    
    for(int i=0; i<this->trial_count; i++){
        for(int j=0; j<this->M; j++){
            if(region->fs.at(j) == 0) ws.at(i).at(j) = numeric_uni_dist(region->min.at(j), region->max.at(j))(rng);
            else ws.at(i).at(j) = region->min.at(j);
        }
        
        utils::normalize<numeric>(ws.at(i));
        
        for(int j=0; j<this->M; j++){
            bs.at(i) -= (region->mean.at(j) * ws.at(i).at(j));
        }
        
    }
    
    if(this->debug){
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Generated weights " << std::endl;
        printVec<numeric>(ws);
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Generated biases " << std::endl;
        printVec<numeric>(bs);
    }
    
    this->__chooseBestPlane(X, y, ws, bs);
    
}

template<typename RNG> void HashTree<RNG>::__hash(const numeric_vec2d& X, const int_vec1d& y, RNG& rng){
    
    Region * root = new Region();
    root->reserve(this->M, this->n_samples, this->C, true);
    root->region_samples.resize(this->n_samples);
    
    for(int i=0; i<this->n_samples; i++){
        root->region_samples.at(i) = i;
        for(int j=0; j<this->M; j++){
            numeric v = X.at(i).at(this->dims.at(j));
            if(root->min.at(j) > v) root->min.at(j) = v;
            if(root->max.at(j) < v) root->max.at(j) = v;
            root->mean.at(j) += v;
        }
        root->node.probs.at(y.at(i)) += 1;
    }
    
    numeric roi_from_min = 0, roi_from_max = 0;
    
    int tracker = 0;
    for(int i=0; i<this->M; i++){
        
        root->mean.at(i) /= this->n_samples;
        
        if(std::fabs(root->min.at(i) - root->max.at(i)) < EPS){
            root->fs.push_back(1);
            tracker++;
        }
        else{
            root->fs.push_back(0);
        
            numeric v1 = root->min.at(i) - root->mean.at(i);
            numeric v2 = root->max.at(i) - root->mean.at(i);
            roi_from_min += (v1 * v1);
            roi_from_max += (v2 * v2);
        }
    }
    
    if(roi_from_min > roi_from_max) root->roi = std::sqrt(roi_from_min);
    else root->roi = std::sqrt(roi_from_max);
    
    numeric balance = 0;
    for(int i=0; i<this->C; i++){
        root->node.probs.at(i) /= this->n_samples;
        root->node.probs.at(i) *= this->balance_factor.at(i);
        balance += root->node.probs.at(i);
    }
    
    for(int i=0; i<this->C; i++) root->node.probs.at(i) /= balance;
    root->impurity = this->__computeGiniImpurity(root->node.probs, this->n_samples);
    root->node.height = 0;
    
    this->most_impure_region_ind = 0;
    this->max_impurity = 0;

    if(tracker == this->M){
        root->node.leaf = true;
        root->node.label = std::distance(root->node.probs.begin(), std::max_element(root->node.probs.begin(), root->node.probs.end()));
    }

    
    this->regions.push_back(root);
    this->impure_regions.push_back(0);
    
    //this->left_child.push_back(-1);
    //this->right_child.push_back(-1);
    
    if(tracker == this->M) return;
    
    if(this->debug){
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Root partition description " << std::endl;
        std::cout << "min values ";
        printVec<numeric>(root->min);
        std::cout << "max values ";
        printVec<numeric>(root->max);
        std::cout << "mean values ";
        printVec<numeric>(root->mean);
        std::cout << "probs ";
        printVec<numeric>(root->node.probs);
        std::cout << "region indexes ";
        printVec<int>(root->region_samples);
        std::cout << "impurities " << root->impurity << std::endl;
        std::cout << "roi " << root->roi << std::endl;
        std::cout << "height " << root->node.height << std::endl;
        std::cout << std::endl;
    }
    
    //int c=0;
    while(this->impure_regions.size() > 0){
        //c++;
        if(this->debug){
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Total impure regions " << this->impure_regions.size() << std::endl;
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Most impure region samples " << std::endl;
            printVec<int>(this->regions.at(this->most_impure_region_ind)->region_samples);
        }
        
        this->max_impurity = 0;
        this->__splitRegionGlobally(X, y, this->regions.at(this->most_impure_region_ind), rng);
        
        //if(c > 4)break;
    }
    
}

template<typename RNG> void HashTree<RNG>::fit(const numeric_vec2d& X, const int_vec1d& y){
    
    RNG rng(this->seed);
    
    this->n_samples = X.size();
    this->n_dims = X.at(0).size();

    int possible_node_count = 2 * this->n_samples;
    int possible_internal_node_count = static_cast<int>(this->n_samples/2);
    
    //this->left_child.reserve(possible_node_count);
    //this->right_child.reserve(possible_node_count);

    this->impure_regions.reserve(this->n_samples);
    this->regions.reserve(possible_node_count);
    
    this->weights.reserve(possible_internal_node_count);
    this->biases.reserve(possible_internal_node_count);
    
    int_vec1d dim_vector(this->n_dims);
    for(int i=0; i<this->n_dims; i++) dim_vector.at(i) = i;
    this->dims = utils::randomSample<int, RNG>(dim_vector, this->M, rng);

    if(this->debug){
        std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Selected dims ";
        printVec<int>(this->dims);
    }
    
    this->cuda_matmul.init(X, this->dims);
    this->__hash(X, y, rng);
    this->cuda_matmul.clear();
    
    //printVec<int>(this->left_child);
    //printVec<int>(this->right_child);
}

template<typename RNG> void HashTree<RNG>::predictProba(const numeric_vec2d& Xtest, numeric_vec2d& probs, bool accumulate){
    
    int w_ind;
    size_t _smp = Xtest.size(), n_weights = this->weights.size();
    Region * region;
    
    this->cuda_matmul.init(Xtest, this->dims);
    int * bits = this->cuda_matmul.matmul(this->weights, this->biases);
    this->cuda_matmul.clear();
    
    for(size_t i=0; i<_smp; i++){
        
        region = this->regions.at(0);
        
        //std::cout << "**" << std::endl;
        if(this->debug){
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Testing Sample " << i+1 << std::endl;
        }
        
        while(!region->node.leaf){
            w_ind = region->node.height;
            
            //std::cout << w_ind << std::endl;
            
            if(this->debug){
                std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Using weights ";
                printVec<numeric>(this->weights.at(w_ind));
                std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Using bias " << this->biases.at(w_ind) << std::endl;
            }
            
            //numeric dot = 0;
            //for(int j=0; j<this->M; j++){
            //    dot += (this->weights.at(w_ind).at(j)*Xtest.at(i).at(this->dims.at(j)));
            //}
            //dot += this->biases.at(w_ind);
            
            int dot = bits[i*n_weights + w_ind];
            
            if(dot == 0) region = region->left;
            else region = region->right;
            
            //region = this->regions.at(temp);
        }
        
        if(this->debug){
            std::cout << "[DEBUG][" << __LINE__ << ":" << __FILE__ << "]:[" << __FUNCTION__ << "] " << "Found probabilities ";
            printVec<numeric>(region->node.probs);
        }
        
        if(accumulate){
            for(int j=0; j<this->C; j++){
                probs.at(i).at(j) += region->node.probs.at(j);
            }
        }
        else{
            for(int j=0; j<this->C; j++){
                probs.at(i).at(j) = region->node.probs.at(j);
            }
        }
    }
    
    delete[] bits;
}

template<typename RNG> void HashTree<RNG>::predictLabel(const numeric_vec2d& a, int_vec1d& b, bool c, int_vec2d& d){
}

template<typename RNG> std::unique_ptr<GenericParams> HashTree<RNG>::getParams(){
}

template<typename RNG> std::unique_ptr<GenericProp> HashTree<RNG>::getProperties(){
}

template<typename RNG> void HashTree<RNG>::dumpModel(std::string a, int b, ...){
}

#endif /* HASHTREE_HPP */

