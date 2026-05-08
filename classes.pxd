from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

ctypedef float numeric

cdef extern from "GuidedRandomForest.hpp":

    cdef cppclass GuidedRandomForest:

        int L
        int M
        int trial_count
        bool extend_plane
        int n_samples_to_split
        int seed
        int verbose

        GuidedRandomForest() except+
        GuidedRandomForest(int, int, int, int, bool, int, int) except+
        void fit(vector[vector[numeric]]&, vector[int]&)
        vector[int] predict(vector[vector[numeric]]&)
        vector[vector[vector[numeric]]] getWeights()
        vector[vector[numeric]] getBiases()
        vector[vector[int]] getDims()
        vector[int] getTreeHeights()
        vector[int] getNodeCount()
        vector[int] getLeafNodeCount()
        vector[int] getLabelMap()
        void saveTrees(string)
