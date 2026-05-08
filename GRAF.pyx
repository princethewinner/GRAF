# distutils: language = c++
# distutils: sources = GuidedRandomForest.cpp

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cimport classes

ctypedef float numeric

cdef class GRAF:

    cdef classes.GuidedRandomForest * tbhc

    def __cinit__(self, int L, int M, int n_samples_to_split=2, int trial_count=0, bool extend_plane=True, int seed=0, int verbose=0):

        if trial_count == 0: trial_count=M

        self.tbhc = new classes.GuidedRandomForest(L, M, n_samples_to_split, trial_count, extend_plane, seed, verbose)

    def fit(self, vector[vector[numeric]]& X, vector[int]& y):
        self.tbhc.fit(X, y)

    def predict(self, vector[vector[numeric]]& X):
        return self.tbhc.predict(X)

    def getWeights(self):
        return self.tbhc.getWeights()

    def getBiases(self):
        return self.tbhc.getBiases()

    def getDims(self):
        return self.tbhc.getDims()

    def getTreeHeights(self):
        return self.tbhc.getTreeHeights()

    def getNodeCount(self):
        return self.tbhc.getNodeCount()

    def getLeafNodeCount(self):
        return self.tbhc.getLeafNodeCount()

    def saveTrees(self, string fname):
        self.tbhc.saveTrees(fname)

    def __dealloc__(self):

        if self.tbhc is not NULL:
            del self.tbhc

    @property
    def label_map(self):
        return self.tbhc.getLabelMap()
