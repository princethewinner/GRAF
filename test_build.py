import GRAFPlaneSearchGPUROI as GRAF
import numpy as np
import sys

np.random.seed(0)
data = np.random.random((10, 3))

labels = np.array([0]*5 + [1]*5, dtype=np.int)

test_data = np.random.random((5, 3))
test_labels = np.array([0]*2 + [1]*3, dtype=np.int)

#data = np.r_[data, data[:2]]
#labels = np.concatenate((labels, labels[:2]))

print('\ndata')
print(data)

print('\nlabels')
print(labels)

L = int(sys.argv[1])
M = int(sys.argv[2])
fname = sys.argv[3]

model = GRAF.GRAF(L=L, M=M, n_samples_to_split=2, trial_count=M)
model.fit(data, labels)

model.saveTrees(fname.encode('utf-8'))

pred = model.predict(data)

print('\nTrain Predictions')
print(pred)

print('\nTest data')
print(test_data)
print('\nTest labels')
print(test_labels)
print('\nTest Predictions')
test_pred = model.predict(test_data)
print(test_pred)


#weights = model.getWeights()
#biases = model.getBiases()
#dims = model.getDims()
#heights = model.getTreeHeights()

#print('\n\n')
#for i in range(len(weights)):
#    print('weights')
#    print(weights[i])

#    print('biases')
#    print(biases[i])

#    print('dims')
#    print(dims[i])

#    print('Heights')
#    print(heights[i])

#    print('\n\n')

#print('\n\nPredictProb:\n')
#print(model.predictProba(data))


#print('\n\nPerTreePredict:\n')
#print(model.perTreePredict(data))


#print('\n\nLabelMap:\n')
#print(model.label_map)

#print('\n\nperSampleDepth:\n')
#print(model.getPerSampleDepth())

#print('\n\nperSampleDepthRanked:\n')
#print(model.getPerSampleRankedDepth())
