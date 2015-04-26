import numpy as np
from hashtable import LSHkNN 

def readfileAPM(filename):
	fp = open(filename)
	res = []
	for line in fp:
		y = line.split()
		if(len(y) > 128):
			vec=np.vectorize(int)(np.array(y[(134-128):]))
			res.append(vec)
	return np.array(res)

print "reading file "
data = readfileAPM('/home/eliezer/datasets/apm/apm100.train.query.300.keys')
print data[0]
print "file processed"
print "indexing"
knn = LSHkNN.initVoronoiLSH(2,10,data)
print "querying"
x = knn.kNNQuery(10,data[0])
print x
