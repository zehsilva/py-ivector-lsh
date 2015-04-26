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

data = readfileAPM('/home/eliezer/datasets/apm/apm100.train.query.300.keys')
knn = LSHkNN.initVoronoiLSH(2,10,data)


