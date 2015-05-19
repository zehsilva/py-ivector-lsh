import numpy as np
import random
import pandas
#import scipy.spatial.distance as scdist
import datetime

#from timer import Timer

class LSHTable:
    
    def __init__(self, numkeys, lshfun, data):
        self.numK = numkeys
        self.numT = len(lshfun)
        self.f = lshfun
        self.data = data
        self.table = pandas.Series({l: pandas.Series({k: [] for k in
                                                      xrange(self.numK)})
                                    for l in xrange(self.numT)})
    
    def map_index(self, key):
        vecfun = np.vectorize(lambda fun: fun.hash(key))
        
        return vecfun(self.f)

    def index_data(self):
        for i in xrange(len(self.data)):
            e = self.data[i]
            
            for L, ind in enumerate(self.map_index(e)):
                ind = ind % self.numK
                self.table[L][ind].append(i)
    
    def candidate_set(self, qpoint):
        result = []
        
        for L, ind in enumerate(self.map_index(qpoint)):
            ind = ind % self.numK
            result.extend(self.table[L][ind])
        
        return np.unique(result)

class LSHkNN:
    
    @staticmethod
    def create_voronoi_lsh(L, k, data):
        funlst = []
        #dist = scdist.cosine #euclidean
        dist = lambda a, b: 1 - np.dot(a, b)
        
        for i in xrange(L):
            idx = random.sample(data, k)
            funlst.append(VoronoiLSH(idx, dist))
        
        table = LSHTable(k, np.array(funlst), data)
        knn = LSHkNN(dist, table)
        
        #with Timer() as t:
        table.index_data()
        
        #print "=> elasped index_data: %s s" % t.secs
        
        return knn
    
    def __init__(self, distfun, lshtble):
        self.dist = distfun
        self.table = lshtble
    
    def candidate_distance(self, query):
        #fcalcdist = lambda ind: (self.dist(self.table.data[ind], query), ind)
        #
        #return [fcalcdist(i) for i in self.table.candidate_set(query)]
        candidates = self.table.candidate_set(query)
        
        return self.dist(self.table.data[candidates, :],
                         query[:, np.newaxis])[:, 0], candidates
    
    def knn_query(self, k, query):
        distlst, indices = self.candidate_distance(query)
        idxs = np.argsort(distlst)[:k]
        
        return distlst[idxs], indices[idxs]
    
    def range_query(self, radi, query):
        distlst = self.candidate_distance(query)
        
        return [(x, y) for (x, y) in distlst if x < radi]

class VoronoiLSH:
    
    def __init__(self, anchorpoints, dist):
        self.dist = dist
        self.C = anchorpoints
    
    def hash(self, q):
        #r = scdist.cdist(np.array([q]), self.C, 'euclidean',p=1).argmin
        r = np.argmin([self.dist(x, q) for x in self.C])
        #fdist=np.vectorize(lambda x:self.dist(x,q))
        #r = fdist(self.C).argmin()
        
        return r

class E2LSH:
    
    def __init__(self, d, w, m):
        self.d = d
        self.w = w
        self.m = m

class CossineLSH:
    
    def __init__(self, d, numplanes, m):
        self.d = d
        self.b = numplanes
        self.m = m 

