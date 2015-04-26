import numpy as np
import random
from pandas import Series
from random import sample
import scipy.spatial.distance as scdist
import datetime
from timer import Timer

class Lshtable:
	def __init__(self,numkeys,lshfun,data):
		self.numK = numkeys
		self.numT = len(lshfun)
		self.f = lshfun
		self.data = data
		self.table = Series({l:Series({k:[] for k in range(self.numK)}) for l in range(self.numT)})
	
	def mapindex(self,key):
		vecfun=np.vectorize(lambda fun:fun.hash(key))
		return vecfun(self.f)

	def indexdata(self):
		for i in range(len(self.data)):
			e=self.data[i]
			L=0
			t1=datetime.datetime.now()
			for ind in self.mapindex(e):
				ind = ind % self.numK
				self.table[L][ind].append(i)
				L+=1
			t2=datetime.datetime.now()
			print "time point="+str(t2-t1)
	def candidateset(self,qpoint):
		L=0
		result = []
		for ind in self.mapindex(qpoint):
			ind = ind % self.numK
                        result.extend(self.table[L][ind])
			L+=1
		return np.unique(result)

class LSHkNN:
	def __init__(self,distfun,lshtble):
		self.dist = distfun
		self.table = lshtble

	def initVoronoiLSH(L,k,data):
		funlst = []
		dist = scdist.euclidean
		print "data[0] "
		print data[0]
		for i in range(L):
			idx = random.sample(data,k)
			
			#print idx 
			funlst.append(VoronoiLSH(idx,dist))
		table = Lshtable(k,np.array(funlst),data)
		knn  = LSHkNN(dist,table)
		with Timer() as t:
			table.indexdata()	
		print "=> elasped indexdata: %s s" % t.secs
		return knn

    	initVoronoiLSH = staticmethod(initVoronoiLSH)


	def candidateDistance(self,query):
		fcalcdist=lambda ind:(self.dist(self.table.data[ind],query),ind)
		return [fcalcdist(i) for i in self.table.candidateset(query)]
	def kNNQuery(self,k,query):
	        distlst = self.candidateDistance(query)
		distlst.sort()
		return distlst[:k]
	def rangeQuery(self,radi,query):
		distlst = candidateDistance(query)
		return [(x,y) for (x,y) in distlst if x < radi]

class VoronoiLSH:
	def __init__(self,anchorpoints,dist):
		self.dist=dist
		self.C = anchorpoints
	def hash(self,q):
		t1=datetime.datetime.now()
		#r = scdist.cdist(np.array([q]), self.C, 'euclidean',p=1).argmin
		
		r = np.array([self.dist(x,q) for x in self.C]).argmin()
		#fdist=np.vectorize(lambda x:self.dist(x,q))
		#r = fdist(self.C).argmin()
		print "time hash="+str(datetime.datetime.now()-t1)
		return r
class E2LSH:
	def __init__(self,d,w,m):
		self.d = d
		self.w = w
		self.m = m

class CossineLSH:
	def __init__(self,d,numplanes,m):
		self.d = d
		self.b = numplanes
		self.m = m 

