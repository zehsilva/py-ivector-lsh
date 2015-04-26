from Pandas import Series

class Lshtable:
	def __init__(self,numkeys,lshfun,data):
		self.numK = numkeys
		self.numT = len(lshfun)
		self.f = lshfun
		self.data = data
		self.table = Series(l:Series({k:[] for k in range(self.numk)}) for l in range(self.numT)})
	
	def mapindex(self,key):
		vecfun=np.vectorize(lambda f:f(key))
		return vecfun(self.f)

	def indexdata(self):
		for i in range(len(self.data)):
			e=self.data[i]
			L=0
			for ind in self.mapindex(e):
				ind = ind % self.numK
				self.table[L][ind].append(i)
				L+=1

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
	
	def kNNQuery(self,k,query):
		fcalcdist=lambda ind:(self.dist(self.table.data[ind],query),ind)
		distlst = [fcalcdist(i) for i in self.table.candidateset(query)]
		distlst.sort(reverse=True)
		return distlst[:k]

		
