# python
import numpy as np
from numpy import genfromtxt
import scipy
import random
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, dok_matrix
from scipy import sparse, io
data = genfromtxt('ratings.dat', delimiter='::') # UserID::MovieID::Rating::Timestamp
data = np.delete(data, 3, 1) # delete timestamp
x = data[:,0] # userid
y = data[:,1] # movieid
n = int(max(x))
m = int(max(y))
x = x - 1 # userid starting from 0
y = y - 1 # same for movieid
v = data[:,2]

# store in csv to be read in julia
import csv
f = open('MovieLens1m.csv', 'w')
writer = csv.writer(f)
for i in range(len(v)):
	writer.writerow( (int(x[i]), int(y[i]), int(v[i]) ))
f.close()