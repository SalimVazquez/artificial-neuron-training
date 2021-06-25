import matplotlib.pyplot as plot
from mxnet import autograd, np, npx
from d2l import mxnet as d2l
import pandas as pd

npx.set_np()

def divideArray(l, n):
	for i in range(0, len(l), n): 
		yield l[i:i + n]

# get data from data.txt
data = pd.read_table('data.txt', delimiter='\t', header=None)
X, Y = data.iloc[0, :], data.iloc[1, :]

# cleaning matriz X
a = X[0].split(";")
b = []

# convert string to float
for i in range(len(a)):
	for j in range(0, len(a[i]), 2):
		b.append(float(a[i][j]))

# Split list to array, for create matriz X
X = list(divideArray(b, 3))
X = np.array(X)
print('matriz X->\n',X,'\nDimention:',X.shape)

# cleaning array Y
a = Y[0].split(";")
b = []
# convert string to float
for i in range(len(a)):
	b.append(float(a[i]))

# create array Y
Y = np.array(b)
print('array Y->',Y,'\nDimention:',Y.shape)