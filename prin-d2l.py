from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plot
from mxnet import autograd, np, npx
from d2l import mxnet as d2l
import pandas as pd
import math

# globals 
npx.set_np()
root = Tk()
fields = (
    'Lambda',
    'Error permisible',
)

def divideArray(l, n):
	for i in range(0, len(l), n): 
		yield l[i:i + n]

def ReadFile():
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
	# cleaning array Y
	a = Y[0].split(";")
	b = []
	# convert string to float
	for i in range(len(a)):
		b.append(float(a[i]))
	# create array Y
	Y = np.array(b)
	return X, Y

def start(p):
	X, Y = ReadFile()
	lamb = float(p['Lambda'].get())
	eps = float(p['Error permisible'].get())
	if lamb > 0 and lamb <= 1:
		print('matriz X->\n',X,'\nDimention:',X.shape)
		print('array Y->',Y,'\nDimention:',Y.shape)
		print('Lambda: ', lamb)
		print('Error perm: ', eps)
		# adding bias
		auxDimention = X.shape
		bias = [float(1) for i in range(auxDimention[0])]
		bias = np.array(bias)
		X = np.insert(X, 0, bias, axis=1)
		print('matriz X->\n',X,'\nNew dimention X:',X.shape)
		m, n = X.shape[0], X.shape[1]
		if n > 1 or m >= 2:
			W = np.random.rand(n,1)
			print('W->\n',W)
			U = np.dot(X, W)
			print('U->\n',U)
		else:
			print('Dimensiones incorrectas')
	else:
		print('Parametro incorrecto en lambda')

def makeform(root, fields):
    title = Label(root, text="Inicialización", width=20, font=("bold",20))
    title.pack()
    entries = {}
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=30, text=field+": ", anchor='w')
        ent = Entry(row)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent
    return entries

if __name__ == '__main__':
    root.title("Entrenamiento Neurona - UPCH IA")
    root.geometry("300x250")
    root.resizable(0,0)
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents:fetch(e)))
    b1 = Button(root, text = 'Iniciar',
        command=(lambda e=ents: start(e)), bg="green",fg='white')
    b1.pack(side = LEFT, padx = 5, pady = 5, expand = YES)
    b2 = Button(root, text = 'Quit', command = root.quit, bg="red",fg='white')
    b2.pack(side = LEFT, padx = 5, pady = 5, expand = YES)
    root.mainloop()