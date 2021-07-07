#!/usr/bin/python
# -*- coding: utf-8 -*-
from tkinter import *
from tkinter import messagebox
# Library for graph
import matplotlib.pyplot as plot
# Library of D2L
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init, nd
from mxnet.gluon import nn
import pandas as pd
# Class for Activation Function
from FAEscalon import FAEscalon

# globals
npx.set_np()
# Architecture of networks
# with one output and FA "relu"
net = nn.Sequential()
net.add(nn.Dense(1, activation='relu'))
# net.add(nn.Dense(1), FAEscalon())
# normal distribution in weights (sigma)
net.initialize(init.Normal(sigma=0.03))
root = Tk()
fields = ('Lambda', 'Error permisible')
loss = gluon.loss.L2Loss()

def graphEvol(x,y,lamb,w):
    (fig, axs) = plot.subplots(2, 1)
    labelCol = 'Weights'
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].set_title('Weights finals')
    columns = ['W {:X}'.format(i + 1) for i in range(len(w[0]))]
    rows = ['Weights' for i in range(1)]
    data = [['{:02}'.format(w[r][c]) for c in range(len(w[0]))]
            for r in range(1)]
    table = axs[0].table(cellText=data, rowLabels=rows,
                         colLabels=columns, cellLoc='center',
                         loc='center')
    table.set_fontsize(16)
    table.scale(1.2, 1.1)
    plot.xlabel('Epochs')
    plot.ylabel('|E|')
    axs[1].set_title('Evolution of the error norm')
    axs[1].plot(x,y,
        markerfacecolor='blue',
        markersize=6,
        color='skyblue',
        linewidth=3,
        label='Lambda: ' + str(lamb),
        )
    plot.legend(bbox_to_anchor=(1, 1), loc='upper center',
                borderaxespad=0.)
    plot.show()

def load_array(data_arrays, batch_size, is_train=True):
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

def divideArray(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def ReadFile():
    # get data from data.txt
    data = pd.read_table('data.txt', delimiter='\t', header=None)
    (X, Y) = (data.iloc[0, :], data.iloc[1, :])
    # cleaning matriz X
    a = X[0].split(';')
    b = []
    # convert string to float
    for i in range(len(a)):
        for j in range(0, len(a[i]), 2):
            b.append(float(a[i][j]))
    # Split list to array, for create matriz X
    X = list(divideArray(b, 3))
    X = np.array(X)
    # cleaning array Y
    a = Y[0].split(';')
    b = []
    # convert string to float
    for i in range(len(a)):
        b.append(float(a[i]))
    # create array Y
    Y = np.array(b)
    return (X, Y)

def start(p):
    epochs = []
    evol_norm = []
    i = 0
    lr = float(p['Lambda'].get())
    eps = float(p['Error permisible'].get())
    if lr > 0 and lr <= 1:
        (X, Y) = ReadFile()
        print ('matriz X->\n', X, '\nDimention:', X.shape)
        print ('array Y->', Y, '\nDimention:', Y.shape)
        print ('Lambda: ', lr)
        print ('Error perm: ', eps)
        (m, n) = (X.shape[0], X.shape[1])
        if n > 1 or m >= 2:
            data_iter = load_array((X, Y), m)
            trainer = gluon.Trainer(net.collect_params(), 'sgd',
                                    {'learning_rate': lr})
            while True:
                print ('<---Epoch #', i + 1, '--->')
                epochs.append(i)
                for (x, y) in data_iter:
                    with autograd.record():
                        l = loss(net(x), y)
                    l.backward()
                    trainer.step(m)
                l = loss(net(X), Y)
                print(f'epoch {i+1}, loss {l.mean().asnumpy():f}')
                print(f'weights {net[0].weight.data()}')
                evol_norm.append(l.mean().asnumpy())
                if l.mean().asnumpy() > eps:
                    print('Try again')
                    i += 1
                else:
                    break
            w = net[0].weight.data()
            graphEvol(epochs, evol_norm, lr, w)
        else:
            messagebox.showerror('Parametros incorrectos',
                                 'Dimensiones no correctas')
    else:
        messagebox.showerror('Parametros incorrectos',
                             'Lambda fuera de parametros (0, 1]')


def makeform(root, fields):
    title = Label(root, text='Inicializacion', width=20,
                  font=('bold', 20))
    title.pack()
    entries = {}
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=30, text=field + ': ', anchor='w')
        ent = Entry(row)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent
    return entries

if __name__ == '__main__':
    root.title('Entrenamiento Neurona - UPCH IA')
    root.geometry('300x250')
    root.resizable(0, 0)
    ents = makeform(root, fields)
    root.bind('<Return>', lambda event, e=ents: fetch(e))
    b1 = Button(root, text='Iniciar', command=lambda e=ents: start(e),
                bg='green', fg='white')
    b1.pack(side=LEFT, padx=5, pady=5, expand=YES)
    b2 = Button(root, text='Quit', command=root.quit, bg='red',
                fg='white')
    b2.pack(side=LEFT, padx=5, pady=5, expand=YES)
    root.mainloop()