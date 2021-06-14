from tkinter import *
from tkinter import messagebox
import numpy as np
import random
import math
import matplotlib.pyplot as plot

# globals 
root = Tk()
fields = (
    'Error permisible',
)

def FAEscalon(U):
    Yc = []
    for i in range(len(U)):
        if U[i] <= 0:
            Yc.append(0)
        else:
            Yc.append(1)
    return np.array(Yc)

def calculateError(E):
    result = 0
    for i in range(len(E)):
        result = result + math.pow(E[i], 2)
    return math.sqrt(result)

def ReadFile():
    f = open('data.txt', 'r')
    data = f.read()
    clean = data.rsplit("\n")
    X = np.matrix(clean[0]) # create matrix X
    y = clean[1].replace(";", ",")
    b = []
    for i in range(0, len(y), 2):
        b.append(y[i])
    Yaux = [int(e) for e in b]
    Y = np.array(Yaux) # create array Y
    f.close()
    return X, Y

def graphEvol(list):
    epochs1 = []
    epochs2 = []
    epochs3 = []
    eNorm1 = []
    eNorm2 = []
    eNorm3 = []
    w1 = 0
    w2 = 0
    w3 = 0
    lamb1 = 0
    lamb2 = 0
    lamb3 = 0
    weights = []
    for i in range(len(list)):
        if list[i]['LambdaID'] == 1:
            epochs1.append(list[i]['Epoch'])
            eNorm1.append(list[i]['Enorm'])
            lamb1 = list[i]['Lambda']
            if np.any(list[i]['W']):
                w1 = list[i]['W']
        elif list[i]['LambdaID'] == 2:
            epochs2.append(list[i]['Epoch'])
            eNorm2.append(list[i]['Enorm'])
            lamb2 = list[i]['Lambda']
            if np.any(list[i]['W']):
                w2 = list[i]['W']
        else:
            epochs3.append(list[i]['Epoch'])
            eNorm3.append(list[i]['Enorm'])
            lamb3 = list[i]['Lambda']
            if np.any(list[i]['W']):
                w3 = list[i]['W']
    labels = ('Lambda', 'Pesos')
    data1 = [[lamb1, w1]]
    data2 = [[lamb2, w2]]
    data3 = [[lamb3, w3]]
    weights.append(data1)
    weights.append(data2)
    weights.append(data3)
    plot.xlabel('Epocas')
    plot.ylabel('Norma del error')
    plot.title('Evolución del error')
    plot.plot(epochs1, eNorm1, markerfacecolor='blue',
             markersize=6, color='skyblue', linewidth=3, label='Lambda: '+str(lamb1))
    plot.plot(epochs2, eNorm2, markerfacecolor='red',
             markersize=6, color='red', linewidth=3, label='Lambda: '+str(lamb2))
    plot.plot(epochs3, eNorm3, markerfacecolor='green',
             markersize=6, color='green', linewidth=3, label='Lambda: '+str(lamb3))
    table = plot.table(cellText=weights, colLabels=labels, loc='bottom')
    table.set_fontsize(35)
    plot.subplots_adjust(left=0.2, bottom=0.2)
    plot.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    plot.show()

def printList(list):
    for i in range(len(list)):
        print(list[i])

def start(entries):
    epochs = []
    evolNorm = []
    evaluations = []
    i = 0
    X, Y = ReadFile()
    eps = float(entries['Error permisible'].get())
    print('X:\n', X)
    print('Y: ', Y)
    print('Error: ', eps)
    dimensionsX = X.shape
    print('dimensionsX: ', dimensionsX)
    # adding bias
    bias = [1 for i in range(dimensionsX[0])]
    X = np.insert(X, 0, bias, axis=1)
    dimensionsX = X.shape
    print('dimensionsX: ', dimensionsX)
    m = dimensionsX[0]
    n = dimensionsX[1]
    if n > 1 or m >= 2:
        W = np.random.rand(n,1)
        auxW = W
        k = 0
        for j in range(3):
            W = auxW
            lamb = random.uniform(0,1)
            while True:
                print('<----- Epoca #', i+1, ' || Lambda: ', lamb, ' ----->')
                print('W:\n', W)
                U = X.dot(W)
                print('U:\n', U)
                Yc = FAEscalon(U)
                print('Yc:', Yc)
                E = Yc - Y
                print('E:', E)
                EtX = np.dot(E.transpose(), X)
                print('Et * X: ',EtX)
                Ne = lamb * EtX
                print('n * Et * X: ', Ne)
                W = W.transpose() - Ne
                print('new W: ',W)
                enorm = calculateError(E)
                print('ENorm:', enorm)
                dictData = {'LambdaID': j+1, 'Epoch': i+1, 'Enorm': enorm, 'Lambda': lamb, 'W': 0}
                evaluations.append(dictData)
                if enorm > eps:
                    W = W.transpose()
                    print('Try again!')
                    i += 1
                else:
                    k = i + k
                    break
            print('Finish')
            evaluations[k]['W'] = W
            i = 0
        printList(evaluations)
        graphEvol(evaluations)
    else:
        messagebox.showerror("Parametros incorrectos", "Dimensiones no correctas")

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