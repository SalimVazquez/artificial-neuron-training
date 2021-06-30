from tkinter import *
from tkinter import messagebox
import numpy as np
import random
import math
import matplotlib.pyplot as plot

# globals 
root = Tk()
fields = (
    'Lambda',
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

def graphEvol(x,y, lamb):
    plot.xlabel('Epocas')
    plot.ylabel('Norma del error')
    plot.title('Evolución de la norma del error')
    plot.plot(x, y, markerfacecolor='blue',
             markersize=6, color='skyblue', linewidth=3, label='Lambda: '+str(lamb))
    plot.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    plot.show()

def start(entries):
    epochs = []
    evolNorm = []
    i = 0
    X, Y = ReadFile()
    lamb = float(entries['Lambda'].get())
    eps = float(entries['Error permisible'].get())
    if lamb > 0 and lamb <= 1:
        print('X:\n', X)
        print('Y: ', Y)
        print('Lambda: ', lamb)
        print('Error: ', eps)
        dimensionsX = X.shape
        print('dimensionsX: ', dimensionsX)
        # adding bias
        bias = [1 for i in range(dimensionsX[0])]
        X = np.insert(X, 0, bias, axis=1)
        dimensionsX = X.shape
        print('bias added\ndimensionsX: ', dimensionsX)
        m = dimensionsX[0]
        n = dimensionsX[1]
        if n > 1 or m >= 2:
            W = np.random.rand(n,1)
            while True:
                print('<----- Epoca #', i+1, ' ----->')
                epochs.append(i)
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
                evolNorm.append(enorm)
                if enorm > eps:
                    W = W.transpose()
                    print('Try again!')
                    i += 1
                else:
                    break
            print('Finish')
            messagebox.showinfo("Norma del error", str(enorm))
            messagebox.showinfo("Configuración W", str(W))
            graphEvol(epochs, evolNorm, lamb)
        else:
            messagebox.showerror("Parametros incorrectos", "Dimensiones no correctas")
    else:
        messagebox.showerror("Parametros incorrectos", "Lambda fuera de parametros (0, 1]")

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