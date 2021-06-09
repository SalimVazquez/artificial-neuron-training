from tkinter import *
from tkinter import messagebox
import numpy as np
import random

# globals 
root = Tk()
fields = (
    'Lambda',
    'Error permisible',
)
X # Variable for matrix A
Y # Variable for array B

def ReadFile():
    global X
    global Y
    f = open('data.txt', 'r')
    data = f.read()
    clean = data.rsplit("\n")
    print('X:', clean[0])
    print('Y:', clean[1])
    X = np.matrix(clean[0]) # create matrix X
    y = clean[1].replace(";", ",")
    Y = np.array((y)) # create array Y
    f.close()

def start(entries):
    ReadFile()

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