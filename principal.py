from tkinter import *
from tkinter import messagebox
import numpy as np
import random

# globals 
root = Tk()

if __name__ == '__main__':
    root.title("Entrenamiento Neurona - UPCH IA")
    root.geometry("300x250")
    root.resizable(0,0)
    title = Label(root, text="Inicialización", width=20, font=("bold",20))
    title.pack()
    b1 = Button(root, text = 'Iniciar', bg="green",fg='white')
    b1.pack(side = LEFT, padx = 5, pady = 5, expand = YES)
    b2 = Button(root, text = 'Quit', command = root.quit, bg="red",fg='white')
    b2.pack(side = LEFT, padx = 5, pady = 5, expand = YES)
    root.mainloop()