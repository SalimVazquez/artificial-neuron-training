from tkinter import *
from tkinter import messagebox
import numpy as np
import random

# globals 
root = Tk()

if __name__ == '__main__':
    root.title("Entrenamiento Neurona - UPCH IA")
    root.geometry("300x450")
    root.resizable(0,0)
    title = Label(root, text="Inicialización", width=20, font=("bold",20))