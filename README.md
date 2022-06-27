# Algoritmo de entrenamiento básico
Una neurona artificial es un sistema que recibe un conjunto de entradas y devuelve una salida (la salida puede estar compuesta de más de una señal).
Este sistema es posible adaptarlo para que dada una entrada coincida con una salida, a este proceso se le denomina **entrenamiento**.

El sistema debe de recibir una matriz<sup>m x n</sup> y un vector <sup>m x 1</sup>, desde el archivo, [data.txt](data.txt).
> ';' representa un salto de linea, o otra fila

#### Ejemplo de entradas en data.txt
||X|(8x4)||
|---|---|---|---|
| 1 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 1 | 0 | 1 | 0 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 0 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 0 |
| 1 | 1 | 1 | 1 |

| Y (1x8)=| 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
|---|---|---|---|---|---|---|---|---

### Parametrización
- **Error permisible**: parámetro de ajuste que indica la tolerancia de validación en el resultado analítico.
- **Entradas (X, Y)**: una colección de entradas ponderadas, con las que la neurona produce una decisión.

### Función de activación
- **Escalon o Umbral**: la cual es bastante sencilla, en dicha función recibimos un valor si este es mayor o igual a 0 entonces obtenemos 1 si es menor a 0 obtenemos 0.
    > valor { valor >= 0 return 1 ; valor < 0 return 0; }

### Requerimientos
```
Python >= 3.8
Tkinter >= 8.6
Numpy >= 1.19.5
Matplotlib >= 3.3.4
```
De igual forma se anexo un txt con los plugins necesarios, puede instarlos con `pip install -r requerimientos.txt`.
Luego de esto, ejecuta `principal.py`.
___
Si desea saber más, revise la [Wiki](https://github.com/SvS30/artificial-neuron-training/wiki).