from numpy import array, matrix, insert, any
from math import pow, sqrt

class Utils:

    def formato_a_2_decimales(data):
        """Funcion para formatear los registros a 2 decimales

        Args:
            data (list): Lista con los datos a formatear

        Returns:
            (list): Lista con datos formateados
        """
        datos_formateados = [[round(value,2) for value in row] for row in data]
        return datos_formateados

    @staticmethod
    def conversion_de_pesos(weights):
        """Funcion encargada de formatear los pesos a 2 decimales y agrupar a una sola lista

        Args:
            weights (list): Lista con los pesos finales

        Returns:
            (list): Lista con pesos formateados
        """
        aux = []
        contador = 1
        for weight in weights:
            weight = Utils.formato_a_2_decimales(weight)
            aux.insert(contador, list(weight))
            contador += 1
        return aux

    @staticmethod
    def obtener_pesos(data):
        """Funcion para recolectar los pesos finales

        Args:
            data (list): Lista con todos los registros del algoritmo

        Returns:
            (list): Lista de pesos finales
        """
        pesos = []
        for i in range(len(data)):
            if any(data[i]['Pesos']):
                pesos.append(data[i]['Pesos'])
        return array(pesos).tolist()

    @staticmethod
    def normalizar_norma_de_error(maximum, iteraciones, normas_de_error):
        """Funcion para calcular y completar las listas de norma de error,
        para que sean del mismo tamaño, estas se completan con '-'.

        Args:
            maximum (int): Maxima iteracion registrada.
            iteraciones (list): Lista con las iteraciones registradas.
            normas_de_error (list): Lista con las normas de error registradas.

        Returns:
            (list): Lista con las normas de error normalizadas con la misma longitud
        """
        for iteracion in range(len(iteraciones)):
            resto = maximum - max(iteraciones[iteracion])
            if resto > 0:
                [ normas_de_error[iteracion].append('-') for i in range(resto) ]
        return normas_de_error

    @staticmethod
    def obtener_maxima_iteracion(lista_iteraciones):
        """Funcion para obtener el mayor numero de iteraciones registradas.

        Args:
            lista_iteraciones (list): Lista con las iteraciones realizadas por el algoritmo.

        Returns:
            (int): Numero con la mayor interaccion registrada.
        """
        max_iteracion = 0
        for i in range(len(lista_iteraciones)):
            max_iteracion = max(max_iteracion, lista_iteraciones[i][-1])
        return max_iteracion

    def formatear_a_diccionario(a, b ,c):
        """Funcion de formatear un diccionario.

        Args:
            a (float): Lambda usada.
            b (list): Norma de error obtenida con esa lambda.
            c (list): Iteraciones realizadas en el algoritmo con esa lambda.

        Returns:
            (dict): Diccionario con información por lambda.
        """
        return { 'Lambda': a, 'Norma error': b, 'iteraciones': c }

    def obtener_colecciones(list_data):
        """Funcion encargada de recuperar las colecciones asociadas al id de lambda buscado con anterioridad.

        Args:
            list_data (list): Lista con el historial del algoritmo de una lambda.
        """
        norma_error = []
        iteraciones = []
        for data in list_data:
            norma_error.append(data['Norma'])
            iteraciones.append(data['Iteraciones'])
        lambda_usada = list_data[0]['Lambda']
        return Utils.formatear_a_diccionario(lambda_usada, norma_error, iteraciones)

    @staticmethod
    def dividir_por_lambdas(criterio, evaluaciones):
        """Funcion encargada de separar las evaluaciones por el id de lambda.

        Args:
            criterio (list): Lista con el criterio a buscar, en este caso el id de lambda.
            evaluaciones (list): Lista con las evaluaciones/historial del algoritmos.
        """
        data = [ registro for registro in evaluaciones if
            all( id_lambda == registro['LambdaID'] for id_lambda in criterio ) ]
        return Utils.obtener_colecciones(data)

    @staticmethod
    def funcion_activacion_escalon(pesos):
        """Funcion para simular la funcion de activacion Escalon o Umbral.

        Args:
            pesos (list): Pesos calculados.

        Returns:
            numpy.array: array con 0's y 1's.
        """
        y_calculada = []
        for i in range(len(pesos)):
            if pesos[i] <= 0:
                y_calculada.append(0)
            else:
                y_calculada.append(1)
        return array(y_calculada)

    @staticmethod
    def calcular_error(errors):
        """Funcion para calcular el error.

        Args:
            errors (list): Lista con los errores calculados.

        Returns:
            float: Error calculado.
        """
        resultado = 0
        for i in range(len(errors)):
            resultado = resultado + pow(errors[i], 2)
        return sqrt(resultado)

    @staticmethod
    def imprimir_lista(lista):
        """Funcion general para imprimir alguna lista.

        Args:
            lista (list)
        """
        for i in range(len(lista)):
            print(lista[i])

    @staticmethod
    def leer_archivo():
        """Funcion para recuperar las entradas del archivo data.txt.

        Returns:
            X (numpy.matrix): Matriz con las entradas de X.
            Y (numpy.array): Array con las entradas de Y.
        """
        file_data = open('data.txt', 'r')
        data = file_data.read()
        clean = data.rsplit("\n")
        X = matrix(clean[0])
        y = clean[1].replace(";", ",")
        auxiliar = []
        for i in range(0, len(y), 2):
            auxiliar.append(y[i])
        Y_aux = [int(e) for e in auxiliar]
        Y = array(Y_aux)
        file_data.close()
        return X, Y

    @staticmethod
    def agregar_bias(X, dimensiones_x):
        """Funcion encargada de agregar el bias al eje (0,0).

        Args:
            X (numpy.matrix): Matriz con las entradas de X.
            dimensiones_x (list): Dimensiones de la matriz X.

        Returns:
            X (numpy.matrix): Matriz con las entradas de X y el bias.
        """
        bias = [1 for i in range(dimensiones_x[0])]
        X = insert(X, 0, bias, axis=1)
        return X