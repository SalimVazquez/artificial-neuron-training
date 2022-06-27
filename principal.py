from tkinter import LEFT, YES, Button, Tk
from numpy import random, dot
from Graficos import Graficos
from Utils import Utils

def obtener_datos_para_tabla(list):
    # Preparamos en una lista los registros por lambdas
    contenedor_iteraciones = [ list[0]['iteraciones'], list[1]['iteraciones'], list[2]['iteraciones'] ]
    contenedor_norma_error = [ list[0]['Norma error'], list[1]['Norma error'], list[2]['Norma error'] ]
    # Normalizamos los datos
    max_iteracion = Utils.obtener_maxima_iteracion(contenedor_iteraciones)
    norma_error_formateada = Utils.formato_a_2_decimales(contenedor_norma_error)
    norma_error_normalizada = Utils.normalizar_norma_de_error(max_iteracion, contenedor_iteraciones, norma_error_formateada)
    # Preparamos los datos para la tabla
    colls_labels_iteracion = [ x for x in range(max_iteracion) ]
    cell_values_by_norma_error = []
    cell_values_by_norma_error.append(norma_error_normalizada[0])
    cell_values_by_norma_error.append(norma_error_normalizada[1])
    cell_values_by_norma_error.append(norma_error_normalizada[2])
    row_labels = [
        list[0]['Lambda'], list[1]['Lambda'], list[2]['Lambda'] ]
    return cell_values_by_norma_error, row_labels, colls_labels_iteracion

def preparar_datos(evaluaciones):
    # Datos para obtener informacion para la tabla
    data = []
    data.append(Utils.dividir_por_lambdas([1], evaluaciones))
    data.append(Utils.dividir_por_lambdas([2], evaluaciones))
    data.append(Utils.dividir_por_lambdas([3], evaluaciones))
    table_data = obtener_datos_para_tabla(data)
    # Datos para obtener informacion para la grafica
    chart_data = []
    chart_data.append(Utils.dividir_por_lambdas([1], evaluaciones))
    chart_data.append(Utils.dividir_por_lambdas([2], evaluaciones))
    chart_data.append(Utils.dividir_por_lambdas([3], evaluaciones))
    pesos_tmp = Utils.obtener_pesos(evaluaciones)
    pesos = Utils.conversion_de_pesos(pesos_tmp)
    return chart_data, table_data, pesos

def entrenar(X, Y, pesos, id_lambda, lambd, error_ps, evaluaciones):
    """Funcion encargada de realizar el entrenamiento a las entradas

    Args:
        X (numpy.matrix): Entradas de X
        Y (numpy.array): Entradas de Y
        pesos (numpy.array): pesos
        id_lambda (int): Lambda's id
        lambd (float): Tasa de aprendizaje
        error_ps (float): Error permisible
        evaluaciones (list): Lista para almacenar la evolucion del error

    Returns:
        evaluaciones (list): Registro de la evolucion del error
    """
    print('===== BUCLE DE ENTRENAMIENTO =====')
    contador_iteraciones = 0
    while True:
        print(f"===== Iteracion {contador_iteraciones+1}, Lambda: {lambd} =====")
        # Calculo de uk: producto matricial entre X y los pesos
        U = X.dot(pesos)
        print(f"U {U.shape}:\n {U}")
        # Calculo de yc: Uso de la funcion de activacion
        Y_calculada = Utils.funcion_activacion_escalon(U)
        print(f"Y_calculada {Y_calculada.shape}: {Y_calculada}")
        # Calculo del error ek: Resta vectorial entre Y deseada e Y calculada
        error = Y_calculada - Y
        print(f"Error: {error}")
        # Actualizacion de pesos: (Wk + (n*(E^t*X)))
        nuevos_pesos = pesos.transpose() - (lambd * dot(error.transpose(), X))
        print(f"nuevos pesos: {nuevos_pesos}")
        # Obtenemos la norma de error
        norma_error = Utils.calcular_error(error)
        print(f"norma_error: {norma_error}")
        if norma_error > error_ps:
            diccionario_data = {
                'LambdaID': id_lambda+1,
                'Lambda': lambd,
                'Iteraciones': contador_iteraciones+1,
                'Norma': norma_error,
                'Pesos': 0
            }
            evaluaciones.append(diccionario_data)
            pesos = nuevos_pesos.transpose()
            print('Error muy alto!')
            contador_iteraciones += 1
        else:
            print('Pesos final:', pesos)
            print('Norma del error final:', norma_error)
            diccionario_data = {
                'LambdaID': id_lambda+1,
                'Lambda': lambd,
                'Iteraciones': contador_iteraciones+1,
                'Norma': norma_error,
                'Pesos': pesos
            }
            evaluaciones.append(diccionario_data)
            print('Error aceptable')
            break
    print('===== FIN BUCLE DE ENTRENAMIENTO =====')
    return evaluaciones

def iniciar(parametros):
    evaluaciones = []
    X, Y = Utils.leer_archivo()
    error_ps = float(parametros['Error permisible'].get())
    print(f"X {X.shape}:\n{X}")
    print(f"Y {Y.shape}: {Y}")
    print(f"Error: {error_ps}")
    X = Utils.agregar_bias(X, X.shape)
    print(f"X con bias ({X.shape}):\n{X}")
    if X.shape[0] < 2 or X.shape[1] <= 1:
        Graficos.mostrar_mensaje('Error', 'Parametros Incorrectos', 'Dimensiones incorrectas')
    else:
        pesos_originales = random.rand(X.shape[1],1)
        for iterador in range(3):
            lambda_aleatoria = round(random.uniform(0,1),2)
            evaluaciones = entrenar(X, Y, pesos_originales, iterador, lambda_aleatoria, error_ps, evaluaciones)
        chart_data, data_table, pesos = preparar_datos(evaluaciones)
        Graficos.crear_grafica_y_tabla(chart_data, data_table, pesos)


if __name__ == '__main__':
    """Funcion main para crear la ventana inicial.
    """
    ventana = Tk()
    ventana.title("Entrenamiento Neurona - IA")
    ventana.resizable(0,0)
    entries = Graficos.crear_formulario(ventana)
    b1 = Button(ventana, text='Iniciar',
        command=(lambda parametros=entries: iniciar(parametros)), bg="green",fg='white')
    b1.pack(side=LEFT, padx=5, pady=5, expand=YES)
    b2 = Button(ventana, text='Quit', command=ventana.quit, bg="red",fg='white')
    b2.pack(side=LEFT, padx = 5, pady=5, expand=YES)
    ventana.mainloop()