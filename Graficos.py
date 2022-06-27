from tkinter import LEFT, RIGHT, TOP, YES, Entry, Frame, Label, messagebox
from matplotlib import pyplot as plt
from numpy import any

PARAMETROS = (
    'Error permisible',
)

class Graficos:
    @staticmethod
    def crear_formulario(ventana):
        """Funcion para crear el formulario dentro de la ventana de Tkinter

        Args:
        ventana (Tkinter): Ventana de Tkinter

        Returns:
        entries (list): Campos a solicitar
        """
        title = Label(ventana, text="Modelación", width=20)
        title.pack()
        entries = {}
        for parametro in PARAMETROS:
            cuerpo_ventana = Frame(ventana)
            label = Label(cuerpo_ventana, width=30, text=f"{parametro}: ", anchor="w")
            inputs = Entry(cuerpo_ventana)
            cuerpo_ventana.pack(side=TOP, fill="both", padx=5, pady=5)
            label.pack(side=LEFT)
            inputs.pack(side=RIGHT, expand=YES, fill="both")
            entries[parametro] = inputs
        return entries

    @staticmethod
    def mostrar_mensaje(type, title, message):
        """Funcion para mostrar algun mensaje en pantalla

        Args:
            type (str): Tipo de mensaje a mostrar
            title (str): Titulo del mensaje
            message (str): Cuerpo del mensaje
        """
        if type == 'Info':
            messagebox.showinfo(title, message)
        elif type == 'Warning':
            messagebox.showwarning(title, message)
        else:
            messagebox.showerror(title, message)

    @staticmethod
    def crear_grafica_y_tabla(chart_data, table_data, pesos):
        """Funcion para crear la grafica de:
          - Evolucion de la norma de error por iteraciones
            - Eje X: iteraciones
            - Eje Y: norma de error
          - Tabla con las normas de error y iteraciones
          - Tabla con los pesos finales

        Args:
            chart_data (list): Datos para realizar grafica
            table_data (list): Datos para realizar tabla
            pesos (list): Datos con los pesos finales
        """
        plt.plot(chart_data[0]['iteraciones'], chart_data[0]['Norma error'], markerfacecolor='blue', markersize=6, color='skyblue', linewidth=3, label='Lambda: ' + str(chart_data[0]['Lambda']))
        plt.plot(chart_data[1]['iteraciones'], chart_data[1]['Norma error'], markerfacecolor='blue', markersize=6, color='yellowgreen', linewidth=3, label='Lambda: ' + str(chart_data[1]['Lambda']))
        plt.plot(chart_data[2]['iteraciones'], chart_data[2]['Norma error'], markerfacecolor='blue', markersize=6, color='orangered', linewidth=3, label='Lambda: ' + str(chart_data[2]['Lambda']))
        pesos_aux = [['Pesos'], [pesos[0]], [pesos[1]], [pesos[2]]] 
        table_weights = plt.table(
            cellText=pesos_aux,
            cellLoc ='center',
            colWidths=[0.35 for x in pesos_aux],
            loc='bottom right'
        )
        the_table = plt.table(
            cellText = table_data[0],
            rowLabels = table_data[1],
            colLabels = table_data[2],
            rowColours =["palegreen"] * len(table_data[1]),
            colColours =["palegreen"] * len(table_data[2]),
            cellLoc ='center',
            loc ='bottom'
        )
        table_weights.auto_set_font_size(False)
        table_weights.set_fontsize(7)
        table_weights.scale(1, 1)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(7)
        the_table.scale(1, 1)
        plt.subplots_adjust(left=0.043, bottom=0.165, right=0.730, top=0.948)
        plt.title('Evolución del error')
        plt.ylabel('Norma del error')
        plt.xticks([])
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
        plt.show()