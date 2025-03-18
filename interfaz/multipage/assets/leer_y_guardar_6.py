import serial
import time
import pandas as pd
from tkinter import Tk, Label, Button, Frame
from tkinter import filedialog

# Variables globales
global df
global is_reading

df = pd.DataFrame(columns=["Dato 1", "Dato 2", "Dato 3", "Dato 4"]) #,"Dato 5", "Dato 6"])
is_reading = False

# Inicializar Arduino
arduino = serial.Serial('COM15', 115200)

# Función para iniciar la lectura
def iniciar_lectura():
    global is_reading
    is_reading = True
    leer_datos()

# Función para finalizar la lectura
def finalizar_lectura():
    global is_reading
    is_reading = False

    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv")])

    if file_path:
        # Exportar datos a CSV en la ubicación seleccionada
        df.to_csv(file_path, index=False)

# Función para leer y actualizar la interfaz
def leer_datos():
    global df
    if is_reading:
        data = arduino.readline().decode()    #.strip() para eliminar cualquier espacio que pueda haber
        datos_separados = data.strip().split(',')   
        
        # Asegurarse de tener al menos 6 valores antes de actualizar las etiquetas
        if len(datos_separados) >= 4: #6:
            label_1.config(text=f'Dato 1: {datos_separados[0]}')
            label_2.config(text=f'Dato 2: {datos_separados[1]}')
            label_3.config(text=f'Dato 3: {datos_separados[2]}')
            label_4.config(text=f'Dato 4: {datos_separados[3]}')
            #label_5.config(text=f'Dato 5: {datos_separados[4]}')
            #label_6.config(text=f'Dato 6: {datos_separados[5]}')

            # Agregar datos al DataFrame
            df = df._append(pd.Series(datos_separados, index=df.columns), ignore_index=True)

        # Vuelve a llamar a la función después de 100 ms (ajusta según tus necesidades)
        root.after(100, leer_datos)





# Función para reiniciar
def reiniciar():
    global df
    df = pd.DataFrame(columns=["Dato 1", "Dato 2", "Dato 3", "Dato 4" ]) #, "Dato 5", "Dato 6"])
    label_1.config(text='Dato 1: ')
    label_2.config(text='Dato 2: ')
    label_3.config(text='Dato 3: ')
    label_4.config(text='Dato 4: ')
    #label_5.config(text='Dato 5: ')
    #label_6.config(text='Dato 6: ')

# Configuración de la interfaz
root = Tk()
root.title("Visualizador de Datos Arduino")

# Establecer el fondo de la interfaz a blanco
root.configure(bg="white")

frame = Frame(root, bg="white")
frame.pack(expand=True, fill="both")

label_font = ("Arial", 16)

label_1 = Label(frame, text='Dato 1: ', font=label_font, bg="white")
label_1.pack(pady=10)

label_2 = Label(frame, text='Dato 2: ', font=label_font, bg="white")
label_2.pack(pady=10)

label_3 = Label(frame, text='Dato 3: ', font=label_font, bg="white")
label_3.pack(pady=10)

label_4 = Label(frame, text='Dato 4: ', font=label_font, bg="white")
label_4.pack(pady=10)

# Utilizar configuraciones adicionales en lugar de Style para los botones
button_config = {
    "font": ("Arial", 14),
    "padx": 5,
    "bg": "#d3d3d3",  # Gris claro
    "fg": "#007acc",  # Celeste
    "relief": "flat",
}

boton_inicio = Button(frame, text="Iniciar Lectura", command=iniciar_lectura, **button_config)
boton_inicio.pack(side="left", padx=10)

boton_finalizar = Button(frame, text="Finalizar Lectura", command=finalizar_lectura, **button_config)
boton_finalizar.pack(side="left", padx=10)

boton_reiniciar = Button(frame, text="Reiniciar", command=reiniciar, **button_config)
boton_reiniciar.pack(side="right", padx=10)

# Iniciar el bucle de eventos de la interfaz
root.mainloop()