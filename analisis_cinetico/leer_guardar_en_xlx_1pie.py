import serial
import pandas as pd
import keyboard  # Asegúrate de que la librería keyboard está instalada

# Inicializar Arduino (cambiar COM8 según lo indicado)
try:
    arduino_derecha = serial.Serial('COM6', 115200, timeout=1)
    arduino_derecha.flushInput()
    arduino_derecha.flushOutput()
    print("Conexión establecida con Arduino en COM8")
except serial.SerialException as e:
    print(f"Error al conectar con Arduino en COM8: {e}")
    exit()

# Variables de control
running = True
guardando_datos = False
contador = 0
archivo_numero = 1
MAX_LECTURAS = 1000

# Inicializar DataFrame
columnas = [f"Sensor {i+1}" for i in range(9)]
df_datos = pd.DataFrame(columns=columnas)

def leer_datos(arduino, dataframe):
    """Función para leer datos del Arduino y guardar en un DataFrame si es necesario."""
    global guardando_datos, contador, archivo_numero

    if arduino.in_waiting > 0:
        try:
            linea = arduino.readline().decode('utf-8').strip()
            datos = linea.split(',')

            if len(datos) == 9:  # Asegurar que la línea tiene los datos esperados
                fila = [int(valor) for valor in datos]
                print(f"Datos: {fila}")

                if guardando_datos:
                    dataframe.loc[len(dataframe)] = fila
                    contador += 1

                    # Si alcanzamos el máximo de lecturas, guardar el archivo y reiniciar
                    if contador >= MAX_LECTURAS:
                        nombre_archivo = f"Derecha_SIN_S1_{archivo_numero}_6.csv"
                        dataframe.to_csv(nombre_archivo, index=False)
                        print(f"Datos guardados en '{nombre_archivo}'")

                        # Reiniciar variables para el siguiente bloque de datos
                        contador = 0
                        guardando_datos = False
                        archivo_numero += 1
                        dataframe.drop(dataframe.index, inplace=True)  # Vaciar el DataFrame
        except Exception as e:
            print(f"Error al leer datos: {e}")

# Bucle principal
try:
    print("Presiona 'p' para iniciar el guardado de 1000 datos en un nuevo archivo.")
    while running:
        if keyboard.is_pressed('p') and not guardando_datos:
            guardando_datos = True
            print("Iniciando guardado de datos...")

        leer_datos(arduino_derecha, df_datos)

except KeyboardInterrupt:
    print("Programa interrumpido por el usuario.")

# Cerrar conexión serial
if 'arduino_derecha' in globals() and arduino_derecha.is_open:
    arduino_derecha.close()
    print("Conexión cerrada.")


