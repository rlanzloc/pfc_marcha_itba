import serial
import pandas as pd
from datetime import datetime
import threading

# Inicializar DataFrame con 8 sensores más una columna de tiempo para cada pie
df_derecha = pd.DataFrame(columns=["Tiempo"] + [f"Sensor {i+1}" for i in range(9)])
df_izquierda = pd.DataFrame(columns=["Tiempo"] + [f"Sensor {i+1}" for i in range(9)])

# Inicializar Arduinos (cambiar COM4 y COM6 si es necesario)
try:
    arduino_derecha = serial.Serial('COM10', 115200, timeout=1)
    arduino_derecha.flushInput()
    arduino_derecha.flushOutput()
    print("Conexión establecida con Arduino derecha en COM4")
except serial.SerialException:
    print("Error al conectar con Arduino derecha en COM4")

try:
    arduino_izquierda = serial.Serial('COM7', 115200, timeout=1)
    arduino_izquierda.flushInput()
    arduino_izquierda.flushOutput()
    print("Conexión establecida con Arduino izquierda en COM6")
except serial.SerialException:
    print("Error al conectar con Arduino izquierda en COM6")

# Variable para controlar el bucle de lectura
running = True

def leer_datos(arduino, dataframe, pie):
    """Función para leer datos del Arduino y guardarlos en un DataFrame."""
    global running
    while running:
        if arduino.in_waiting > 0:
            try:
                linea = arduino.readline().decode('utf-8').strip()
                datos = linea.split(',')
                if len(datos) == 9:
                    tiempo = datetime.now().strftime('%H:%M:%S.%f')
                    fila = [tiempo] + [int(valor) for valor in datos]
                    dataframe.loc[len(dataframe)] = fila
                    print(f"{pie} - {fila}")
            except Exception as e:
                print(f"Error al leer datos de {pie}: {e}")

# Crear hilos para leer datos de ambos pies si las conexiones son exitosas
hilos = []

if 'arduino_derecha' in globals() and arduino_derecha.is_open:
    hilo_derecha = threading.Thread(target=leer_datos, args=(arduino_derecha, df_derecha, "Derecha"))
    hilos.append(hilo_derecha)
    hilo_derecha.start()
    print("Hilo de lectura para pie derecho iniciado")

if 'arduino_izquierda' in globals() and arduino_izquierda.is_open:
    hilo_izquierda = threading.Thread(target=leer_datos, args=(arduino_izquierda, df_izquierda, "Izquierda"))
    hilos.append(hilo_izquierda)
    hilo_izquierda.start()
    print("Hilo de lectura para pie izquierdo iniciado")


# Mantener el programa corriendo hasta que se presione Enter
input("Presiona ENTER para finalizar...\n")
running = False

# Esperar a que ambos hilos terminen
for hilo in hilos:
    hilo.join()

# Cerrar conexiones seriales
if 'arduino_derecha' in globals() and arduino_derecha.is_open:
    arduino_derecha.close()
if 'arduino_izquierda' in globals() and arduino_izquierda.is_open:
    arduino_izquierda.close()

# Guardar los datos en archivos CSV
df_derecha.to_csv('datos_derecha_PESO.csv', index=False)
df_izquierda.to_csv('datos_izquierda_PESO.csv', index=False)
