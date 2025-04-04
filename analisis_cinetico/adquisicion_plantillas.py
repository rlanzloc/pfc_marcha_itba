import serial
import pandas as pd
from datetime import datetime
import threading

# Inicializar DataFrame con 8 sensores más una columna de tiempo para cada pie
df_izquierda = pd.DataFrame(columns=['Hora', 'Tiempo'] + [f"Izquierda_S{i+1}" for i in range(8)])
df_derecha = pd.DataFrame(columns=['Hora', 'Tiempo'] + [f"Derecha_S{i+1}" for i in range(8)])

# Inicializar Arduinos (cambiar COM4 y COM6 si es necesario)
try:
    arduino_derecha = serial.Serial('COM3', 115200, timeout=1)
    arduino_derecha.flushInput()
    arduino_derecha.flushOutput()
    print("Conexión establecida con Arduino derecha")
except serial.SerialException:
    print("Error al conectar con Arduino derecha")

try:
    arduino_izquierda = serial.Serial('COM8', 115200, timeout=1)
    arduino_izquierda.flushInput()
    arduino_izquierda.flushOutput()
    print("Conexión establecida con Arduino izquierda")
except serial.SerialException:
    print("Error al conectar con Arduino izquierda")

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
                    hora_actual = datetime.now().strftime('%H:%M:%S.%f')
                    tiempo_arduino = datos[0]
                    
                    # Obtener valores de sensores
                    valores_sensores = [int(valor) for valor in datos[1:9]]  # Sensores 1-8
                    
                    # Invertir orden SOLO para el pie izquierdo
                    if pie == "Izquierda":
                        valores_sensores = valores_sensores[::-1]  # Invierte la lista
                    
                    # Crear fila como diccionario
                    nueva_fila = {'Hora': hora_actual, 'Tiempo': tiempo_arduino}
                    for i in range(8):
                        nueva_fila[f"{pie}_S{i+1}"] = valores_sensores[i]
                    
                    dataframe.loc[len(dataframe)] = nueva_fila
                    
                    # Imprimir datos (incluyendo el orden invertido si es izquierdo)
                    print(f"{pie} - Hora: {hora_actual}, Tiempo: {tiempo_arduino}, Sensores: {valores_sensores}")
                    
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
df_derecha.to_csv('datos_derecha_sin_proteccion_3.csv', index=False)
df_izquierda.to_csv('datos_izquierda_sin_proteccion_3.csv', index=False)
