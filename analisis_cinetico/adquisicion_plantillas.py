import serial
import pandas as pd
from datetime import datetime
import threading
import time

# Initialize DataFrames for both feet
df_izquierda = pd.DataFrame(columns=['Hora', 'Tiempo'] + [f"Izquierda_S{i+1}" for i in range(8)])
df_derecha = pd.DataFrame(columns=['Hora', 'Tiempo'] + [f"Derecha_S{i+1}" for i in range(8)])

# Initialize Arduino connections
arduino_derecha = None
arduino_izquierda = None

try:
    arduino_derecha = serial.Serial('COM6', 115200, timeout=1)
    arduino_derecha.flushInput()
    arduino_derecha.flushOutput()
    print("Conexión establecida con Arduino derecha (COM6)")
except serial.SerialException as e:
    print(f"Error al conectar con Arduino derecha: {e}")

try:
    arduino_izquierda = serial.Serial('COM3', 115200, timeout=1)
    arduino_izquierda.flushInput()
    arduino_izquierda.flushOutput()
    print("Conexión establecida con Arduino izquierda (COM3)")
except serial.SerialException as e:
    print(f"Error al conectar con Arduino izquierda: {e}")

# Control variable for reading loop
running = True

def leer_datos(arduino, dataframe, pie):
    """Improved function to read data from Arduino with proper exit handling"""
    global running
    while running:
        try:
            if arduino.in_waiting > 0:
                linea = arduino.readline().decode('utf-8').strip()
                if linea:
                    datos = linea.split(',')
                    if len(datos) == 9:
                        hora_actual = datetime.now().strftime('%H:%M:%S.%f')
                        tiempo_arduino = datos[0]
                        
                        # Get sensor values
                        valores_sensores = [int(valor) for valor in datos[1:9]]
                        
                        # Reverse order ONLY for left foot
                        if pie == "Izquierda":
                            valores_sensores = valores_sensores[::-1]
                        
                        # Create new row
                        nueva_fila = {'Hora': hora_actual, 'Tiempo': tiempo_arduino}
                        for i in range(8):
                            nueva_fila[f"{pie}_S{i+1}"] = valores_sensores[i]
                        
                        dataframe.loc[len(dataframe)] = nueva_fila
                        print(f"{pie} - Hora: {hora_actual}, Tiempo: {tiempo_arduino}, Sensores: {valores_sensores}")
            else:
                time.sleep(0.05)  # Small sleep to prevent CPU overuse
        except Exception as e:
            if running:  # Only print errors if we're still running
                print(f"Error al leer datos de {pie}: {e}")

# Create and start threads
threads = []

if arduino_derecha and arduino_derecha.is_open:
    hilo_derecha = threading.Thread(target=leer_datos, args=(arduino_derecha, df_derecha, "Derecha"))
    hilo_derecha.daemon = True  # Set as daemon thread to exit when main thread exits
    threads.append(hilo_derecha)
    hilo_derecha.start()
    print("Hilo de lectura para pie derecho iniciado")

if arduino_izquierda and arduino_izquierda.is_open:
    hilo_izquierda = threading.Thread(target=leer_datos, args=(arduino_izquierda, df_izquierda, "Izquierda"))
    hilo_izquierda.daemon = True  # Set as daemon thread
    threads.append(hilo_izquierda)
    hilo_izquierda.start()
    print("Hilo de lectura para pie izquierdo iniciado")

# Wait for ENTER press with timeout checks
try:
    input("Presiona ENTER para finalizar...\n")
except KeyboardInterrupt:
    print("\nInterrupción por teclado recibida")

running = False
print("Deteniendo hilos...")

# Wait for threads to finish (with timeout)
for thread in threads:
    thread.join(timeout=2.0)  # 2 second timeout

# Close serial connections
if arduino_derecha and arduino_derecha.is_open:
    arduino_derecha.close()
    print("Conexión con Arduino derecha cerrada")
if arduino_izquierda and arduino_izquierda.is_open:
    arduino_izquierda.close()
    print("Conexión con Arduino izquierda cerrada")

# Save data to CSV
try:
    df_derecha.to_csv('datos_derecha_S3_3.csv', index=False)
    df_izquierda.to_csv('datos_izquierda_S3_3.csv', index=False)
    print("Datos guardados exitosamente")
except Exception as e:
    print(f"Error al guardar archivos CSV: {e}")

print("Programa terminado")