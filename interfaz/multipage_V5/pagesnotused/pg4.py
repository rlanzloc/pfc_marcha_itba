import dash
from dash import dcc
from dash import dcc, html, callback, Output, Input
from dash import html
from dash import dash_table
from dash.dash_table.Format import Group
from dash.dependencies import Input, Output, State

import serial
import threading
import pandas as pd
import datetime


dash.register_page(__name__, name='Arduino read')
# Configuración de los módulos Bluetooth

bluetooth_serial_der = serial.Serial("COM3", 115200)  # Ajusta el puerto según sea necesario
# Configuración del módulo Bluetooth izquierdo (comentado para pruebas)
# bluetooth_serial_izq = serial.Serial("COM4", 9600)  # Ajusta el puerto según sea necesario

# Variables globales para almacenar los valores de los sensores y control de lectura
sensor_values_der = [0] * 8
# sensor_values_izq = [0] * 8  # Sensor izquierdo comentado para pruebas
reading = False
data_storage_der = []
# data_storage_izq = []  # Almacenamiento izquierdo comentado para pruebas
lock = threading.Lock()  # Lock para sincronización de datos

# Función para leer datos del Bluetooth derecho en un hilo separado
def read_bluetooth_der():
    global sensor_values_der, reading, data_storage_der
    first_line_received = False
    while True:
        if not first_line_received and bluetooth_serial_der.in_waiting > 0:
            try:
                data = bluetooth_serial_der.readline().decode('utf-8').strip()
                if data and len(data.split(',')) == 8:
                    first_line_received = True
                    print("Primera línea válida recibida. Iniciando lectura...")
            except ValueError:
                print(f"Error leyendo datos del Bluetooth derecho: datos no válidos '{data}'")
            except Exception as e:
                print(f"Error leyendo datos del Bluetooth derecho: {e}")
        
        if reading and bluetooth_serial_der.in_waiting > 0:
            try:
                data = bluetooth_serial_der.readline().decode('utf-8').strip()
                if data:  # Verifica que los datos no estén vacíos
                    values = list(map(int, data.split(',')))
                    if len(values) == 8:  # Aseguramos que se leen exactamente 8 valores
                        sensor_values_der = values
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Incluir milisegundos
                        with lock:
                            data_storage_der.append([timestamp] + sensor_values_der)
            except ValueError:
                print(f"Error leyendo datos del Bluetooth derecho: datos no válidos '{data}'")
            except Exception as e:
                print(f"Error leyendo datos del Bluetooth derecho: {e}")

# Función para leer datos del Bluetooth izquierdo en un hilo separado (comentado para pruebas)
# def read_bluetooth_izq():
#     global sensor_values_izq, reading, data_storage_izq
#     while True:
#         if reading and bluetooth_serial_izq.in_waiting > 0:
#             try:
#                 data = bluetooth_serial_izq.readline().decode('utf-8').strip()
#                 if data:  # Verifica que los datos no estén vacíos
#                     values = list(map(int, data.split(',')))
#                     if len(values) == 8:  # Aseguramos que se leen exactamente 8 valores
#                         sensor_values_izq = values
#                         timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Incluir milisegundos
#                         with lock:
#                             data_storage_izq.append([timestamp] + sensor_values_izq)
#             except ValueError:
#                 print(f"Error leyendo datos del Bluetooth izquierdo: datos no válidos '{data}'")
#             except Exception as e:
#                 print(f"Error leyendo datos del Bluetooth izquierdo: {e}")

# Iniciar los hilos de lectura de Bluetooth derecho (y izquierdo, comentado para pruebas)
thread_der = threading.Thread(target=read_bluetooth_der)
thread_der.daemon = True
thread_der.start()

# thread_izq = threading.Thread(target=read_bluetooth_izq)  # Hilo izquierdo comentado para pruebas
# thread_izq.daemon = True  # Hilo izquierdo comentado para pruebas
# thread_izq.start()  # Hilo izquierdo comentado para pruebas



layout = html.Div([
    html.H1("Lectura de Sensores"),
    html.Button('Iniciar Lectura', id='start-button', n_clicks=0),
    html.Button('Finalizar Lectura', id='stop-button', n_clicks=0),
    html.Div(id='output-state', children='Esperando acción...')
])

@callback(
    Output('output-state', 'children'),
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')]
)
def control_reading(start_clicks, stop_clicks):
    global reading, data_storage_der
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-button':
        reading = True
        return 'Leyendo...'

    elif button_id == 'stop-button':
        reading = False
        with lock:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
            filename_der = f'data_read_der_{timestamp}.xlsx'
            df_der = pd.DataFrame(data_storage_der, columns=["Timestamp"] + [f'Sensor {i+1} (Der)' for i in range(8)])
            df_der.to_excel(filename_der, index=False)

        return f'Lectura finalizada. Datos guardados en {filename_der}'

    return 'Esperando acción...'

@callback(
    Output('start-button', 'disabled'),
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')]
)
def disable_start_button(start_clicks, stop_clicks):
    return reading

@callback(
    Output('stop-button', 'disabled'),
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')]
)
def disable_stop_button(start_clicks, stop_clicks):
    return not reading

