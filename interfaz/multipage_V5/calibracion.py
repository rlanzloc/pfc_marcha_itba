# calibracion.py
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(BASE_DIR, "calibracion_individual")

sensor_pie_list = [
    'Derecha_S1', 'Derecha_S2', 'Derecha_S3', 'Derecha_S4',
    'Derecha_S5', 'Derecha_S6', 'Derecha_S7', 'Derecha_S8',
    'Izquierda_S1', 'Izquierda_S2', 'Izquierda_S3', 'Izquierda_S4',
    'Izquierda_S5', 'Izquierda_S6', 'Izquierda_S7', 'Izquierda_S8'
]

xx_data = {}
yy_data = {}

for sensor_pie in sensor_pie_list:
    x_path = os.path.join(base_path, f"x_{sensor_pie}_SIN.csv")
    y_path = os.path.join(base_path, f"y_{sensor_pie}_SIN.csv")
    xx_data[sensor_pie] = pd.read_csv(x_path, header=None).values.flatten()
    yy_data[sensor_pie] = pd.read_csv(y_path, header=None).values.flatten()

