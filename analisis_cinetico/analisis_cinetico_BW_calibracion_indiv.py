import pandas as pd
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.signal import argrelextrema
import scipy.signal as signal
from scipy.signal import find_peaks
from scipy.stats import pearsonr

############## CARGA DE DATOS ###################

# Definir la ruta base
base_path = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/calibracion_indiv_sin_10_04/"

# Lista de sensores y pies (ajusta esto según tus datos)
sensor_pie_list = [
    'Derecha_S1', 'Derecha_S2', 'Derecha_S3', 'Derecha_S4', 
    'Derecha_S5', 'Derecha_S6', 'Derecha_S7', 'Derecha_S8',
    'Izquierda_S1', 'Izquierda_S2', 'Izquierda_S3', 'Izquierda_S4', 
    'Izquierda_S5', 'Izquierda_S6', 'Izquierda_S7', 'Izquierda_S8'
]

# Diccionarios para almacenar los datos
xx_data = {}  # Almacenará los valores de x para cada sensor
yy_data = {}  # Almacenará los valores de y para cada sensor

# Leer los archivos CSV para cada sensor
for sensor_pie in sensor_pie_list:
    # Construir las rutas de los archivos
    x_path = f"{base_path}x_{sensor_pie}_SIN.csv"
    y_path = f"{base_path}y_{sensor_pie}_SIN.csv"

    # Leer los archivos CSV
    xx_data[sensor_pie] = pd.read_csv(x_path, header=None).values.flatten()
    yy_data[sensor_pie] = pd.read_csv(y_path, header=None).values.flatten()

# Carpeta donde están los archivos
folder_path = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/pasadas/pasadas_sin_proteccion"

# Función para detectar el delimitador
def detect_delimiter(file_path, sample_size=5):
    delimiters = [',', ';', '\t']
    with open(file_path, 'r') as f:
        sample = [f.readline() for _ in range(sample_size)]
    
    for delimiter in delimiters:
        counts = [line.count(delimiter) for line in sample]
        if all(count == counts[0] for count in counts) and counts[0] > 0:
            return delimiter
    return ','  # Por defecto si no se detecta

# Listar y leer archivos
dfs = []
archivos_csv = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

for f in archivos_csv:
    file_path = os.path.join(folder_path, f)
    delimiter = detect_delimiter(file_path)
    df = pd.read_csv(file_path, delimiter=delimiter)
    df = df.drop(columns=["Hora"], errors="ignore")
    dfs.append(df)

# Lista de nombres de archivos sin la extensión
variables = [os.path.splitext(f)[0] for f in archivos_csv]

print(variables)

# Filtrar los DataFrames en listas separadas
raw_izq = [df for name, df in zip(variables, dfs) if "izquierda" in name.lower()]
raw_der = [df for name, df in zip(variables, dfs) if "derecha" in name.lower()]


################ PROCESAMIENTO DE DATOS #######################

def procesar_plantillas(datos_derecha, datos_izquierda, xx_data, yy_data):
    def interp(df_list, frecuencia_hz=100):
        """
        Recorre una lista de DataFrames, chequea si los intervalos de tiempo son consistentes con la frecuencia deseada
        (por defecto 100 Hz) y aplica la interpolación si es necesario. Los valores de los sensores serán enteros.
        """
        # Lista para almacenar los DataFrames resultantes (copias)
        dfs_interpolados = []

        for df in df_list:
            # Crear una copia del DataFrame para evitar modificar el original
            df_copy = df.copy()

            # Verificar la diferencia de tiempos
            df_copy['Tiempo_diff'] = df_copy['Tiempo'].diff()

            # Chequear si todos los intervalos de tiempo son consistentes con la frecuencia deseada (100 Hz)
            intervalos_fuera_de_rango = df_copy[df_copy['Tiempo_diff']
                                                != 1 / frecuencia_hz]

            if len(intervalos_fuera_de_rango) > 0:
                # Si hay intervalos irregulares, aplicar interpolación
                # Crear un rango de tiempos con la frecuencia deseada (100 Hz)
                tiempo_inicial = df_copy['Tiempo'].iloc[0]
                tiempo_final = df_copy['Tiempo'].iloc[-1]

                # Crear un rango de tiempos que tiene la misma longitud que el DataFrame
                tiempos_nuevos = np.linspace(
                    tiempo_inicial, tiempo_final, len(df_copy))

                # Interpolar los valores para cada sensor
                df_interpolado = df_copy.copy()

                # Interpolación para las demás columnas
                for columna in df_copy.columns:
                    if columna != 'Tiempo':
                        # Realizar la interpolación lineal de los valores
                        df_interpolado[columna] = np.interp(
                            tiempos_nuevos, df_copy['Tiempo'], df_copy[columna])
                        
                        # Rellenar los posibles NaN resultantes de la interpolación con el valor anterior
                        df_interpolado[columna] = df_interpolado[columna].ffill()  # Rellenar hacia adelante
                        df_interpolado[columna] = df_interpolado[columna].fillna(0)  # Si aún hay NaN, rellenar con 0

                        # Asegurarse de que los valores sean finitos antes de convertir a enteros
                        df_interpolado[columna] = df_interpolado[columna].apply(
                            pd.to_numeric, errors='coerce')  # Convertir a numérico
                        df_interpolado[columna] = df_interpolado[columna].fillna(
                            0).astype(int)  # Convertir a enteros, rellenando NaN con 0

                # Actualizar la columna de 'Tiempo' con los nuevos tiempos
                df_interpolado['Tiempo'] = tiempos_nuevos

                # Eliminar la columna 'Tiempo_diff'
                df_interpolado.drop(columns=['Tiempo_diff'], inplace=True)

                # Agregar el DataFrame interpolado a la lista
                dfs_interpolados.append(df_interpolado)
            else:
                # Si no hay intervalos fuera de rango, añadir el DataFrame tal como está
                dfs_interpolados.append(df_copy)

        return dfs_interpolados

    def preproc_df(dataframes, xx_data, yy_data, R):
          # Iterar sobre cada DataFrame en la lista
        for df in dataframes:
            # Crear una copia del DataFrame para no modificar el original
            # Normalizar el primer valor de Tiempo a 0
            # Restar el primer valor para que el primer tiempo sea 0
            df['Tiempo'] = (df['Tiempo'] - df['Tiempo'].iloc[0])/1000
        
        processed_dataframes = []
        mV_dataframes = []

        # Crear diccionarios de calibración para cada sensor
        calibration_dicts = {}
        for sensor_pie in xx_data.keys():
            calibration_dicts[sensor_pie] = dict(zip(xx_data[sensor_pie], yy_data[sensor_pie]))

        for df in dataframes:
            df_copy = df.copy()
            df_copy.set_index('Tiempo', inplace=True)

            # Corregir valores fuera de rango
            def correct_out_of_range(series):
                return series.where((series >= 0) & (series <= 1023), None).interpolate(limit_direction='both')

            df_copy = df_copy.apply(correct_out_of_range)

            # Convertir a mV correctamente
            df_mV = df_copy.map(lambda x: 1 if (x == 0 or pd.isnull(x)) else int((x / 1023) * 5000))
            mV_dataframes.append(df_mV)
            df_processed = df_mV.copy()
            
            # Procesar columnas según el sensor
            for column in df_processed.columns:
                if column in calibration_dicts:
                    df_processed[column] = df_processed[column].map(lambda x: calibration_dicts[column].get(x, 0))
            processed_dataframes.append(df_processed)
        return processed_dataframes, mV_dataframes
    
    def limpiar_valores_anomalos(df_list, valor_maximo=1023):
        """
        Limpia valores mayores al valor máximo permitido en las columnas distintas de 'Tiempo'
        para cada DataFrame en una lista.
        """
        dfs_limpios = []

        for df in df_list:
            # Filtrar las columnas que no se llaman 'Tiempo'
            columnas_sensores = [col for col in df.columns if col != 'Tiempo']

            # Sustituir valores mayores al límite por NaN solo en columnas distintas de 'Tiempo'
            df_copy = df.copy()
            df_copy[columnas_sensores] = df_copy[columnas_sensores].where(df_copy[columnas_sensores] <= valor_maximo, np.nan)

            dfs_limpios.append(df_copy)

        return dfs_limpios
    
    # Función para aplicar un filtro pasa bajos
    def apply_lowpass_filter(data, cutoff_freq, sampling_rate):
        # Definir el filtro pasa bajos
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)

        # Aplicar el filtro
        filtered_data = filtfilt(b, a, data)

        # Establecer un umbral mínimo para evitar valores negativos
        filtered_data = np.maximum(filtered_data, 0)  # Asegura que los valores no sean negativos

        return filtered_data
    
    # Aplicar las funciones a los DataFrames de las listas raw_der y raw_izq
    raw_der_proc = limpiar_valores_anomalos(raw_der)
    raw_izq_proc = limpiar_valores_anomalos(raw_izq)

    # Aplicar las funciones a los DataFrames de las listas raw_der y raw_izq
    raw_der_final = interp(raw_der_proc, frecuencia_hz=100)
    raw_izq_final = interp(raw_izq_proc, frecuencia_hz=100)

    names_der = ['Derecha_S1', 'Derecha_S2', 'Derecha_S3', 'Derecha_S4', 'Derecha_S5', 'Derecha_S6', 'Derecha_S7', 'Derecha_S8']
    names_izq = ['Izquierda_S1', 'Izquierda_S2', 'Izquierda_S3', 'Izquierda_S4', 'Izquierda_S5', 'Izquierda_S6', 'Izquierda_S7','Izquierda_S8']

    dataframes_der, mV_der = preproc_df(raw_der_final, xx_data, yy_data, 3300)  # DataFrames de la derecha
    dataframes_izq, mV_izq = preproc_df(raw_izq_final, xx_data, yy_data, 10000)

    sampling_rate = 100  # Frecuencia de muestreo en Hz
    cutoff_frequency = 20  # Frecuencia de corte del filtro pasa bajos en Hz

    lowpass_der = []
    lowpass_izq = []

    # Aplicar filtro a los DataFrames de la derecha
    for i, df in enumerate(dataframes_der):
        df_filtered = df.copy()  # Crear una copia del DataFrame para almacenar los datos filtrados

        # Aplicar el filtro a cada columna de sensores
        for column in df.columns:
            df_filtered[column] = apply_lowpass_filter(df[column], cutoff_frequency, sampling_rate)

        lowpass_der.append(df_filtered)

    # Aplicar filtro a los DataFrames de la izquierda
    for i, df in enumerate(dataframes_izq):
        df_filtered = df.copy()  # Crear una copia del DataFrame para almacenar los datos filtrados

        # Aplicar el filtro a cada columna de sensores
        for column in df.columns:
            df_filtered[column] = apply_lowpass_filter(df[column], cutoff_frequency, sampling_rate)

        lowpass_izq.append(df_filtered)

    # Parámetros del filtro Savitzky-Golay
    window_length = 11  # Tamaño de la ventana del filtro (debe ser un número impar)
    polyorder = 3       # Orden del polinomio usado en el filtro

    filt_der = []
    filt_izq = []

    for i, df in enumerate(dataframes_der):
        df_filtered = df.copy()  # Crear una copia del DataFrame para almacenar los datos filtrados

        # Aplicar el filtro a cada columna de sensores
        for column in df.columns:
            df_filtered[column] = savgol_filter(df_filtered[column], window_length=window_length, polyorder=polyorder)
            # Asegurarse de que no haya valores negativos
            df_filtered[column] = np.maximum(df_filtered[column], 0)  # Corregir a 0 cualquier valor negativo

        filt_der.append(df_filtered)

    for i, df in enumerate(dataframes_izq):
        df_filtered = df.copy()  # Crear una copia del DataFrame para almacenar los datos filtrados

        # Aplicar el filtro a cada columna de sensores
        for column in df.columns:
            df_filtered[column] = savgol_filter(df_filtered[column], window_length=window_length, polyorder=polyorder)
            # Asegurarse de que no haya valores negativos
            df_filtered[column] = np.maximum(df_filtered[column], 0)  # Corregir a 0 cualquier valor negativo

        filt_izq.append(df_filtered)

    # Lista para almacenar las sumas de cada DataFrame
    sums_der = []
    sums_izq = []

    # Iterar sobre los DataFrames en la lista filt_der
    for filtered_df in filt_der:
        # Sumar todas las columnas filtradas
        sum_der = filtered_df.sum(axis=1)  # Suma a lo largo de las columnas (eje=1)

        # Agregar el array resultante a la lista sumas_der
        sums_der.append(sum_der)

    # Iterar sobre los DataFrames en la lista filt_der
    for filtered_df in filt_izq:
        # Sumar todas las columnas filtradas
        sum_izq = filtered_df.sum(axis=1)  # Suma a lo largo de las columnas (eje=1)

        # Agregar el array resultante a la lista sumas_der
        sums_izq.append(sum_izq)
    return filt_der, filt_izq, sums_der, sums_izq, dataframes_der, dataframes_izq, mV_der, mV_izq, raw_der_final, raw_izq_final

filt_der, filt_izq, sums_der, sums_izq, dataframes_der, dataframes_izq, mV_der, mV_izq, raw_der_final, raw_izq_final = procesar_plantillas(raw_der, raw_izq, xx_data, yy_data)


######### CORRECCION DE DESFASAJE ENTRE IZQ Y DER ###########

def subset(sums, t_inicio, t_fin):
    """
    Filtra los DataFrames de sums según un rango de tiempo.
    
    Parámetros:
    - sums: Lista de DataFrames o un solo DataFrame.
    - t_inicio: Valor mínimo de tiempo para el filtro.
    - t_fin: Valor máximo de tiempo para el filtro.
    
    Retorna:
    - sums_filtrado: Lista de DataFrames filtrados o un solo DataFrame filtrado.
    """
    # Verifica si sums es una lista de DataFrames o un solo DataFrame
    if isinstance(sums, list):
        # Si es una lista, aplica el filtrado a cada DataFrame en la lista
        sums_subset = [df[(df.index >= t_inicio) & (df.index <= t_fin)] for df in sums]
        sums_subset = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in sums_subset]
    else:
        # Si es un solo DataFrame, aplica el filtrado directamente
        sums_subset = sums[(sums.index >= t_inicio) & (sums.index <= t_fin)]

    return sums_subset

for i, (sum_der, sum_izq, filt_d, filt_i) in enumerate(zip(sums_der, sums_izq, filt_der, filt_izq)):
    tf_der = sum_der.index[-1]
    tf_izq = sum_izq.index[-1]
    
    
    dif = tf_der - tf_izq
    
    sum_der = sum_der.copy()
    filt_d = filt_d.copy()
    sum_der.index = sum_der.index - dif
    filt_d.index = filt_d.index - dif

    ti = sum_izq.index[0]
    
    # Filtrar tanto suma como datos crudos
    sums_der[i] = subset(sum_der, ti, tf_izq)
    sums_izq[i] = subset(sum_izq, ti, tf_izq)
    filt_der[i] = subset(filt_d, ti, tf_izq)  
    filt_izq[i] = subset(filt_i, ti, tf_izq)

# Define cuántos/qué DataFrames quieres visualizar (ejemplo con las últimas 3 pasadas)
filt_der_2 = filt_der # Esto selecciona los últimos 3 DataFrames de filt_der

# Pie DERECHO - KG
if len(filt_der_2) == 0:
    print("No hay datos de PIE DERECHO (KG) para graficar.")
else:
    fig_der_kg, axes = plt.subplots(len(filt_der_2), 1, figsize=(12, 2.5 * len(filt_der_2)))
    if len(filt_der_2) == 1:
        axes = [axes]  # Convertir en lista si solo hay un gráfico
    
    for i, df in enumerate(filt_der_2):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie DERECHO en KG - Pasada {i+1}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    
# Define cuántos/qué DataFrames quieres visualizar (ejemplo con las últimas 3 pasadas)
filt_izq_2 = filt_izq  # Esto selecciona los últimos 2 DataFrames de filt_izq

# Pie IZQUIERDO - KG
if len(filt_izq_2) == 0:
    print("No hay datos de PIE IZQUIERDO (KG) para graficar.")
else:
    fig_izq_kg, axes = plt.subplots(len(filt_izq_2), 1, figsize=(12, 2.5 * len(filt_izq_2)))
    if len(filt_izq_2) == 1:
        axes = [axes]  # Convertir en lista si solo hay un gráfico
    
    for i, df in enumerate(filt_izq_2):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie IZQUIERDO en KG - Pasada {i+1}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    
sums_der_last2 = sums_der

# Evitar error si no hay datos
if not sums_der_last2:
    print("No hay datos para graficar (sums_der tiene menos de 2 pasadas).")
else:
    # Crear figura y subplots (una fila por pasada)
    fig, axes = plt.subplots(
        nrows=len(sums_der_last2),  # Número de filas = número de pasadas (2 o menos)
        ncols=1,
        figsize=(10, len(sums_der_last2) * 3),  # Ajustar altura según pasadas
        squeeze=False
    )
    axes = axes.flatten()  # Convertir en array 1D para acceso fácil

    # Iterar sobre las últimas 2 pasadas
    for i, pasada in enumerate(sums_der_last2):
        pasada.plot(ax=axes[i], label="Derecha", color='b')
        axes[i].set_title(f'Suma de fuerzas (N) - Pasada N°{i + 1}')  # Numeración desde 1
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valores')
        axes[i].legend()

    plt.tight_layout()
    
    
sums_izq_last2 = sums_izq

# Evitar error si no hay datos
if not sums_izq_last2:
    print("No hay datos para graficar (sums_der tiene menos de 2 pasadas).")
else:
    # Crear figura y subplots (una fila por pasada)
    fig, axes = plt.subplots(
        nrows=len(sums_izq_last2),  # Número de filas = número de pasadas (2 o menos)
        ncols=1,
        figsize=(10, len(sums_izq_last2) * 3),  # Ajustar altura según pasadas
        squeeze=False
    )
    axes = axes.flatten()  # Convertir en array 1D para acceso fácil

    # Iterar sobre las últimas 2 pasadas
    for i, pasada in enumerate(sums_izq_last2):
        pasada.plot(ax=axes[i], label="Izquierda", color='g')
        axes[i].set_title(f'Suma de fuerzas (N) - Pasada N°{i + 1}')  # Numeración desde 1
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valores')
        axes[i].legend()

    plt.tight_layout()
    plt.show()
    


def calculo_bw(sums_der, sums_izq):

    grf_der = []
    grf_izq = []
    
    BW = 51

    # Iterar a través de los dataframes y los valores de BW_med correspondientes
    for df in sums_der:
        # Calcular el porcentaje de GRF respecto al BW correspondiente
        porcentaje_der = df / BW

        # Añadir a la lista de porcentajes
        grf_der.append(porcentaje_der)
        
    for df in sums_izq:
        # Calcular el porcentaje de GRF respecto al BW correspondiente
        porcentaje_izq = df / BW

        # Añadir a la lista de porcentajes
        grf_izq.append(porcentaje_izq)
    
    return grf_der, grf_izq

grf_der, grf_izq = calculo_bw(sums_der, sums_izq)

# Definir cantidad de pasadas
num_pasadas = max(len(grf_der), len(grf_izq))

# Evitar error si no hay datos
if num_pasadas == 0:
    print("No data to plot.")
else:
    # Crear figura con subplots separados (2 columnas: izquierda y derecha)
    fig, axes = plt.subplots(nrows=num_pasadas, ncols=2, figsize=(15, num_pasadas * 3), squeeze=False)
    
    # Iterar sobre las pasadas
    for i in range(num_pasadas):
        # --- Gráfico DERECHO (columna 0) ---
        if i < len(grf_der):
            grf_der[i].plot(ax=axes[i, 0], color='b')
            axes[i, 0].axhline(y=1, color='r', linestyle='--', label='Referencia 1N')
            axes[i, 0].set_title(f'GRF Derecha - Pasada N°{i+1}')
            axes[i, 0].set_ylabel(f'%BW')  # Leyenda personalizada
            axes[i, 0].legend()
        
        # --- Gráfico IZQUIERDO (columna 1) ---
        if i < len(grf_izq):
            grf_izq[i].plot(ax=axes[i, 1], color='g')
            axes[i, 1].axhline(y=1, color='r', linestyle='--', label='Referencia 1N')
            axes[i, 1].set_title(f'GRF Izquierda - Pasada N°{i+1}')
            axes[i, 1].set_ylabel(f'%BW')  # Leyenda personalizada
            axes[i, 1].legend()

    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()


from analisis_plantillas_wip import sums_der as sums_der_actual, sums_izq as sums_izq_actual, filt_der as filt_der_actual, filt_izq as filt_izq_actual 


# Asegurate de que ambas listas tengan la misma cantidad de pasadas
num_pasadas = min(len(filt_der), len(filt_der_actual))

# Iterar por pasada
for i in range(num_pasadas):
    df_ant = filt_der[i]
    df_act = filt_der_actual[i]
    
    sensores = [col for col in df_ant.columns if col != 'Tiempo']
    
    fig, axes = plt.subplots(8, 1, figsize=(12, 20))  # 8 sensores → 8 subplots
    fig.suptitle(f'Comparación PIE DERECHO - Pasada {i+1}', fontsize=16, y=1.02)
    
    for j, sensor in enumerate(sensores):
        axes[j].plot(df_ant.index, df_ant[sensor], label='Procesado anterior', color='tab:blue')
        axes[j].plot(df_act.index, df_act[sensor], label='Procesado actual', color='tab:orange', linestyle='--')
        axes[j].set_title(f'Sensor {sensor}')
        axes[j].legend()
        axes[j].set_ylabel('KG')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    
    
    
    # Para pie izquierdo
num_pasadas_izq = min(len(filt_izq), len(filt_izq_actual))

for i in range(num_pasadas_izq):
    df_ant = filt_izq[i]
    df_act = filt_izq_actual[i]
    
    sensores = [col for col in df_ant.columns if col != 'Tiempo']
    
    fig, axes = plt.subplots(8, 1, figsize=(12, 20))  # 8 sensores
    fig.suptitle(f'Comparación PIE IZQUIERDO - Pasada {i+1}', fontsize=16, y=1.02)
    
    for j, sensor in enumerate(sensores):
        axes[j].plot(df_ant.index, df_ant[sensor], label='Procesado anterior', color='tab:blue')
        axes[j].plot(df_act.index, df_act[sensor], label='Procesado actual', color='tab:orange', linestyle='--')
        axes[j].set_title(f'Sensor {sensor}')
        axes[j].legend()
        axes[j].set_ylabel('KG')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    plt.show()


# Asegurar que ambas listas tengan la misma cantidad de pasadas
num_pasadas = min(len(sums_der), len(sums_der_actual))

# Evitar errores si no hay datos
if num_pasadas == 0:
    print("No hay datos para comparar sums_der.")
else:
    fig, axes = plt.subplots(
        nrows=num_pasadas,
        ncols=1,
        figsize=(10, num_pasadas * 3),
        squeeze=False
    )
    axes = axes.flatten()

    for i in range(num_pasadas):
        df_ant = sums_der[i]
        df_act = sums_der_actual[i]

        # Asegurar formato DataFrame
        df_ant = df_ant if isinstance(df_ant, pd.DataFrame) else df_ant.to_frame(name='suma')
        df_act = df_act if isinstance(df_act, pd.DataFrame) else df_act.to_frame(name='suma')

        df_ant.plot(ax=axes[i], label='Anterior', color='tab:blue')
        df_act.plot(ax=axes[i], label='Actual', color='tab:orange', linestyle='--')

        axes[i].set_title(f'Comparación SUMA Pie Derecho - Pasada {i+1}')
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Fuerza total (Kg)')
        axes[i].legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()



num_pasadas = min(len(sums_izq), len(sums_izq_actual))

if num_pasadas == 0:
    print("No hay datos para comparar sums_izq.")
else:
    fig, axes = plt.subplots(
        nrows=num_pasadas,
        ncols=1,
        figsize=(10, num_pasadas * 3),
        squeeze=False
    )
    axes = axes.flatten()

    for i in range(num_pasadas):
        df_ant = sums_izq[i]
        df_act = sums_izq_actual[i]

        df_ant = df_ant if isinstance(df_ant, pd.DataFrame) else df_ant.to_frame(name='suma')
        df_act = df_act if isinstance(df_act, pd.DataFrame) else df_act.to_frame(name='suma')

        df_ant.plot(ax=axes[i], label='Anterior', color='tab:green')
        df_act.plot(ax=axes[i], label='Actual', color='tab:red', linestyle='--')

        axes[i].set_title(f'Comparación SUMA Pie Izquierdo - Pasada {i+1}')
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Fuerza total (Kg)')
        axes[i].legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
