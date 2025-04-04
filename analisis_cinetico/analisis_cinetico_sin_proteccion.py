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

plt.ioff()

############## CARGA DE DATOS ###################3
c1_pathx = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/clusters_sin/x_Cluster1_SIN.csv"
c2_pathx = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/clusters_sin/x_Cluster2_SIN.csv"
c1_pathy = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/clusters_sin/y_Cluster1_SIN.csv"
c2_pathy = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/clusters_sin/y_Cluster2_SIN.csv"

# Lee el archivo CSV y carga los parámetros
xx_1 = pd.read_csv(c1_pathx, header=None).values.flatten()
xx_2 = pd.read_csv(c2_pathx, header=None).values.flatten()
yy_1 = pd.read_csv(c1_pathy, header=None).values.flatten()
yy_2 = pd.read_csv(c2_pathy, header=None).values.flatten()

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

# Filtrar los DataFrames en listas separadas
raw_izq = [df for name, df in zip(variables, dfs) if "izquierda" in name.lower()]
raw_der = [df for name, df in zip(variables, dfs) if "derecha" in name.lower()]

# Obtener los nombres en el mismo orden que los DataFrames filtrados
nombres_izq = [name for name in variables if "izquierda" in name.lower()]
nombres_der = [name for name in variables if "derecha" in name.lower()]


################ PROCESAMIENTO DE DATOS #######################

def procesar_plantillas(datos_derecha, datos_izquierda, xx_1, xx_2, yy_1, yy_2):
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

    def preproc_df(dataframes):
        # Iterar sobre cada DataFrame en la lista
        for df in dataframes:
            # Crear una copia del DataFrame para no modificar el original
            # Normalizar el primer valor de Tiempo a 0
            # Restar el primer valor para que el primer tiempo sea 0
            df['Tiempo'] = (df['Tiempo'] - df['Tiempo'].iloc[0])/1000
        
        processed_dataframes = []
        mV_dataframes = []

        cluster_1 = ['Derecha_S1', 'Derecha_S2', 'Derecha_S3', 'Derecha_S4', 'Derecha_S5', 'Derecha_S6', 'Derecha_S8']
        cluster_2 = ['Derecha_S7', 'Izquierda_S1', 'Izquierda_S2', 'Izquierda_S3', 'Izquierda_S4', 'Izquierda_S5', 'Izquierda_S6', 'Izquierda_S7','Izquierda_S8']

        yy_1_dict = dict(zip(xx_1, yy_1))
        yy_2_dict = dict(zip(xx_2, yy_2))

        for df in dataframes:
            df_copy = df.copy()
            df_copy.set_index('Tiempo', inplace=True)

            # Corregir valores fuera de rango
            def correct_out_of_range(series):
                return series.where((series >= 0) & (series <= 1023), None).interpolate(limit_direction='both')

            df_copy = df_copy.apply(correct_out_of_range)

            # Convertir a mV correctamente
            df_mV = df_copy.map(lambda x: int((x / 1023) * 5000) if pd.notnull(x) else 0)
            mV_dataframes.append(df_mV)

            # Procesar columnas según el cluster
            df_processed = df_mV.copy()
            for column in df_processed.columns:
                if column in cluster_1:
                    df_processed[column] = df_processed[column].map(lambda x: yy_1_dict.get(x, 0))
                elif column in cluster_2:
                    df_processed[column] = df_processed[column].map(lambda x: yy_2_dict.get(x, 0))
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

    dataframes_der, mV_der = preproc_df(raw_der_final)  # DataFrames de la derecha
    dataframes_izq, mV_izq = preproc_df(raw_izq_final)

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
        
   
        
    return filt_der, filt_izq, sums_der, sums_izq, raw_der_proc, raw_izq_proc, raw_der_final, raw_izq_final, mV_der, mV_izq

filt_der, filt_izq, sums_der, sums_izq, raw_der_proc, raw_izq_proc, raw_der_final, raw_izq_final, mV_der, mV_izq = procesar_plantillas(raw_der, raw_izq, xx_1, xx_2, yy_1, yy_2)


# ==================================================
# GRÁFICOS DE mV
# ==================================================

# Pie DERECHO - mV
if len(mV_der) == 0:
    print("No hay datos de PIE DERECHO (mV) para graficar.")
else:
    fig_der_mv, axes = plt.subplots(len(mV_der), 1, figsize=(12, 2.5 * len(mV_der)))
    if len(mV_der) == 1:
        axes = [axes]
    
    for i, df in enumerate(mV_der):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie DERECHO en mV - Pasada {i+1}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
   

# Pie IZQUIERDO - mV
if len(mV_izq) == 0:
    print("No hay datos de PIE IZQUIERDO (mV) para graficar.")
else:
    fig_izq_mv, axes = plt.subplots(len(mV_izq), 1, figsize=(12, 2.5 * len(mV_izq)))
    if len(mV_izq) == 1:
        axes = [axes]
    
    for i, df in enumerate(mV_izq):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie IZQUIERDO en mV - Pasada {i+1}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
   


# ==================================================
# GRÁFICOS DE FUERZA (KG)
# ==================================================

# Pie DERECHO - KG
if len(filt_der) == 0:
    print("No hay datos de PIE DERECHO (KG) para graficar.")
else:
    fig_der_kg, axes = plt.subplots(len(filt_der), 1, figsize=(12, 2.5 * len(filt_der)))
    if len(filt_der) == 1:
        axes = [axes]
    
    for i, df in enumerate(filt_der):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie DERECHO en KG - Pasada {i+1}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
   

# Pie IZQUIERDO - KG
if len(filt_izq) == 0:
    print("No hay datos de PIE IZQUIERDO (KG) para graficar.")
else:
    fig_izq_kg, axes = plt.subplots(len(filt_izq), 1, figsize=(12, 2.5 * len(filt_izq)))
    if len(filt_izq) == 1:
        axes = [axes]
    
    for i, df in enumerate(filt_izq):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie IZQUIERDO en KG - Pasada {i+1}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
 

##################### CÁLCULO DE % BW CON MEDICIONES APOYANDO UN SOLO PIE #######################


def calculo_bw(sums_der, sums_izq):
    # Iterar sobre los DataFrames en sums_der
    for i, df in enumerate(sums_der):
        if i == len(sums_der) - 1:
            prom_der_3 = df[12.00:20.00].mean()  # BW derecho

        if i == len(sums_der) - 2:
            prom_der_2 = df[6.00:14.00].mean()  # BW derecho


    # Iterar sobre los DataFrames en sums_izq
    for i, df in enumerate(sums_izq):
        if i == len(sums_izq) - 1:
            prom_izq_3 = df[10.00:18.00].mean()  # BW derecho

        if i == len(sums_izq) - 2:
            prom_izq_2 = df[2.00:10.00].mean()  # BW derecho
    grf_der = []
    grf_izq = []
    
    print(prom_der_2, prom_der_3)
    print(prom_izq_2, prom_izq_3)
    
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
    else:
        # Si es un solo DataFrame, aplica el filtrado directamente
        sums_subset = sums[(sums.index >= t_inicio) & (sums.index <= t_fin)]
    
    return sums_subset

for i, (sum_der, sum_izq) in enumerate(zip(sums_der, sums_izq)):
    tf_der = sum_der.index[-1]  # Último tiempo de sum_der
    tf_izq = sum_izq.index[-1]  # Último tiempo de sum_izq
    
    dif = tf_der - tf_izq  # Diferencia entre los últimos tiempos
    sum_der.index = sum_der.index - dif  # Ajusta los índices de sum_der
    ti = sum_izq.index[0]  # Primer tiempo de sum_izq
    
    # Aplica subset y actualiza las listas originales
    sums_der[i] = subset(sum_der, ti, tf_izq)  # Filtra sum_der y actualiza sums_der
    sums_izq[i] = subset(sum_izq, ti, tf_izq)  # Filtra sum_izq y actualiza sums_izq)

BW_1_pie_der = subset(sums_der[0], 25.00, 35.00)
BW_1_pie_izq = subset(sums_izq[0], 10.00, 20.00)
BW_2_pies_der_1 = subset(sums_der[1], 10.00, 20.00)
BW_2_pies_izq_1 = subset(sums_izq[1], 10.00, 20.00)
BW_2_pies_der_2 = subset(sums_der[2], 15.00, 25.00)
BW_2_pies_izq_2 = subset(sums_izq[2], 15.00, 25.00)
BW_2_pies_der_3 = subset(sums_der[3], 20.00, 30.00)
BW_2_pies_izq_3 = subset(sums_izq[3], 20.00, 30.00)

pasada_der_2 = subset(sums_der[4], 15.00, 26.00)
pasada_izq_2 = subset(sums_izq[4], 15.00, 26.00)
pasada_der_3 = subset(sums_der[5], 20.00, 31.00)
pasada_izq_3 = subset(sums_izq[5], 20.00, 31.00)

sums_der_subset = [pasada_der_2, pasada_der_3]
sums_izq_subset = [pasada_izq_2, pasada_izq_3]

BW_der_list = [BW_1_pie_der, BW_2_pies_der_1, BW_2_pies_der_2, BW_2_pies_der_3]
BW_izq_list = [BW_1_pie_izq, BW_2_pies_izq_1, BW_2_pies_izq_2, BW_2_pies_izq_3] 

print(BW_der_list[1].mean(), BW_der_list[2].mean(), BW_der_list[3].mean())
print(BW_izq_list[1].mean(), BW_izq_list[2].mean(), BW_izq_list[3].mean())

###################### CALCULO SUMS NORMALIZADO POR BW #####################
grf_der, grf_izq = calculo_bw(sums_der, sums_izq)   

# Definir cantidad de pasadas
num_pasadas = max(len(sums_der), len(sums_izq))

# Evitar error si no hay datos
if num_pasadas == 0:
    print("No data to plot.")
else:
    # Crear figura y subplots (una fila por pasada, dos señales en cada subplot)
    fig, axes = plt.subplots(nrows=num_pasadas, ncols=1, figsize=(10, num_pasadas * 3), squeeze=False)
    axes = axes.flatten()  # Convertir en un array 1D para acceso fácil

    # Iterar sobre las pasadas
    for i in range(num_pasadas):
        if i < len(sums_der):
            sums_der[i].plot(ax=axes[i], label="Derecha", color='b')

        if i < len(sums_izq):
            sums_izq[i].plot(ax=axes[i], label="Izquierda", color='g')

        axes[i].set_title(f'Suma de fuerzas (N) - Pasada N°{i+1}')
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valores')
        #axes[i].set_xlim(14, 24)  # Ajusta si es necesario
        axes[i].legend()

    plt.tight_layout()

# Definir cantidad de pasadas
num_pasadas = max(len(grf_der), len(grf_izq))

# Evitar error si no hay datos
if num_pasadas == 0:
    print("No data to plot.")
else:
    # Crear figura y subplots (una fila por pasada, dos señales en cada subplot)
    fig, axes = plt.subplots(nrows=num_pasadas, ncols=1, figsize=(10, num_pasadas * 3), squeeze=False)
    axes = axes.flatten()  # Convertir en un array 1D para acceso fácil

    # Iterar sobre las pasadas
    for i in range(num_pasadas):
        if i < len(grf_der):
            grf_der[i].plot(ax=axes[i], label="Derecha", color='b')

        if i < len(grf_izq):
            grf_izq[i].plot(ax=axes[i], label="Izquierda", color='g')

        # Agregar línea horizontal en 1
        axes[i].axhline(y=1, color='r', linestyle='--', label='Referencia BW')

        axes[i].set_title(f'Suma de fuerzas (N) - Pasada N°{i+1}')
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valores')
        #axes[i].set_xlim(14, 24)  # Ajusta si es necesario
        axes[i].legend()

    plt.tight_layout()

from analisis_cinetico_BW_calibracion_indiv import sums_der_subset as sums_der_indiv_subset, sums_izq_subset as sums_izq_indiv_subset

# Definir cantidad de pasadas
num_pasadas = max(len(sums_der_subset), len(sums_izq_subset))

# Evitar error si no hay datos
if num_pasadas == 0:
    print("No data to plot.")
else:
    # Crear figura para la derecha
    fig_der, axes_der = plt.subplots(nrows=num_pasadas, ncols=1, figsize=(10, num_pasadas * 3), squeeze=False)
    axes_der = axes_der.flatten()
    
    # Crear figura para la izquierda
    fig_izq, axes_izq = plt.subplots(nrows=num_pasadas, ncols=1, figsize=(10, num_pasadas * 3), squeeze=False)
    axes_izq = axes_izq.flatten()

    # Iterar sobre las pasadas
    for i in range(num_pasadas):
        # Gráfico de la derecha
        if i < len(sums_der_subset):
            sums_der_subset[i].plot(ax=axes_der[i], label="Derecha_Cluster", color='orange')
            sums_der_indiv_subset[i].plot(ax=axes_der[i], label="Derecha_Indiv", color='r')
            axes_der[i].set_title(f'Suma de fuerzas Derecha (N) - Pasada N°{i+1}')
            axes_der[i].set_xlabel('Tiempo')
            axes_der[i].set_ylabel('Valores')
            axes_der[i].legend()
        
        # Gráfico de la izquierda
        if i < len(sums_izq_subset):
            sums_izq_subset[i].plot(ax=axes_izq[i], label="Izquierda_Cluster", color='g')
            sums_izq_indiv_subset[i].plot(ax=axes_izq[i], label="Izquierda_Indiv", color='b')
            axes_izq[i].set_title(f'Suma de fuerzas Izquierda (N) - Pasada N°{i+1}')
            axes_izq[i].set_xlabel('Tiempo')
            axes_izq[i].set_ylabel('Valores')
            axes_izq[i].legend()

    plt.tight_layout()


# Configuración
sensor_numbers = range(1, 9)  # Sensores del 1 al 8
dfs_derecha = filt_der[-2:]  # Últimos 2 DataFrames del pie derecho
dfs_izquierda = filt_izq[-2:]  # Últimos 2 DataFrames del pie izquierdo

# Función para graficar un pie (derecho o izquierdo)
def plot_pie_sensors(pie_type, dfs, figsize=(15, 12)):
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(f'Comparación de sensores - Pie {pie_type} (Últimos 2 pasos)', fontsize=16)
    axs = axs.flatten()
    
    for i, sensor_num in enumerate(sensor_numbers):
        col_name = f'{pie_type}_S{sensor_num}'  # Ej: "Derecha_S1"
        for j, df in enumerate(dfs):
            axs[i].plot(df[col_name], label=f'Pasada {j+2}')
        axs[i].set_title(f'Sensor {sensor_num}')
        axs[i].set_ylabel('Valor en kg')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Generar gráficos para ambos pies
plot_pie_sensors("Derecha", dfs_derecha)
plot_pie_sensors("Izquierda", dfs_izquierda)