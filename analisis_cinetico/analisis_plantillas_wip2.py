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
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d


plt.ioff() #si pones solo un plt.show() al final del codigo se abren todas las imagenes juntas

# Definir la ruta base
base_path = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/calibracion_indiv_sin/"

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

# Filtrar los DataFrames en listas separadas
raw_izq = [df for name, df in zip(variables, dfs) if "izquierda" in name.lower()]
raw_der = [df for name, df in zip(variables, dfs) if "derecha" in name.lower()]

for df in raw_izq:
    # Identificar los nombres exactos de las columnas a intercambiar
    col_3 = [col for col in df.columns if col.endswith('S3')][0]
    col_5 = [col for col in df.columns if col.endswith('S5')][0]
    col_6 = [col for col in df.columns if col.endswith('S6')][0]
    col_8 = [col for col in df.columns if col.endswith('S8')][0]

    # Intercambiar los datos
    df[[col_3, col_5]] = df[[col_5, col_3]]
    df[[col_6, col_8]] = df[[col_8, col_6]]

    # Reordenar columnas: Tiempo primero, luego sensores en orden S1 a S8
    tiempo_col = [col for col in df.columns if col.lower().startswith('tiempo')][0]
    sensores_ordenados = sorted(
        [col for col in df.columns if col != tiempo_col],
        key=lambda x: int(x.split('S')[-1])
    )
    columnas_finales = [tiempo_col] + sensores_ordenados
    df.loc[:, :] = df[columnas_finales]

# Obtener los nombres en el mismo orden que los DataFrames filtrados
nombres_izq = [name for name in variables if "izquierda" in name.lower()]
nombres_der = [name for name in variables if "derecha" in name.lower()]

################ PROCESAMIENTO BASE DE DATOS CRUDOS #######################

def procesar_plantillas(datos_derecha, datos_izquierda):
    def interp(df_list, frecuencia_hz=100):
        dfs_reamostrados = []
        logs_insertados = []
        intervalo = 1 / frecuencia_hz

        for df in df_list:
            df_copy = df.copy()
            tiempo_inicial = df_copy['Tiempo'].iloc[0]
            tiempo_final = df_copy['Tiempo'].iloc[-1]

            # Crear tiempos ideales
            tiempos_ideales = np.arange(tiempo_inicial, tiempo_final + intervalo, intervalo)
            df_copy = df_copy.set_index('Tiempo')

            df_nuevo = []
            log = []

            for t in tiempos_ideales:
                if t in df_copy.index:
                    fila = df_copy.loc[t]
                else:
                    # Tomar la última fila conocida (fill forward)
                    fila = df_copy[df_copy.index < t].iloc[-1]
                    log.append({'TiempoInsertado': round(t, 3)})

                fila_dict = fila.to_dict()
                fila_dict['Tiempo'] = round(t, 3)
                df_nuevo.append(fila_dict)

            df_final = pd.DataFrame(df_nuevo)
            # Asegurar tipos
            for col in df_final.columns:
                if col != 'Tiempo':
                    df_final[col] = df_final[col].astype(int)

            dfs_reamostrados.append(df_final)
            logs_insertados.append(pd.DataFrame(log))

        return dfs_reamostrados, logs_insertados


    def preproc_df(dataframes):
        # Iterar sobre cada DataFrame en la lista
        for df in dataframes:
            # Crear una copia del DataFrame para no modificar el original
            # Normalizar el primer valor de Tiempo a 0
            # Restar el primer valor para que el primer tiempo sea 0
            df['Tiempo'] = (df['Tiempo'] - df['Tiempo'].iloc[0])
            
        
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

            df_processed = df_copy.apply(correct_out_of_range)

            # Convertir a mV correctamente
            df_mV = df_copy.map(lambda x: int((x / 1023) * 5000) if pd.notnull(x) else 0)
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
        para cada DataFrame en una lista. Convierte 'Tiempo' a segundos si está en milisegundos.
        """
        dfs_limpios = []

        for df in df_list:
            df_copy = df.copy()

            # Asegurar que 'Tiempo' crece (no hay regresión de tiempo)
            if (df_copy['Tiempo'].diff() < 0).any():
                print("Advertencia: Hay regresiones en el tiempo, revisar DataFrame.")

            df_copy['Tiempo'] = df_copy['Tiempo'] / 1000.0

            # Limpiar valores de sensores
            columnas_sensores = [col for col in df.columns if col != 'Tiempo']
            df_copy[columnas_sensores] = df_copy[columnas_sensores].where(
                df_copy[columnas_sensores] <= valor_maximo, np.nan)
            
            df_copy[columnas_sensores] = df_copy[columnas_sensores].ffill()

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
    raw_der_final, _ = interp(raw_der_proc, frecuencia_hz=100)
    raw_izq_final, _ = interp(raw_izq_proc, frecuencia_hz=100)

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
        
    sums_der = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in sums_der]
    sums_izq = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in sums_izq]
    
    return filt_der, filt_izq, sums_der, sums_izq, dataframes_der, dataframes_izq, mV_der, mV_izq

#FILT: 8 sensores en kg
#SUMS: suma de sensores en kg
#DATAFRAMES: 8 sensores en kg antes del filtrado
#mV: 8 sensores en mV

filt_der, filt_izq, sums_der, sums_izq, dataframes_der, dataframes_izq, mV_der, mV_izq = procesar_plantillas(raw_der, raw_izq)


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



# GRÁFICOS FILT (kg)
# Pie DERECHO 
der = filt_der
if len(der) == 0:
    print("No hay datos de PIE DERECHO para graficar.")
else:
    fig_der_kg, axes = plt.subplots(len(der), 1, figsize=(12, 2.5 * len(der)))
    if len(der) == 1:
        axes = [axes]
    
    for i, df in enumerate(der):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie DERECHO - Pasada {i+1}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
   
izq = filt_izq
# Pie IZQUIERDO 
if len(izq) == 0:
    print("No hay datos de PIE IZQUIERDO para graficar.")
else:
    fig_izq_kg, axes = plt.subplots(len(izq), 1, figsize=(12, 2.5 * len(izq)))
    if len(izq) == 1:
        axes = [axes]
    
    for i, df in enumerate(izq):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie IZQUIERDO - Pasada {i+1}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)


def correccion_baseline(filt_der, filt_izq, sums_der, sums_izq):
    """
    Aplica la corrección de baseline (offset) tanto a los sensores individuales como a la suma total,
    para ambos pies. La suma se corrige directamente (no se recalcula desde sensores).

    Args:
        filt_der, filt_izq (list of pd.DataFrame): listas de DataFrames con los sensores por pie (8 columnas).
        sums_der, sums_izq (list of pd.DataFrame): listas de DataFrames con una sola columna: suma total por pie.

    Returns:
        sensores_der_corr, sensores_izq_corr (list of pd.DataFrame): sensores corregidos.
        suma_der_corr, suma_izq_corr (list of pd.DataFrame): suma total corregida.
        offsets_sensores_der, offsets_sensores_izq (list of pd.Series): offsets mínimos por sensor.
        offsets_suma_der, offsets_suma_izq (list of float): mínimos restados de la suma total.
    """
    def corregir_sensores(lista_df):
        lista_corr = []
        lista_offsets = []
        for df in lista_df:
            offsets = df.min()
            df_corr = df - offsets
            df_corr[df_corr < 0] = 0
            lista_corr.append(df_corr)
            lista_offsets.append(offsets)
        return lista_corr, lista_offsets

    def corregir_suma(lista_df):
        lista_corr = []
        lista_offsets = []
        for df in lista_df:
            fuerza = df.iloc[:, 0]
            offset = fuerza.min()
            fuerza_corr = fuerza - offset
            fuerza_corr[fuerza_corr < 0] = 0
            df_corr = pd.DataFrame(fuerza_corr, index=df.index)
            lista_corr.append(df_corr)
            lista_offsets.append(offset)
        return lista_corr, lista_offsets

    # Corregir sensores
    sensores_der_corr, offsets_sensores_der = corregir_sensores(filt_der)
    sensores_izq_corr, offsets_sensores_izq = corregir_sensores(filt_izq)

    # Corregir suma directamente (sin recalcularla)
    suma_der_corr, offsets_suma_der = corregir_suma(sums_der)
    suma_izq_corr, offsets_suma_izq = corregir_suma(sums_izq)

    return (sensores_der_corr, sensores_izq_corr,
            suma_der_corr, suma_izq_corr,
            offsets_sensores_der, offsets_sensores_izq,
            offsets_suma_der, offsets_suma_izq)

(filt_der, filt_izq,
 sums_der, sums_izq,
 offsets_sens_der, offsets_sens_izq,
 offsets_sum_der, offsets_sum_izq) = correccion_baseline(
    filt_der, filt_izq, sums_der, sums_izq
)


############### CÁLCULO DE MÍNIMOS DE LA SUMA PARA EXTRAER CICLOS #####################
def minimos(sums_der, sums_izq):
    min_indices_der = []
    min_indices_izq = []
    min_tiempos_der = []
    min_tiempos_izq = []

    # Process right foot data
    for der in sums_der:
        min_der = argrelextrema(der.values, np.less, order=40)[0]
        min_tiempo_der = der.index[min_der]
        
        min_indices_der.append(min_der)
        min_tiempos_der.append(min_tiempo_der)

    # Process left foot data
    for izq in sums_izq:
        min_izq = argrelextrema(izq.values, np.less, order=40)[0]
        min_tiempo_izq = izq.index[min_izq]
        
        min_indices_izq.append(min_izq)
        min_tiempos_izq.append(min_tiempo_izq)

    return min_indices_der, min_indices_izq, min_tiempos_der, min_tiempos_izq


min_indices_der, min_indices_izq, min_tiempos_der, min_tiempos_izq = minimos(sums_der, sums_izq)


################# EXTRACCION DE CICLOS SOLO FASE DE APOYO #########################
def extraer_ciclos(series_list, min_indices_list, threshold_ratio=0.15):
    ciclos_por_pasada = []
    indices_por_pasada = []
    ciclos_completos_por_pasada = []

    for serie, min_indices in zip(series_list, min_indices_list):
        ciclos = []
        indices = []
        ciclos_completos = []

        for i in range(len(min_indices) - 1):
            idx_inicio = min_indices[i]
            idx_fin = min_indices[i + 1]
            ciclo_completo = serie.iloc[idx_inicio:idx_fin + 1]
            ciclos_completos.append(ciclo_completo)

            datos = ciclo_completo.to_numpy().ravel()
            max_valor = datos.max()
            threshold = threshold_ratio * max_valor
            indices_relativos_validos = np.where(datos > threshold)[0]

            if len(indices_relativos_validos) > 0:
                i_rel_inicio = indices_relativos_validos[0]
                i_rel_fin = indices_relativos_validos[-1]
                ciclo = ciclo_completo.iloc[i_rel_inicio:i_rel_fin + 1]
                idx_recorte_inicio = ciclo.index[0]
                idx_recorte_fin = ciclo.index[-1]
            else:
                ciclo = ciclo_completo
                idx_recorte_inicio = ciclo.index[0]
                idx_recorte_fin = ciclo.index[-1]

            ciclos.append(ciclo)
            indices.append((idx_recorte_inicio, idx_recorte_fin))

        ciclos_por_pasada.append(ciclos)
        indices_por_pasada.append(indices)
        ciclos_completos_por_pasada.append(ciclos_completos)

    return ciclos_por_pasada, indices_por_pasada, ciclos_completos_por_pasada



#CICLOS: los valores de la suma en kg y el Tiempo de indice
#INDICES: el inicio y fin en tiempo de los ciclos en este formato (inicio, fin), ambos float
ciclos_der, indices_ciclos_der, ciclos_completos_der = extraer_ciclos(sums_der, min_indices_der)
ciclos_izq, indices_ciclos_izq, ciclos_completos_izq = extraer_ciclos(sums_izq, min_indices_izq)




############## DETECCIÓN DE PICOS Y VALLES EN CADA CICLO, CLASIFICACIÓN DE VÁLIDOS/INVÁLIDOS ########################
def improved_vGRF_detection(ciclos_por_pasada, indices_por_pasada, ciclos_completos_por_pasada,
                                time_threshold=0.2, 
                                min_prominence_factor=0.05,
                                min_peak_height=0.1,
                                support_ratio_threshold=(0.55, 0.75)):
    """
    Detección de patrones vGRF usando ciclos ya recortados, con evaluación de proporción de fase de apoyo.

    Args:
        ciclos_por_pasada: Lista de listas de pd.Series, una por ciclo ya recortado
        indices_por_pasada: Lista de (inicio, fin) en tiempo real de cada ciclo
        ciclos_completos_por_pasada: Lista de listas de pd.Series con los ciclos sin recorte
        time_threshold: Tolerancia de duración respecto a la mediana (20%)
        min_prominence_factor: % de la amplitud para prominencia mínima
        min_peak_height: Altura mínima relativa del pico
        support_ratio_threshold: Tuple con mínimo y máximo aceptables de proporción de fase de apoyo

    Returns:
        Lista de resultados por pasada, con ciclos válidos/invalidos y diagnóstico.
    """
    resultados = []

    for ciclos, indices_ciclos, completos in zip(ciclos_por_pasada, indices_por_pasada, ciclos_completos_por_pasada):
        if len(ciclos) < 2:
            resultados.append({"valid": [], "invalid": [], "details": []})
            continue

        tiempos_inicio = [i[0] for i in indices_ciclos if i[0] is not None]
        tiempos_fin = [i[1] for i in indices_ciclos if i[1] is not None]
        duraciones = [fin - ini for ini, fin in zip(tiempos_inicio, tiempos_fin)]
        median_dur = np.median(duraciones)

        ciclo_resultados = []
        ciclos_validos = []

        for j, (ciclo_apoyo, ciclo_completo, duracion) in enumerate(zip(ciclos, completos, duraciones)):
            y = ciclo_apoyo.values.flatten()
            max_val = np.max(y)
            min_prominence = max_val * min_prominence_factor
            min_height = max_val * min_peak_height

            try:
                peaks, props = find_peaks(y, prominence=min_prominence, height=min_height)
                valleys, _ = find_peaks(-y, prominence=min_prominence)

                is_valid = True
                razones = []

                # Duración del ciclo recortado
                if not ((1 - time_threshold) * median_dur < duracion < (1 + time_threshold) * median_dur):
                    is_valid = False
                    razones.append("Duración anómala")

                if len(peaks) < 2:
                    is_valid = False
                    razones.append(f"Solo {len(peaks)} pico(s)")

                if len(valleys) < 1:
                    is_valid = False
                    razones.append("Sin valle central")

                # Proporción de fase de apoyo respecto al ciclo completo
                duracion_apoyo = ciclo_apoyo.index[-1] - ciclo_apoyo.index[0]
                duracion_completo = ciclo_completo.index[-1] - ciclo_completo.index[0]
                proporcion_apoyo = duracion_apoyo / duracion_completo if duracion_completo > 0 else 0

                if not (support_ratio_threshold[0] <= proporcion_apoyo <= support_ratio_threshold[1]):
                    is_valid = False
                    razones.append(f"Proporción de apoyo fuera de rango ({proporcion_apoyo:.2f})")

            except Exception as e:
                is_valid = False
                peaks = []
                valleys = []
                razones = [f"Error: {str(e)}"]

            if is_valid:
                ciclos_validos.append(j)

            ciclo_resultados.append({
                "cycle_num": j,
                "is_valid": is_valid,
                "peaks": [ciclo_apoyo.index[p] for p in peaks] if len(peaks) else [],
                "valleys": [ciclo_apoyo.index[v] for v in valleys] if len(valleys) else [],
                "exclusion_reasons": razones,
                "duration": duracion,
                "support_ratio": proporcion_apoyo
            })

        resultados.append({
            "valid": ciclos_validos,
            "invalid": [j for j in range(len(ciclos)) if j not in ciclos_validos],
            "details": ciclo_resultados
        })

    return resultados


results_der = improved_vGRF_detection(ciclos_der, indices_ciclos_der, ciclos_completos_der, 
                                        time_threshold=0.2,  # Aumentar para permitir más variación en duración
                                        min_prominence_factor=0.05,  # Reducir para detectar picos menos prominentes
                                        min_peak_height=0.1, # Reducir para incluir picos más pequeños
                                        support_ratio_threshold=(0.55, 0.75)) 
results_izq = improved_vGRF_detection(ciclos_izq, indices_ciclos_izq, ciclos_completos_izq,
                                        time_threshold=0.2,  
                                        min_prominence_factor=0.05,                                          
                                        min_peak_height=0.1,
                                        support_ratio_threshold=(0.55, 0.75)) 


def plot_with_diagnosis(senales_completas, results, indices_ciclos, min_tiempos=None):
    """
    Muestra la señal completa con sombreado en las fases de apoyo:
    - Verde para ciclos válidos
    - Rojo para inválidos
    También muestra picos, valles y los mínimos globales (si se proveen).
    """
    for i, (df, res, recortes) in enumerate(zip(senales_completas, results, indices_ciclos)):
        fig, ax = plt.subplots(figsize=(12, 4))
        col_name = df.columns[0]
        x = df.index
        y = df[col_name]

        # Señal completa
        ax.plot(x, y, color='gray', alpha=0.8, label='vGRF')

        # Sombrear fases de apoyo
        for ciclo in res['details']:
            num = ciclo['cycle_num']
            if num >= len(recortes):
                continue

            inicio, fin = recortes[num]
            if inicio is None or fin is None:
                continue

            color = 'green' if ciclo['is_valid'] else 'red'
            
            recorte_x = df.loc[inicio:fin].index
            recorte_y = df.loc[inicio:fin, col_name]

            ax.fill_between(recorte_x, 0, recorte_y, color=color, alpha=0.2)

            # Marcar picos y valles 
            for peak in ciclo['peaks']:
                ax.scatter(peak, df.loc[peak], color='blue', marker='x', s=40)

            for valley in ciclo['valleys']:
                ax.scatter(valley, df.loc[valley], color='purple', marker='o', s=40)

            if not ciclo['is_valid'] and len(ciclo['exclusion_reasons']) > 0:
                mid_point = inicio + (fin - inicio) / 2
                ax.text(mid_point, y.max() * 0.9,
                        "\n".join(ciclo['exclusion_reasons']),
                        ha='center', va='top', fontsize=8)

        # Agregar mínimos globales si se pasan
        if min_tiempos is not None:
            tiempos_minimos = min_tiempos[i]
            for t_min in tiempos_minimos:
                if t_min in df.index:
                    ax.scatter(t_min, df.loc[t_min], color='orange', marker='o', s=20, label='Mínimo' if t_min == tiempos_minimos[0] else "")

        ax.set_title(f'Pasada {i+1} - Ciclos válidos: {len(res["valid"])}/{len(res["details"])}')
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("vGRF")
        ax.legend()
        plt.tight_layout()



plot_with_diagnosis(sums_der, results_der, indices_ciclos_der, min_tiempos=min_tiempos_der)
plot_with_diagnosis(sums_izq, results_izq, indices_ciclos_izq, min_tiempos=min_tiempos_izq)


############# GRÁFICO IZQUIERDA Y DERECHA INTERCALADOS, SE DIFERENCIA VALIDEZ DE CADA CICLO ####################
def plot_izquierda_derecha(sums_der_subset, sums_izq_subset,
                                        min_indices_der, min_indices_izq,
                                        results_der, results_izq):
    """
    Grafica la señal completa de ambos pies coloreada por ciclo completo (apoyo + swing),
    según validez del ciclo, y diferenciando pie por opacidad.
    """
    for i, (df_der, df_izq, mins_der, mins_izq, res_der, res_izq) in enumerate(
        zip(sums_der_subset, sums_izq_subset, min_indices_der, min_indices_izq, results_der, results_izq)
    ):
        fig, ax = plt.subplots(figsize=(14, 5))
        col_der = df_der.columns[0]
        col_izq = df_izq.columns[0]

        # Pie derecho
        for j, ciclo in enumerate(res_der['details']):
            if j + 1 >= len(mins_der):
                continue
            idx_inicio = df_der.index[mins_der[j]]
            idx_fin = df_der.index[mins_der[j+1]]
            tramo = df_der.loc[idx_inicio:idx_fin, col_der]
            color = 'green' if ciclo['is_valid'] else 'red'
            ax.plot(tramo.index, tramo.values, color=color, alpha=0.9, label='Derecho' if j == 0 else None)

        # Pie izquierdo
        for j, ciclo in enumerate(res_izq['details']):
            if j + 1 >= len(mins_izq):
                continue
            idx_inicio = df_izq.index[mins_izq[j]]
            idx_fin = df_izq.index[mins_izq[j+1]]
            tramo = df_izq.loc[idx_inicio:idx_fin, col_izq]
            color = 'green' if ciclo['is_valid'] else 'red'
            ax.plot(tramo.index, tramo.values, color=color, alpha=0.4, label='Izquierdo' if j == 0 else None)

        ax.set_title(f'Señal continua vGRF - Pie Derecho/Izquierdo - Ciclos completos coloreados')
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("vGRF")
        ax.legend()
        ax.set_ylim(bottom=0)
        plt.tight_layout()



plot_izquierda_derecha(
    sums_der,
    sums_izq,
    min_indices_der,
    min_indices_izq,
    results_der,
    results_izq
)

plt.show()
######### APORTES RELATIVOS DE SENSORES PERO SOBRE LA SUMA TOTAL, SIN SACAR SENSORES NO VALIDOS Y SIN SEPARAR EN FASES ###########
######### NO ACTUALIZADA PORQUE NO DEBERIA TOMAR min_indices (sino indices_ciclos) ##########
def calcular_aportes_sensores(pasadas_crudas, resultados_deteccion, min_indices):
    """
    Calcula el aporte relativo de cada sensor para ciclos válidos detectados.
    
    Args:
        pasadas_crudas: Lista de DataFrames con datos crudos de los sensores
        resultados_deteccion: Resultado de improved_vGRF_detection()
        min_indices: Índices de mínimos usados en la detección
        
    Returns:
        dict: Diccionario estructurado con:
            - 'por_ciclo': Lista de DataFrames con aportes por ciclo
            - 'resumen': DataFrame con estadísticas agregadas
            - 'sensores': Lista de nombres de sensores
    """
    resultados = {
        'por_ciclo': [],
        'resumen': None,
        'sensores': pasadas_crudas[0].columns.tolist() if len(pasadas_crudas) > 0 else []
    }
    
    datos_para_resumen = []

    for i, (df_crudo, res_pasada) in enumerate(zip(pasadas_crudas, resultados_deteccion)):
        if not res_pasada['valid']:
            continue
            
        if isinstance(df_crudo.index, pd.DatetimeIndex):
            df_crudo.index = df_crudo.index.round('1ms')  # Correcto para datetime
        else:
            df_crudo.index = pd.to_numeric(df_crudo.index).round(3)  # Para índices numéricos

        for ciclo_num in res_pasada['valid']:
            # Extraer datos del ciclo
            inicio = min_indices[i][ciclo_num-1]
            fin = min_indices[i][ciclo_num]
            datos_ciclo = df_crudo.iloc[inicio:fin].copy()
            
            # Calcular aportes relativos
            suma_total = datos_ciclo.sum(axis=1)
            aportes = datos_ciclo.div(suma_total, axis=0).fillna(0)
            
            # Almacenar resultados
            ciclo_info = {
                'pasada': i+1,
                'ciclo': ciclo_num,
                'aportes': aportes,
                'media_por_sensor': aportes.mean(),
                'duracion': resultados_deteccion[i]['details'][ciclo_num-1]['duration']
            }
            
            resultados['por_ciclo'].append(ciclo_info)
            datos_para_resumen.append(aportes.mean())
    
    # Crear resumen agregado
    if datos_para_resumen:
        df_resumen = pd.DataFrame(datos_para_resumen)
        resultados['resumen'] = {
            'media': df_resumen.mean(),
            'desviacion': df_resumen.std(),
            'min': df_resumen.min(),
            'max': df_resumen.max()
        }
    
    return resultados

'''
# 1. Calcular los aportes
aportes_der = calcular_aportes_sensores(
    filt_der, 
    results_der, 
    min_indices_der
)

aportes_izq = calcular_aportes_sensores(
    filt_izq, 
    results_izq, 
    min_indices_izq
)
'''

######### APORTES RELATIVOS DE SENSORES POR FASE PERO SOBRE LA SUMA TOTAL, SIN DISCRIMINAR SENSORES NO VALIDOS ###########
######### NO ACTUALIZADA PORQUE NO DEBERIA TOMAR min_indices (sino indices_ciclos) ##########
def calcular_aportes_por_fase(pasadas_crudas, resultados_deteccion, min_indices):
    """
    Calcula el aporte relativo de cada sensor por fase (HS, FF, MS, HO) en ciclos válidos.

    Returns:
        dict: Resultados por fase, por ciclo, y resumen agregado.
    """
    resultados = {
        'por_ciclo': [],  # Lista de dicts con aportes por fase por ciclo
        'resumen': {},    # Media y std por fase
        'sensores': pasadas_crudas[0].columns.tolist() if pasadas_crudas else []
    }

    fases = ['loading_response', 'midstance', 'terminal_stance', 'pre_swing']
    datos_resumen = {fase: [] for fase in fases}

    for i, (df_crudo, res_pasada) in enumerate(zip(pasadas_crudas, resultados_deteccion)):
        if not res_pasada['valid']:
            continue

        for ciclo_info in res_pasada['details']:
            if not ciclo_info['is_valid']:
                continue
            
            ciclo_num = ciclo_info['cycle_num']
            inicio = min_indices[i][ciclo_num - 1]
            fin = min_indices[i][ciclo_num]
            datos_ciclo = df_crudo.iloc[inicio:fin].copy().reset_index(drop=True)

            peaks = ciclo_info['peaks']
            valleys = ciclo_info['valleys']
            
            # Validar que haya al menos 2 picos y 1 valle
            if len(peaks) < 2 or len(valleys) < 1:
                continue

            p1, p2 = peaks[0], peaks[1]
            v = valleys[0]

            # Definir segmentos
            segmentos = {
                'loading_response': datos_ciclo.iloc[:p1],
                'midstance': datos_ciclo.iloc[p1:v],
                'terminal_stance': datos_ciclo.iloc[v:p2],
                'pre_swing': datos_ciclo.iloc[p2:]
            }

            aportes_fase = {}
            for fase, segmento in segmentos.items():
                if segmento.empty:
                    aporte = pd.Series(0, index=df_crudo.columns)
                else:
                    suma_total = segmento.sum(axis=1)
                    aportes = segmento.div(suma_total, axis=0).fillna(0)
                    aporte = aportes.mean()
                aportes_fase[fase] = aporte
                datos_resumen[fase].append(aporte)

            resultados['por_ciclo'].append({
                'pasada': i + 1,
                'ciclo': ciclo_num,
                'aportes_por_fase': aportes_fase,
                'duracion': ciclo_info['duration']
            })

    # Resumen global
    for fase in fases:
        df_fase = pd.DataFrame(datos_resumen[fase])
        resultados['resumen'][fase] = {
            'media': df_fase.mean(),
            'desviacion': df_fase.std(),
            'min': df_fase.min(),
            'max': df_fase.max()
        }

    return resultados

'''
aportes_por_fase_der = calcular_aportes_por_fase(filt_der, results_der, min_indices_der)
aportes_por_fase_izq = calcular_aportes_por_fase(filt_izq, results_izq, min_indices_izq)
'''


# === Coordenadas de sensores ===
coord_sensores = [
    ['S1', 6.4, 182.0],
    ['S2', 0.0, 149.3],
    ['S3', 28.7, 154.0],
    ['S4', 56.8, 149.5],
    ['S5', 54.4, 93.6],
    ['S6', 6.4, 16.0],
    ['S7', 23.4, 0.1],
    ['S8', 41.3, 14.6]
]

coord_df = pd.DataFrame(coord_sensores, columns=["Sensor", "X", "Y"]).set_index("Sensor")

# Reflejo en X para el pie izquierdo
coord_df_izq = coord_df.copy()
coord_df_izq["X"] = -coord_df_izq["X"]

# Offsets para que (0,0) no quede pegado al borde
offset_izq = abs(coord_df_izq["X"].min()) + 10
offset_der = abs(coord_df["X"].min()) + 10

coord_df_izq["X"] += offset_izq
coord_df_der = coord_df.copy()
coord_df_der["X"] += offset_der

# === Función para calcular COP por ciclos válidos ===
def calcular_COP_por_ciclos_validos(fuerza_df, vgrf_df, coord_df, indices_ciclos, validos, lado):
    copx_total = []
    copy_total = []

    sufijo = 'Derecha' if lado.upper() == 'R' else 'Izquierda'
    coord_actual = coord_df.copy()
    coord_actual.index = [f"{sufijo}_{s}" for s in coord_actual.index]

    for (ini, fin), es_valido in zip(indices_ciclos, validos):
        if not es_valido:
            continue

        ciclo_fuerza = fuerza_df[ini:fin]
        ciclo_vgrf = vgrf_df[ini:fin]
        
        coordenadas = coord_actual.loc[ciclo_fuerza.columns][['X', 'Y']].to_numpy()
        matriz_fuerza = ciclo_fuerza.to_numpy()
        vector_vgrf = ciclo_vgrf.to_numpy().flatten()
        vector_vgrf[vector_vgrf == 0] = np.nan

        cop_x = np.sum(matriz_fuerza * coordenadas[:, 0], axis=1) / vector_vgrf
        cop_y = np.sum(matriz_fuerza * coordenadas[:, 1], axis=1) / vector_vgrf

        copx_total.append(pd.Series(cop_x).reset_index(drop=True))
        copy_total.append(pd.Series(cop_y).reset_index(drop=True))

    return copx_total, copy_total

# === Calcular COP ===
copx_izq_list, copy_izq_list = calcular_COP_por_ciclos_validos(
    filt_izq[0], sums_izq[0], coord_df_izq,
    indices_ciclos_izq[0], [d['is_valid'] for d in results_izq[0]['details']], lado='L'
)

copx_der_list, copy_der_list = calcular_COP_por_ciclos_validos(
    filt_der[0], sums_der[0], coord_df_der,
    indices_ciclos_der[0], [d['is_valid'] for d in results_der[0]['details']], lado='R'
)

# === Graficar ===
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
pies = ['Izquierdo', 'Derecho']
copx_all = [copx_izq_list, copx_der_list]
copy_all = [copy_izq_list, copy_der_list]
coord_dfs = [coord_df_izq, coord_df_der]
sensor_colores = ['blue', 'yellow', 'grey', 'pink', 'green', 'orange', 'purple', 'red']
colores = plt.colormaps['tab10']
margen = 20

for ax, pie, copx_list, copy_list, coord_df in zip(axes, pies, copx_all, copy_all, coord_dfs):
    x_min = coord_df['X'].min() - margen
    x_max = coord_df['X'].max() + margen
    y_min = coord_df['Y'].min() - margen
    y_max = coord_df['Y'].max() + margen

    # Grilla de fondo
    ax.imshow(np.zeros((500, 500)), extent=[x_min, x_max, y_min, y_max], cmap='gray', alpha=0.05)

    # Sensores
    for i, (sensor, fila) in enumerate(coord_df.iterrows()):
        x, y = fila['X'], fila['Y']
        ax.add_patch(Circle((x, y), radius=7.32, color=sensor_colores[i], alpha=0.5))
        ax.text(x, y, sensor, ha='center', va='center', fontsize=8)

    ax.add_patch(Circle((0, 0), radius=1, color='black', alpha=0.7))  # punto (0,0)

    # COP por ciclo
    for i, (cop_x, cop_y) in enumerate(zip(copx_list, copy_list)):
        ax.plot(cop_x, cop_y, linestyle='--', color=colores(i % 10), alpha=0.6)

    # COP promedio
    if copx_list:
        copx_prom = pd.concat(copx_list, axis=1).mean(axis=1)
        copy_prom = pd.concat(copy_list, axis=1).mean(axis=1)
        ax.plot(copx_prom, copy_prom, color='black', linewidth=2, label='COP promedio')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'Pie {pie}')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.grid(True)
    ax.legend()

plt.tight_layout()





######### VUELVE A CALCULAR PICOS Y VALLES DESPUES DE CORREGIR LA CURVA, MAS ABAJO#############
def recalcular_picos_valleys(datos_ciclo):

    max_val = np.max(datos_ciclo)
    min_prominence = max_val * 0.05  # 5% prominencia
    min_height = max_val * 0.1       # 10% altura mínima

    peaks, _ = find_peaks(datos_ciclo, prominence=min_prominence, height=min_height)
    valleys, _ = find_peaks(-datos_ciclo, prominence=min_prominence)

    if len(peaks) < 2 or len(valleys) < 1:
        return None, None
    return peaks[:2], valleys[:1]

############### CORRECCIÓN y GRAFICO DE SENSORES Y CURVA vGRF ################ 
def graficar_suma_por_fases_con_sensores_v2(df_sensores, resultados, indices_ciclos, lado="R"):

    sufijo = "Derecha" if lado.upper() == "R" else "Izquierda"

    for ciclo_info, indices_ciclo in zip(resultados[0]['details'], indices_ciclos[0]):
        if not ciclo_info['is_valid']:
            continue

        inicio, fin = indices_ciclo[0], indices_ciclo[1]
        datos_ciclo = df_sensores.loc[inicio:fin].copy()
        suma_total = datos_ciclo.sum(axis=1)

        peaks = ciclo_info['peaks']
        valleys = ciclo_info['valleys']
        
        if len(peaks) < 2 or len(valleys) < 1:
            print(f"Ciclo {ciclo_info['cycle_num']}: No hay suficientes picos/valles para definir fases.")
            continue

        p1, p2 = peaks[0], peaks[1]
        v = valleys[0]

        segmentos = {
            'loading_response': (inicio, p1),
            'midstance':        (p1, v),
            'terminal_stance':  (v,  p2),
            'pre_swing':        (p2, fin)
        }

        colores_fases = {
            'loading_response': 'red',
            'midstance': 'green',
            'terminal_stance': 'blue',
            'pre_swing': 'orange'
        }

        '''sensores_por_fase = {
            "loading_response": [f"{sufijo}_S5", f"{sufijo}_S6", f"{sufijo}_S7", f"{sufijo}_S8"], #ACA CAMBIAMOS SEGUN LOS SENSORES QUE TERMINEMOS ELIGIENDO
            "midstance": list(datos_ciclo.columns),
            "terminal_stance": list(datos_ciclo.columns),
            "pre_swing": [f"{sufijo}_S1", f"{sufijo}_S2", f"{sufijo}_S3", f"{sufijo}_S4"]
        }
        
        # Preparar estructura para datos reescalados
        datos_reescalados = {
            'suma_total_original': suma_total.copy(),
            'suma_reescalada': pd.Series(0, index=datos_ciclo.index),
            'fases': {},
            'picos_valores_reescalados': {'p1': None, 'p2': None},
            'valle_valor_reescalado': None,
            'indices_originales': {'p1': p1, 'p2': p2, 'valle': v}
        }

        suma_reescalada = pd.Series(np.zeros(len(datos_ciclo)), index=datos_ciclo.index, dtype=float)
        fase_sumas = {}

        for fase, (start, end) in segmentos.items():
            sensores_fase = sensores_por_fase.get(fase, [])
            datos_fase = datos_ciclo.loc[start:end].copy()

            if fase in ["loading_response", "pre_swing"]:
                sensores_no_principales = [s for s in datos_fase.columns if s not in sensores_fase]

                # Restar el aporte de sensores no principales frame a frame RESTA TODOO EL APORTE, SE PUEDE PONER PARA QUE RESTE UN % NOMAS
                aporte_no_principales = datos_fase[sensores_no_principales].sum(axis=1)
                suma_fase = datos_fase.sum(axis=1) - aporte_no_principales
            else:
                # Fases centrales: se usa todo directamente
                suma_fase = datos_fase.sum(axis=1)

            suma_reescalada.loc[start:end] = suma_fase.values
            fase_sumas[fase] = suma_fase

            datos_reescalados['fases'][fase] = {
                'start': start,
                'sensores_principales': sensores_fase,
                'sensores_escalados': [s for s in datos_fase.columns if s not in sensores_fase],
                'suma_fase': suma_fase.copy(),
                'datos_sensores': datos_fase.copy()
            }

        # CALCULO OFFSET ENTRE EL ULTIMO VALOR DE UNA FASE Y EL INICIO DE OTRA, PARA DESPLAZAR VERTICALMENTE
        # LOADING RESPONSE Y PRE-SWING NO SE DEBEN DESPLAZAR    
        offset_midstance = datos_reescalados['fases']['loading_response']['suma_fase'].iloc[-1] - fase_sumas['midstance'].iloc[0]
        datos_reescalados['fases']['midstance']['suma_fase'] += offset_midstance
        
        offset_terminal = datos_reescalados['fases']['midstance']['suma_fase'].iloc[-1] - fase_sumas['terminal_stance'].iloc[0]
        datos_reescalados['fases']['terminal_stance']['suma_fase'] += offset_terminal
        
        
        suma_reescalada_corregida = pd.Series(0, index=datos_ciclo.index, dtype=float)
        for fase, (start, end) in segmentos.items():
            suma_reescalada_corregida.loc[start:end] = datos_reescalados['fases'][fase]['suma_fase']
        
        
        picos_recalc, valles_recalc = recalcular_picos_valleys(suma_reescalada_corregida)
        
        if picos_recalc is None or valles_recalc is None:
            p1_recalc, p2_recalc, v_recalc = p1, p2, v
        else:
            p1_recalc, p2_recalc = picos_recalc
            v_recalc = valles_recalc[0]
        
        p1_recalc, p2_recalc = [suma_reescalada_corregida.index[p] for p in picos_recalc] if len(picos_recalc) else []
        v_recalc = suma_reescalada_corregida.index[v_recalc] if v_recalc else []
        
        segmentos = {
            'loading_response': (inicio, p1_recalc),
            'midstance': (p1_recalc, v_recalc),
            'terminal_stance': (v_recalc, p2_recalc),
            'pre_swing': (p2_recalc, fin)
        }
        
        #FALTARIA ACTUALIZAR datos_reescalados() CON ESTA NUEVA INFO DE PICOS Y VALLES SIENTO (O CAPAZ QUEDARNOS CON LOS NUEVOS Y LISTO)
        '''
        plt.figure(figsize=(14, 7))
        for fase, (start, end) in segmentos.items():
            start = round(start, 6)
            end = round(end, 6)
            
            plt.axvspan(start, end, alpha=0.2, color=colores_fases[fase], label=fase)

        plt.plot(suma_total.index, suma_total.values, '--', label="Suma original")
        #plt.plot(suma_reescalada_corregida.index, suma_reescalada_corregida.values, 'k-', label="Suma reescalada")

        colormap = plt.colormaps['tab10']
        for i, sensor in enumerate(datos_ciclo.columns):
            plt.plot(datos_ciclo.index, datos_ciclo[sensor], color=colormap(i), alpha=0.6, label=sensor)
        
        #plt.plot(p1_recalc, suma_reescalada_corregida.loc[p1_recalc], 'ro')
        #plt.plot(p2_recalc, suma_reescalada_corregida.loc[p2_recalc], 'ro')
        #plt.plot(v_recalc,  suma_reescalada_corregida.loc[v_recalc], 'bo')
        plt.plot(p1, suma_total.loc[p1], 'ro')
        plt.plot(p2, suma_total.loc[p2], 'ro')
        plt.plot(v,  suma_total.loc[v], 'bo')

        plt.title(f"{sufijo} - Ciclo {ciclo_info['cycle_num']}")
        plt.xlabel("Índice de muestra")
        plt.ylabel("Señal")
        plt.legend()
        plt.tight_layout()



graficar_suma_por_fases_con_sensores_v2(filt_der[0], results_der, indices_ciclos_der, lado="R")
graficar_suma_por_fases_con_sensores_v2(filt_izq[0], results_izq, indices_ciclos_izq, lado="L")


def calculo_bw(sums_der, sums_izq):

    grf_der = []
    grf_izq = []
    
    BW = 52

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
