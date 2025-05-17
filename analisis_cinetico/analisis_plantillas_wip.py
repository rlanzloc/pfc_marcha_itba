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

plt.ioff()

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

# Obtener los nombres en el mismo orden que los DataFrames filtrados
nombres_izq = [name for name in variables if "izquierda" in name.lower()]
nombres_der = [name for name in variables if "derecha" in name.lower()]

################ PROCESAMIENTO DE DATOS #######################

def procesar_plantillas(datos_derecha, datos_izquierda):
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
        
    sums_der = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in sums_der]
    sums_izq = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in sums_izq]
        
    return filt_der, filt_izq, sums_der, sums_izq, dataframes_der, dataframes_izq, mV_der, mV_izq

filt_der, filt_izq, sums_der, sums_izq, dataframes_der, dataframes_izq, mV_der, mV_izq = procesar_plantillas(raw_der, raw_izq)

# ==================================================
# GRÁFICOS 
# ==================================================

# Pie DERECHO 

der = dataframes_der
if len(der) == 0:
    print("No hay datos de PIE DERECHO para graficar.")
else:
    fig_der_kg, axes = plt.subplots(len(der), 1, figsize=(12, 2.5 * len(der)))
    if len(der) == 1:
        axes = [axes]
    
    for i, df in enumerate(der):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie DERECHO - Pasada {i+2}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
   
izq = dataframes_izq
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
        axes[i].set_title(f'Pie IZQUIERDO - Pasada {i+2}', pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
 
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

        axes[i].set_title(f'Suma de fuerzas (N) - Pasada N°{i+2}')
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valores')
        #axes[i].set_xlim(14, 24)  # Ajusta si es necesario
        axes[i].legend()

    plt.tight_layout()
    plt.show()



'''
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



# Definir el rango de tiempo deseado
t_inicio = 5  # Cambia estos valores según tu necesidad
t_fin = 15

sums_der_subset = subset(sums_der, t_inicio, t_fin)
sums_izq_subset = subset(sums_izq, t_inicio, t_fin)

filt_der_subset = subset(filt_der, t_inicio, t_fin)
filt_izq_subset = subset (filt_izq, t_inicio, t_fin)

min_indices_der, min_indices_izq, min_tiempos_der, min_tiempos_izq = minimos(sums_der_subset, sums_izq_subset)


def extraer_ciclos(series_list, min_indices_list):
    """
    Extrae ciclos de marcha entre mínimos consecutivos de una lista de señales.

    Parámetros:
    - series_list: lista de pd.Series (una por pasada), ej: sums_der o sums_izq
    - min_indices_list: lista de arrays de índices de mínimos, misma longitud que series_list

    Retorna:
    - ciclos_por_pasada: lista de listas de pd.Series, cada uno representa un ciclo
    """
    ciclos_por_pasada = []

    for serie, min_indices in zip(series_list, min_indices_list):
        ciclos = []
        for i in range(len(min_indices) - 1):
            idx_inicio = min_indices[i]
            idx_fin = min_indices[i + 1]
            ciclo = serie.iloc[idx_inicio:idx_fin+1]
            ciclos.append(ciclo)
        ciclos_por_pasada.append(ciclos)

    return ciclos_por_pasada

ciclos_der = extraer_ciclos(sums_der_subset, min_indices_der)
ciclos_izq = extraer_ciclos(sums_izq_subset, min_indices_izq)


def improved_vGRF_detection(pasadas, min_indices, time_threshold=0.2, 
                          min_prominence_factor=0.05, min_peak_height=0.1):
    """
    Detección de patrones vGRF SIN validación de posición de picos
    
    Args:
        pasadas: Lista de DataFrames con señales
        min_indices: Índices de mínimos
        time_threshold: Máxima variación permitida en duración (20%)
        min_prominence_factor: % de la amplitud para detección de picos (10%)
        min_peak_height: Altura mínima relativa para picos (25% del máximo)
    
    Returns:
        dict: Resultados por pasada con detalles de exclusión
    """
    results = []
    
    for i, (df, mins) in enumerate(zip(pasadas, min_indices)):
        if len(mins) < 2:
            results.append({"valid": [], "invalid": [], "details": []})
            continue
            
        # 1. Análisis de duración
        durations = np.diff(df.index[mins])
        median_dur = np.median(durations)
        valid_time = (durations > (1-time_threshold)*median_dur) & \
                    (durations < (1+time_threshold)*median_dur)
        
        # 2. Detección de patrones (sin validación de posición)
        cycle_results = []
        valid_cycles = []
        
        for j in range(len(mins)-1):
            cycle_data = df.iloc[mins[j]:mins[j+1]]
            y = cycle_data.values.flatten()
            
            # Parámetros adaptativos
            max_val = np.max(y)
            min_prominence = max_val * min_prominence_factor
            min_height = max_val * min_peak_height
            
            # Detección de características
            try:
                peaks, props = find_peaks(y, prominence=min_prominence, height=min_height)
                valleys, _ = find_peaks(-y, prominence=min_prominence)
                
                # Criterios de validación (simplificados)
                is_valid = True
                exclusion_reason = []
                
                if not valid_time[j]:
                    is_valid = False
                    exclusion_reason.append("Duración anómala")
                
                if len(peaks) < 2:
                    is_valid = False
                    exclusion_reason.append(f"Solo {len(peaks)} pico(s)")
                
                if len(valleys) < 1:
                    is_valid = False
                    exclusion_reason.append("Sin valle central")
                
            except Exception as e:
                is_valid = False
                exclusion_reason.append(f"Error: {str(e)}")
            
            # Guardar resultados
            if is_valid:
                valid_cycles.append(j+1)
            
            cycle_results.append({
                "cycle_num": j+1,
                "is_valid": is_valid,
                "peaks": peaks.tolist() if 'peaks' in locals() else [],
                "valleys": valleys.tolist() if 'valleys' in locals() else [],
                "exclusion_reasons": exclusion_reason,
                "duration": durations[j]
            })
        
        results.append({
            "valid": valid_cycles,
            "invalid": [j+1 for j in range(len(mins)-1) if j+1 not in valid_cycles],
            "details": cycle_results
        })
    
    return results


def plot_with_diagnosis(pasadas, results, min_indices):
    """Visualización con información de exclusión"""
    for i, (df, res) in enumerate(zip(pasadas, results)):
        fig, ax = plt.subplots(figsize=(12, 4))
        
        col_name = df.columns[0]
        x = df.index.to_numpy()
        y = df[col_name].to_numpy()
        # Señal completa
        ax.plot(x, y, color='gray', alpha=0.5, label='Señal cruda')
        
        # Resaltar ciclos
        for cycle in res['details']:
            start = df.index[min_indices[i][cycle['cycle_num']-1]]
            end = df.index[min_indices[i][cycle['cycle_num']]]
            
            color = 'green' if cycle['is_valid'] else 'red'
            ax.axvspan(start, end, color=color, alpha=0.2)
            
            # Marcar picos/valles
            if cycle['peaks']:
                for peak in cycle['peaks']:
                    pos = df.index[min_indices[i][cycle['cycle_num']-1] + peak]
                    ax.scatter(pos, df.loc[pos], color='blue', marker='x', s=80)
            
            if cycle['valleys']:
                for valley in cycle['valleys']:
                    pos = df.index[min_indices[i][cycle['cycle_num']-1] + valley]
                    ax.scatter(pos, df.loc[pos], color='purple', marker='o', s=60)
            
            # Texto de diagnóstico
            if not cycle['is_valid'] and len(cycle['exclusion_reasons']) > 0:
                mid_point = start + (end - start)/2
                ax.text(mid_point, np.max(df.values)*0.9, 
                       "\n".join(cycle['exclusion_reasons']),
                       ha='center', va='top', fontsize=8)
        
        ax.set_title(f'Pasada {i+1} - Ciclos válidos: {len(res["valid"])}/{len(res["details"])}')
        ax.legend()
        plt.tight_layout()
       
        
        
# 1. Ejecutar detección
results_der = improved_vGRF_detection(
    sums_der_subset,
    min_indices_der,
    time_threshold=0.2,  # Aumentar para permitir más variación en duración
    min_prominence_factor=0.05,  # Reducir para detectar picos menos prominentes
    min_peak_height=0.1  # Reducir para incluir picos más pequeños
)

# 1. Ejecutar detección
results_izq = improved_vGRF_detection(
    sums_izq_subset,
    min_indices_izq,
    time_threshold=0.2,  # Aumentar para permitir más variación en duración
    min_prominence_factor=0.05,  # Reducir para detectar picos menos prominentes
    min_peak_height=0.1 # Reducir para incluir picos más pequeños
)

# 2. Visualizar con diagnóstico
plot_with_diagnosis(sums_der_subset, results_der, min_indices_der)
# 2. Visualizar con diagnóstico
plot_with_diagnosis(sums_izq_subset, results_izq, min_indices_izq)


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


aportes_por_fase_der = calcular_aportes_por_fase(filt_der, results_der, min_indices_der)
aportes_por_fase_izq = calcular_aportes_por_fase(filt_izq, results_izq, min_indices_izq)

for ciclo in aportes_por_fase_der['por_ciclo']:
    print(f"Pasada {ciclo['pasada']}, Ciclo {ciclo['ciclo']}:")
    for fase, aporte in ciclo['aportes_por_fase'].items():
        print(f"  {fase}:")
        print((aporte * 100).round(2).astype(str) + " %")



# Lista original de sensores
coord_sensores = [
    ['S1', 6.4, 152.1],
    ['S2', 0, 119.4],
    ['S3', 28.7, 124.1],
    ['S4', 56.8, 119.6],
    ['S5', 54.4, 63.7],
    ['S6', 6.4, -13.9],
    ['S7', 23.4, -29.8],
    ['S8', 41.3, -15.3]
]

# Conversión a diccionario para fácil acceso a las coordenadas
sensor_coords_der = {sensor: (x, -y) for sensor, x, y in coord_sensores}
sensor_coords_izq = {sensor: (-x, -y) for sensor, x, y in coord_sensores}


def plot_footprint_ciclo_unico(aporte_ciclo, sensor_coords, pie='R'):
    """
    Genera un mapa de calor por fase del ciclo (HS, FF, MS, HO) de un solo ciclo.

    Args:
        aporte_ciclo: Dict con Series de aportes por sensor por fase
        sensor_coords: Dict con coordenadas (x, y) por sensor
        pie: 'L' o 'R' (para etiquetado)
    """
    fases = ['loading_response', 'midstance', 'terminal_stance', 'pre_swing']
    fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

    for i, fase in enumerate(fases):
        media_fase = aporte_ciclo[fase]
        x, y, intensidad = [], [], []

        for sensor, value in media_fase.items():
            sensor_simple = sensor.split('_')[-1]
            if sensor_simple in sensor_coords:
                xi, yi = sensor_coords[sensor_simple]
                x.append(xi)
                y.append(yi)
                intensidad.append(float(value))

        sc = axs[i].scatter(x, y, c=intensidad, cmap='jet', s=1000, edgecolor='k', vmin=0, vmax=1)
        axs[i].set_title(f"{fase.replace('_', ' ').title()} ({pie})")
        axs[i].invert_yaxis()
        axs[i].axis('equal')
        axs[i].axis('off')

        for j, (xi, yi, val) in enumerate(zip(x, y, intensidad)):
            axs[i].text(xi, yi, f"{val:.2f}", ha='center', va='center', fontsize=9, color='black')

    cbar = fig.colorbar(sc, ax=axs, orientation='vertical', fraction=0.015, pad=0.04)
    cbar.set_label('Aporte relativo')
   

# Tomar el ciclo 0
ciclo_0 = aportes_por_fase_der['por_ciclo'][0]['aportes_por_fase']
plot_footprint_ciclo_unico(ciclo_0, sensor_coords_der, pie='R')

ciclo_0 = aportes_por_fase_izq['por_ciclo'][0]['aportes_por_fase']
plot_footprint_ciclo_unico(ciclo_0, sensor_coords_izq, pie='L')


def graficar_suma_por_fases(sums_data, res_pasada, min_indices, ciclo_num=1):
    """
    Grafica la suma de sensores para un ciclo específico, resaltando las fases.
    
    Args:
        sums_data: Lista de Series (suma de sensores por pasada)
        res_pasada: Resultados de detección para la pasada
        min_indices: Índices de inicio/fin de ciclos
        ciclo_num: Número del ciclo a graficar (1-based)
    """
    if not res_pasada['valid']:
        print("Pasada no válida.")
        return
    
    ciclo_info = next((c for c in res_pasada['details'] if c['cycle_num'] == ciclo_num and c['is_valid']), None)
    if not ciclo_info:
        print(f"Ciclo {ciclo_num} no válido o no encontrado.")
        return
    
    inicio = min_indices[ciclo_num - 1]
    fin = min_indices[ciclo_num]
    datos_ciclo = sums_data.iloc[inicio:fin].copy().reset_index(drop=True)
    
    peaks = ciclo_info['peaks']
    valleys = ciclo_info['valleys']
    
    if len(peaks) < 2 or len(valleys) < 1:
        print("No hay suficientes picos/valles para definir fases.")
        return
    
    p1, p2 = peaks[0], peaks[1]
    v = valleys[0]
    
    # Definir segmentos
    segmentos = {
        'loading_response': (0, p1),
        'midstance': (p1, v),
        'terminal_stance': (v, p2),
        'pre_swing': (p2, len(datos_ciclo))
    }
    
    # Configurar gráfico
    plt.figure(figsize=(12, 5))
    
    # Graficar la curva de suma
    plt.plot(datos_ciclo, label='Suma de sensores', color='black', linewidth=2)
    
    # Colores para cada fase
    colores = {
        'loading_response': 'red',
        'midstance': 'green',
        'terminal_stance': 'blue',
        'pre_swing': 'orange'
    }
    
    # Resaltar cada fase
    for fase, (start, end) in segmentos.items():
        plt.axvspan(start, end, color=colores[fase], alpha=0.2, label=fase)
    
    # Marcar puntos clave
    plt.scatter([p1, p2], [datos_ciclo.iloc[p1], datos_ciclo.iloc[p2]], color='red', zorder=5, label='Picos')
    plt.scatter([v], [datos_ciclo.iloc[v]], color='blue', zorder=5, label='Valle')
    
    plt.xlabel('Muestras')
    plt.ylabel('Valor sumado')
    plt.title(f'Suma de sensores - Ciclo {ciclo_num} - División por fases')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def graficar_suma_multiple_ciclos(sums_data_list, resultados_deteccion, min_indices, n_ciclos=3):
    """Grafica la suma de sensores para los primeros n ciclos válidos."""
    ciclos_graficados = 0
    
    for sums_data, res_pasada, indices in zip(sums_data_list, resultados_deteccion, min_indices):
        if not res_pasada['valid']:
            continue
            
        for ciclo_info in res_pasada['details']:
            if not ciclo_info['is_valid'] or ciclos_graficados >= n_ciclos:
                continue
                
            graficar_suma_por_fases(sums_data, res_pasada, indices, ciclo_info['cycle_num'])
            ciclos_graficados += 1

graficar_suma_multiple_ciclos(sums_der_subset, results_der, min_indices_der, n_ciclos=2)


def graficar_suma_por_fases_con_sensores(df_sensores_list, res_pasada, min_indices, ciclo_num=1, pasada_index=0):
    """
    Grafica la suma de sensores y cada sensor individual para un ciclo específico, resaltando las fases.
    
    Args:
        df_sensores_list: Lista de DataFrames (uno por pasada), cada uno con columnas por sensor
        res_pasada: Resultados de detección para la pasada
        min_indices: Índices de inicio/fin de ciclos para esta pasada
        ciclo_num: Número del ciclo a graficar (1-based)
        pasada_index: Índice de la pasada en df_sensores_list
    """
    if not res_pasada['valid']:
        print("Pasada no válida.")
        return

    ciclo_info = next((c for c in res_pasada['details'] if c['cycle_num'] == ciclo_num and c['is_valid']), None)
    if not ciclo_info:
        print(f"Ciclo {ciclo_num} no válido o no encontrado.")
        return

    df_ciclo = df_sensores_list[pasada_index]
    inicio = min_indices[ciclo_num - 1]
    fin = min_indices[ciclo_num]
    datos_ciclo = df_ciclo.iloc[inicio:fin].copy().reset_index(drop=True)
    suma_ciclo = datos_ciclo.sum(axis=1)

    peaks = ciclo_info['peaks']
    valleys = ciclo_info['valleys']

    if len(peaks) < 2 or len(valleys) < 1:
        print("No hay suficientes picos/valles para definir fases.")
        return

    p1, p2 = peaks[0], peaks[1]
    v = valleys[0]

    # Fases del ciclo
    segmentos = {
        'loading_response': (0, p1),
        'midstance': (p1, v),
        'terminal_stance': (v, p2),
        'pre_swing': (p2, len(datos_ciclo))
    }

    colores_fases = {
        'loading_response': 'red',
        'midstance': 'green',
        'terminal_stance': 'blue',
        'pre_swing': 'orange'
    }

    # Plot
    plt.figure(figsize=(14, 6))

    # Fases como bandas de color
    for fase, (start, end) in segmentos.items():
        plt.axvspan(start, end, color=colores_fases[fase], alpha=0.2, label=fase)

    # Curva suma total
    plt.plot(suma_ciclo, label='Suma total', color='black', linewidth=2)

    # Curvas individuales de sensores
    colores_sensores = plt.cm.tab10.colors
    for i, sensor in enumerate(datos_ciclo.columns):
        plt.plot(datos_ciclo[sensor], label=sensor, color=colores_sensores[i % len(colores_sensores)], alpha=0.8)

    # Marcar eventos clave
    plt.scatter([p1, p2], [suma_ciclo.iloc[p1], suma_ciclo.iloc[p2]], color='red', zorder=5, label='Picos')
    plt.scatter([v], [suma_ciclo.iloc[v]], color='blue', zorder=5, label='Valle')

    plt.xlabel('Muestras')
    plt.ylabel('Valor FSR')
    plt.title(f'Ciclo {ciclo_num} - Suma total y sensores individuales')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def graficar_suma_multiple_ciclos_con_sensores(sums_data_list, resultados_deteccion, min_indices_list, n_ciclos=3):
    """
    Grafica la suma de sensores y sensores individuales para los primeros n ciclos válidos.
    """
    ciclos_graficados = 0

    for i, (df_sensores, res_pasada, indices) in enumerate(zip(sums_data_list, resultados_deteccion, min_indices_list)):
        if not res_pasada['valid']:
            continue

        for ciclo_info in res_pasada['details']:
            if not ciclo_info['is_valid'] or ciclos_graficados >= n_ciclos:
                continue

            graficar_suma_por_fases_con_sensores(
                df_sensores_list=sums_data_list,
                res_pasada=res_pasada,
                min_indices=indices,
                ciclo_num=ciclo_info['cycle_num'],
                pasada_index=i  # <- se pasa aquí el índice de la pasada
            )
            ciclos_graficados += 1


graficar_suma_multiple_ciclos_con_sensores(filt_der_subset, results_der, min_indices_der, n_ciclos=2)
graficar_suma_multiple_ciclos_con_sensores(filt_izq_subset, results_izq, min_indices_izq, n_ciclos=2)

plt.show()
'''