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
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import pearsonr

plt.ioff()

############## CARGA DE DATOS ###################3
c1_pathx = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/clusters/x_Cluster1.csv"
c2_pathx = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/clusters/x_Cluster2.csv"
c1_pathy = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/clusters/y_Cluster1.csv"
c2_pathy = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/clusters/y_Cluster2.csv"

# Lee el archivo CSV y carga los parámetros
xx_1 = pd.read_csv(c1_pathx, header=None).values.flatten()
xx_2 = pd.read_csv(c2_pathx, header=None).values.flatten()
yy_1 = pd.read_csv(c1_pathy, header=None).values.flatten()
yy_2 = pd.read_csv(c2_pathy, header=None).values.flatten()

# Carpeta donde están los archivos
folder_path = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/pasadas"

# Listar todos los archivos CSV en la carpeta
archivos_csv = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# Leer los CSVs y almacenarlos en una lista
dfs = [pd.read_csv(os.path.join(folder_path, f), delimiter=";") for f in archivos_csv]

# Lista de nombres de archivos sin la extensión
variables = [os.path.splitext(f)[0] for f in archivos_csv]

# Filtrar los DataFrames en listas separadas
raw_izq = [df for name, df in zip(variables, dfs) if "izquierda" in name.lower()]
raw_der = [df for name, df in zip(variables, dfs) if "derecha" in name.lower()]

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
        
    return filt_der, filt_izq, sums_der, sums_izq, raw_der_final, raw_izq_final, mV_der, mV_izq

filt_der, filt_izq, sums_der, sums_izq, raw_der_final, raw_izq_final, mV_der, mV_izq = procesar_plantillas(raw_der, raw_izq, xx_1, xx_2, yy_1, yy_2)


##################### CÁLCULO DE % BW CON MEDICIONES APOYANDO UN SOLO PIE #######################

def calculo_bw(sums_der, sums_izq):
    ti_der = 10.00
    tf_der = 15.00
    
    ti_izq = 24.00
    tf_izq =29.00
  
    # Iterar sobre los DataFrames en sums_der
    for i, df in enumerate(sums_der):
        if i == len(sums_der) - 1:  # Último índice de sums_der
            prom_der = df.loc[ti_der:tf_der].mean(numeric_only=True)  # BW derecho

    # Iterar sobre los DataFrames en sums_izq
    for i, df in enumerate(sums_izq):
        if i == len(sums_izq) - 1:  # Último índice de sums_izq
            prom_izq = df.loc[ti_izq:tf_izq].mean(numeric_only=True)  # BW izquierdo
    
    grf_der = []
    grf_izq = []

    # Iterar a través de los dataframes y los valores de BW_med correspondientes
    for df in sums_der:
        # Calcular el porcentaje de GRF respecto al BW correspondiente
        porcentaje_der = df / prom_der

        # Añadir a la lista de porcentajes
        grf_der.append(porcentaje_der)
        
    for df in sums_izq:
        # Calcular el porcentaje de GRF respecto al BW correspondiente
        porcentaje_izq = df / prom_izq

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
    
plt.show()
###################### CALCULO SUMS NORMALIZADO POR BW #####################
grf_der, grf_izq = calculo_bw(sums_der, sums_izq)   

############# SELECCIÓN DE CICLOS VÁLIDOS ##################
def contacto_inicial(sums_der, sums_izq):
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

min_indices_der, min_indices_izq, min_tiempos_der, min_tiempos_izq = contacto_inicial(sums_der, sums_izq)


def compute_avg_cycle(cycles_indices, all_cycles):
    """Genera la curva promedio interpolando todas las curvas a 100 puntos."""
    avg_cycle = np.mean(
    [np.interp(np.linspace(0, 100, num=100), np.linspace(0, 100, num=len(all_cycles[i-1])), all_cycles[i-1])
    for i in cycles_indices], axis=0)

    return avg_cycle

def exclude_by_shape(cycles_indices, all_cycles, avg_cycle):
    shape_diffs = []
    
    for i in cycles_indices:
        cycle = all_cycles[i - 1]
        normalized_cycle = np.interp(np.linspace(0, 100, num=100), np.linspace(0, 100, num=len(cycle)), cycle)
        shape_diff = np.linalg.norm(normalized_cycle - avg_cycle)
        shape_diffs.append(shape_diff)
    
    p75_threshold = np.percentile(shape_diffs, 75)
    
    excluded_shape = [[cycles_indices[i] for i in range(len(shape_diffs)) if shape_diffs[i] > p75_threshold]]  # Asegurar lista de listas
    
    return excluded_shape

def find_cycles_to_exclude(pasadas, min_indices, threshold_time=0.20, threshold_slope=0.3):
    """
    Identifica ciclos anómalos en vGRF basándose en:
    1. Duración atípica, 2. Estructura de picos/valles (vs promedio), 3. Distancia al promedio.
    
    Args:
        pasadas: Lista de DataFrames con señales de vGRF.
        min_indices: Lista de arrays con índices de mínimos.
        threshold_time: Umbral para exclusión por duración (ej. 0.2 = 20% diferencia).
        threshold_slope: Umbral para exclusión por derivada (pendiente).
    
    Returns:
        tuple: (excluded_cycles, remaining_cycles, exclude_duration_list, exclude_shape_list)
    """
    num_pasadas = len(pasadas)
    excluded_cycles = [[] for _ in range(num_pasadas)]
    remaining_cycles = [[] for _ in range(num_pasadas)]
    exclude_duration_list = [[] for _ in range(num_pasadas)]
    exclude_shape_list = [[] for _ in range(num_pasadas)]

    for i, (pasada, mins) in enumerate(zip(pasadas, min_indices)):
        if len(mins) < 2:
            continue
            
        # 1. Obtener todos los ciclos entre mínimos consecutivos
        all_cycles = [pasada.iloc[mins[i]:mins[i+1]].values.flatten() for i in range(len(mins)-1)]
        
        # 2. Exclusión por duración atípica
        durations = [pasada.index[mins[i+1]] - pasada.index[mins[i]] for i in range(len(mins)-1)]
        avg_duration = np.mean(durations)
        duration_diffs = np.abs((durations - avg_duration)/avg_duration)
        exclude_duration = [i+1 for i in np.where(duration_diffs > threshold_time)[0]]
        excluded_cycles[i].extend(exclude_duration)
        exclude_duration_list[i] = exclude_duration
        
        remaining = [i+1 for i in range(len(durations)) if (i+1) not in exclude_duration]

        # 3. Exclusión por forma (comparación con estructura del promedio)
        if remaining:
            avg_cycle = compute_avg_cycle(remaining, all_cycles)
            
            # A. Encontrar picos/valles del PROMEDIO como referencia
            peaks_avg, _ = find_peaks(avg_cycle, height=0.6*np.max(avg_cycle), prominence=0.5)
            valleys_avg, _ = find_peaks(-avg_cycle, height=0.6*np.max(avg_cycle), prominence=0.5)
            
            # B. Umbrales para ciclos individuales (basados en el promedio)
            height_threshold = 0.5 * np.max(avg_cycle)
            prominence_threshold = 0.4
            distance_threshold = len(avg_cycle)//4  # 25% del ciclo
            
            # C. Exclusión por distancia al promedio (método original)
            excluded_by_distance = exclude_by_shape(remaining, all_cycles, avg_cycle)[0]
            
            # D. Exclusión por estructura de picos/valles (vs promedio)
            excluded_by_structure = []
            for idx in remaining:
                cycle = all_cycles[idx-1]
                normalized_cycle = np.interp(np.linspace(0, 100, 100), np.linspace(0, 100, len(cycle)), cycle)
                
                # Detectar picos y valles del ciclo actual
                peaks, _ = find_peaks(normalized_cycle, 
                                    height=height_threshold,
                                    prominence=prominence_threshold,
                                    distance=distance_threshold)
                valleys, _ = find_peaks(-normalized_cycle,
                                      height=height_threshold,
                                      prominence=prominence_threshold,
                                      distance=distance_threshold)
                
                # Condición: Misma cantidad de picos/valles que el promedio
                if not (len(peaks) == len(peaks_avg) and len(valleys) == len(valleys_avg)):
                    excluded_by_structure.append(idx)
            
            # Combinar ambos criterios (excluir si falla en cualquiera)
            exclude_shape = list(set(excluded_by_distance + excluded_by_structure))
            excluded_cycles[i].extend(exclude_shape)
            exclude_shape_list[i] = exclude_shape
            remaining = [idx for idx in remaining if idx not in exclude_shape]

        # 4. Ciclos restantes válidos
        remaining_cycles[i] = remaining
        
    return excluded_cycles, remaining_cycles, exclude_duration_list, exclude_shape_list


# Definir el rango de tiempo deseado
t_inicio = 15  # Cambia estos valores según tu necesidad
t_fin = 30

# Definir el subset
sums_der_pasadas = [sums_der[0], sums_der[4]]
sums_izq_pasadas = [sums_izq[0], sums_izq[4]]

sums_der_subset = subset(sums_der_pasadas, t_inicio, t_fin)
sums_izq_subset = subset(sums_izq_pasadas, t_inicio, t_fin)

min_indices_subset_der, min_indices_subset_izq, min_tiempos_subset_der, min_tiempos_subset_izq = contacto_inicial(sums_der_subset, sums_izq_subset)

excluded_cycles_der, remaining_cycles_der, exclude_duration_der, exclude_shape_der = find_cycles_to_exclude(sums_der_subset, min_indices_subset_der, threshold_time=0.25)
excluded_cycles_izq, remaining_cycles_izq, exclude_duration_izq, exclude_shape_izq = find_cycles_to_exclude(sums_izq_subset, min_indices_subset_izq, threshold_time=0.25)

print(excluded_cycles_der)
print(excluded_cycles_izq)

# Crear una figura con subgráficos (una fila para cada lista)
fig, axes = plt.subplots(2, len(sums_der_subset), figsize=(15, 8), sharex=True)

# Graficar los datos derechos
for i, (df, min_idx, min_tiempo) in enumerate(zip(sums_der_subset, min_indices_subset_der, min_tiempos_subset_der)):
    # Graficar la señal completa
    axes[0, i].plot(df.index, df, label="Señal derecha", color='b')

    # Graficar los mínimos
    axes[0, i].scatter(min_tiempo, df.iloc[min_idx], color='b', zorder=3)
    '''
    # Graficar los picos detectados
    for peaks in detected_peaks_der[i]:
        # Obtener los valores de los picos usando los índices
        peak_values = df.iloc[peaks]  # Esto devuelve los valores de la señal en los índices de los picos
        
        # Graficar los picos sobre la señal
        axes[0, i].scatter(df.index[peaks], peak_values, color='r',  zorder=3)
    '''
    axes[0, i].set_title(f"Señal Derecha {i+1}")
    axes[0, i].legend()

# Graficar los datos izquierdos
for i, (df, min_idx, min_tiempo) in enumerate(zip(sums_izq_subset, min_indices_subset_izq, min_tiempos_subset_izq)):
    # Graficar la señal completa
    axes[1, i].plot(df.index, df, label="Señal izquierda", color='g')

    # Graficar los mínimos
    axes[1, i].scatter(min_tiempo, df.iloc[min_idx], color='b', zorder=3)
    '''
    # Graficar los picos detectados
    for peaks in detected_peaks_izq[i]:
        # Obtener los valores de los picos usando los índices
        peak_values = df.iloc[peaks]  # Esto devuelve los valores de la señal en los índices de los picos
        
        # Graficar los picos sobre la señal
        axes[1, i].scatter(df.index[peaks], peak_values, color='r', zorder=3)
    '''
    axes[1, i].set_title(f"Señal Izquierda {i+1}")
    axes[1, i].legend()

plt.tight_layout()
plt.show()


'''
################### CÁLCULO DE COP #######################
# Crear una lista de listas con datos por filas
coord_sensores = [
    ['S1', 6.4, 152.1],
    ['S2', 0, 119.4],
    ['S3', 28.7, 124.1 ],
    ['S4', 56.8, 119.6],
    ['S5', 54.4, 63.7],
    ['S6', 6.4, -13.9],
    ['S7', 23.4, -29.8],
    ['S8', 41.3, -15.3] ]

# Crear un DataFrame a partir de la lista de listas
df_coord = pd.DataFrame(coord_sensores, columns=['SENSOR', 'X', 'Y'])


def calcular_COP(fuerza_sensores, vgrf, coordenadas_iniciales):
    """
    Calcula el Centro de Presión (COP) en X y Y usando fuerzas de sensores y valores de vgrf.

    Parámetros:
    - fuerza_sensores: Lista de DataFrames, donde cada df tiene fuerzas de sensores en columnas.
    - vgrf: Lista de DataFrames, donde cada df tiene una sola columna con el valor de vgrf por fila.
    - df_coord: DataFrame con las coordenadas de los sensores (columnas 'X' y 'Y').

    Retorna:
    - COPx_list: Lista con los DataFrames de COP en X para cada DataFrame de cada df en fuerza_sensores.
    - COPy_list: Lista con los DataFrames de COP en Y para cada DataFrame de cada df en fuerza_sensores.
    """

    # Extraer coordenadas (X, Y) como lista de tuplas
    coordenadas = coordenadas_iniciales[['X', 'Y']].apply(tuple, axis=1).tolist()

    COPx_list = []  # Lista para almacenar DataFrames de COPx
    COPy_list = []  # Lista para almacenar DataFrames de COPy

    for df_fuerza, df_vgrf in zip(fuerza_sensores, vgrf):
        # Convertir a matrices numpy
        coordenadas_array = np.array(coordenadas)  # (N_sensores, 2)
        matriz_fuerza = df_fuerza.to_numpy()  # (N_filas, N_sensores)
        vector_vgrf = df_vgrf.to_numpy().flatten()  # (N_filas,)

        # Calcular sumas ponderadas de X y Y
        suma_x = np.sum(matriz_fuerza * coordenadas_array[:, 0], axis=1)
        suma_y = np.sum(matriz_fuerza * coordenadas_array[:, 1], axis=1)

        # Calcular COP dividiendo por vgrf
        cop_x = suma_x / vector_vgrf
        cop_y = suma_y / vector_vgrf

        # Crear un DataFrame para cada conjunto de COPs (uno por cada df de fuerza_sensores)
        COPx_df = pd.DataFrame(cop_x, columns=['COPx'])
        COPy_df = pd.DataFrame(cop_y, columns=['COPy'])

        # Asignar el índice correcto del DataFrame original de fuerzas
        COPx_df.index = df_fuerza.index
        COPy_df.index = df_fuerza.index

        # Agregar a las listas
        COPx_list.append(COPx_df)
        COPy_list.append(COPy_df)

    return COPx_list, COPy_list

COPx_der, COPy_der = calcular_COP(filt_der, sums_der, df_coord)
'''
'''
# Crear la grilla
fig, ax = plt.subplots(figsize=(10, 8))

# Configurar el tamaño de la grilla
grid_size = 300
x_min, x_max = df_coord['X'].min() - 10,  df_coord['X'].max() + 10
y_min, y_max =  df_coord['Y'].min() - 10,  df_coord['Y'].max() + 10

# Crear la grilla y pintar el fondo de blanco
grid = np.zeros((grid_size, grid_size))
ax.imshow(grid, extent=[x_min, x_max, y_min, y_max], cmap='gray')

# Agregar círculos en las posiciones de los sensores
circle1 = Circle((df_coord['X'][0], df_coord['Y'][0]), radius=9.15, color='blue', alpha=0.5)
ax.add_patch(circle1)

circle2 = Circle((df_coord['X'][1], df_coord['Y'][1]), radius=9.15, color='yellow', alpha=0.5)
ax.add_patch(circle2)

circle3 = Circle((df_coord['X'][2], df_coord['Y'][2]), radius=9.15, color='grey', alpha=0.5)
ax.add_patch(circle3)

circle4 = Circle((df_coord['X'][3], df_coord['Y'][3]), radius=9.15, color='pink', alpha=0.5)
ax.add_patch(circle4)

circle5 = Circle((df_coord['X'][4], df_coord['Y'][4]), radius=9.15, color='green', alpha=0.5)
ax.add_patch(circle5)

circle8 = Circle((df_coord['X'][5], df_coord['Y'][5]), radius=9.15, color='orange', alpha=0.5)
ax.add_patch(circle8)

circle7 = Circle((df_coord['X'][6], df_coord['Y'][6]), radius=9.15, color='white', alpha=0.5)
ax.add_patch(circle7)

circle6 = Circle((df_coord['X'][7], df_coord['Y'][7]), radius=9.15, color='red', alpha=0.5)
ax.add_patch(circle6)

circle0 = Circle((0, 0), radius=1, color='white', alpha=1)
ax.add_patch(circle0)

# Agregar los puntos de COP (asumiendo que ya tienes las listas COPx y COPy)
ax.plot(COPx_der[0].iloc[2413:2613], COPy_der[0].iloc[2413:2613], label='COP')  # Rojo para COP, puedes ajustar el color y estilo

# Configurar ejes y mostrar el gráfico
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grilla con Sensores y COP')
plt.grid(True)

# Si deseas evitar que se agregue una leyenda repetida para cada COP, puedes agregar:
# plt.legend()  # Si deseas ver la leyenda del COP

plt.show()
'''
'''
# Definir cantidad de pasadas
num_pasadas = max(len(sums_der), len(sums_izq))

# Evitar error si no hay datos
if num_pasadas == 0:
    print("No data to plot.")
else:
    # Crear figura y subplots
    fig, axes = plt.subplots(nrows=num_pasadas * 2, ncols=1, figsize=(10, num_pasadas * 3 * 2), squeeze=False)
    axes = axes.flatten()  # Convert to 1D array for easy indexing

    # Iterar intercaladamente entre sums_der y sums_izq
    for i in range(num_pasadas):
        # Graficar sums_der (si existe)
        if i < len(sums_der):
            sums_der[i].plot(ax=axes[i * 2], legend=True)
            axes[i * 2].set_title(f'Suma de fuerzas (N) Pasada N°{i+1} (Derecha)')
            axes[i * 2].set_xlabel('Tiempo')
            axes[i * 2].set_ylabel('Valores')
            #axes[i * 2].set_xlim(14, 24)  # Ajusta si es necesario

        # Graficar sums_izq (si existe)
        if i < len(sums_izq):
            sums_izq[i].plot(ax=axes[i * 2 + 1], legend=True)
            axes[i * 2 + 1].set_title(f'Suma de fuerzas (N) Pasada N°{i+1} (Izquierda)')
            axes[i * 2 + 1].set_xlabel('Tiempo')
            axes[i * 2 + 1].set_ylabel('Valores')
            #axes[i * 2 + 1].set_xlim(14, 24)  # Ajusta si es necesario

    plt.tight_layout()
    plt.show()


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
    plt.show()
'''