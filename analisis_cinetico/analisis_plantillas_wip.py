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
from scipy.interpolate import interp1d

plt.ioff() #si pones solo un plt.show() al final del codigo se abren todas las imagenes juntas

# Definir la ruta base
base_path = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/calibracion_indiv_sin_21_06/"

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

            # Si el tiempo parece estar en milisegundos, lo paso a segundos 
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

############### CÁLCULO DE MÍNIMOS DE LA SUMA PARA EXTRAER CICLOS #####################
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
    

(filt_der, filt_izq, sums_der, sums_izq, offsets_sens_der, offsets_sens_izq, offsets_sum_der, offsets_sum_izq) = correccion_baseline(filt_der, filt_izq, sums_der, sums_izq)
 

'''# GRÁFICOS FILT (kg)
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
    plt.subplots_adjust(hspace=0.4)'''


def segmentar_ciclos_vgrf(series_list, frecuencia_hz=100, subida_minima=3, post_min_muestras=10, threshold_ratio=0.05):
    """
    Detección de mínimos válidos (contacto inicial) y segmentación de ciclos de marcha,
    combinando cambio de pendiente (derivada) + validación de subida posterior,
    y barrido de rescate para zonas sin mínimos claros.
    """
    min_indices_all = []
    min_tiempos_all = []
    ciclos_apoyo_all = []
    indices_apoyo_all = []
    ciclos_completos_all = []

    for serie in series_list:
        y = serie.values.ravel()
        tiempos = np.array(serie.index)

        dy = np.diff(y, prepend=y[0])

        min_indices = []
        min_separacion = int(0.7 * frecuencia_hz)  # separación mínima entre mínimos principales

        peso_maximo = 5

        candidatos = []

        for idx in range(post_min_muestras, len(y)):
            if dy[idx - 1] < 0 and dy[idx] >= 0:
                ventana_post = y[idx + 1: idx + 1 + post_min_muestras]
                subida = ventana_post.max() - y[idx]

                if subida >= subida_minima:
                    ventana_post = int(0.2 * frecuencia_hz)
                    fin = min(len(y), idx + ventana_post)
                    ventana_posterior = y[idx:fin]
                    if len(ventana_posterior) == 0:
                        continue

                    idx_local_min = np.argmin(ventana_posterior)
                    idx_minimo_real = idx + idx_local_min

                    if y[idx_minimo_real] < peso_maximo:
                        candidatos.append(idx_minimo_real)

        # Filtrar por separación mínima entre principales
        candidatos = sorted(set(candidatos))
        for idx_minimo_real in candidatos:
            if len(min_indices) == 0 or (idx_minimo_real - min_indices[-1]) >= min_separacion:
                min_indices.append(idx_minimo_real)


        ### --- BARRIDO DE RESCATE: robusto con derivada local en todos los gaps ---

        rescates = []
        umbral_gap = int(1.7 * frecuencia_hz)
        margen_gap = int(0.05 * frecuencia_hz)
        umbral_cercania = int(0.4 * frecuencia_hz)
        ventana = int(0.4 * frecuencia_hz)
        peso_maximo = 5

        # --- GAP INICIAL ---
        if len(min_indices) > 0 and min_indices[0] > umbral_gap:
            ini = 0
            fin = min_indices[0] - margen_gap
            if fin > ini:
                print(f"[DEBUG] Revisando GAP INICIAL: ini={tiempos[ini]:.3f}, fin={tiempos[fin]:.3f}")
                dy_gap = np.diff(y[ini:fin], prepend=y[ini])
                for j in range(1, len(dy_gap)):
                    if dy_gap[j - 1] < 0 and dy_gap[j] >= 0:
                        idx_local = ini + j
                        fin_ventana = min(len(y), idx_local + ventana)
                        subida = y[fin_ventana:fin_ventana + post_min_muestras].max() - y[idx_local]
                        if y[idx_local] < peso_maximo:
                            if subida >= subida_minima:
                                too_close = any(abs(idx_local - idx_existente) < umbral_cercania for idx_existente in min_indices + rescates)
                                if not too_close:
                                    rescates.append(idx_local)
                                    print(f"[DEBUG - INICIAL] AGREGADO idx={idx_local} valor={y[idx_local]:.2f} subida={subida:.2f}")
                                else:
                                    print(f"[DEBUG - INICIAL] IGNORADO idx={idx_local} por cercanía")
                            else:
                                print(f"[DEBUG - INICIAL] NO guardado idx={idx_local} subida={subida:.2f} < {subida_minima}")
                        else:
                            print(f"[DEBUG - INICIAL] DESCARTADO idx={idx_local} valor={y[idx_local]:.2f} > peso_maximo")

        # --- GAPS INTERMEDIOS ---
        for i in range(len(min_indices) - 1):
            gap = min_indices[i + 1] - min_indices[i]
            if gap > umbral_gap:
                ini = min_indices[i] + margen_gap
                fin = min_indices[i + 1] - margen_gap
                if fin > ini:
                    print(f"[DEBUG] Revisando GAP INTERMEDIO: ini={tiempos[ini]:.3f}, fin={tiempos[fin]:.3f}")
                    dy_gap = np.diff(y[ini:fin], prepend=y[ini])
                    for j in range(1, len(dy_gap)):
                        if dy_gap[j - 1] < 0 and dy_gap[j] >= 0:
                            idx_local = ini + j
                            fin_ventana = min(len(y), idx_local + ventana)
                            subida = y[fin_ventana:fin_ventana + post_min_muestras].max() - y[idx_local]
                            if y[idx_local] < peso_maximo:
                                if subida >= subida_minima:
                                    too_close = any(abs(idx_local - idx_existente) < umbral_cercania for idx_existente in min_indices + rescates)
                                    if not too_close:
                                        rescates.append(idx_local)
                                        print(f"[DEBUG - INTERMEDIO] AGREGADO idx={idx_local} subida={subida:.2f}")
                                    else:
                                        print(f"[DEBUG - INTERMEDIO] IGNORADO idx={idx_local} por cercanía")
                                else:
                                    print(f"[DEBUG - INTERMEDIO] NO guardado idx={idx_local} subida={subida:.2f} < {subida_minima}")
                            else:
                                print(f"[DEBUG - INTERMEDIO] DESCARTADO idx={idx_local} valor={y[idx_local]:.2f} > peso_maximo")

        # --- GAP FINAL ---
        if len(min_indices) > 0 and (len(y) - min_indices[-1]) > umbral_gap:
            ini = min_indices[-1] + margen_gap
            fin = len(y)
            if fin > ini:
                print(f"[DEBUG] Revisando GAP FINAL: ini={tiempos[ini]:.3f}, fin={tiempos[fin-1]:.3f}")
                dy_gap = np.diff(y[ini:fin], prepend=y[ini])
                for j in range(1, len(dy_gap)):
                    if dy_gap[j - 1] < 0 and dy_gap[j] >= 0:
                        idx_local = ini + j
                        fin_ventana = min(len(y), idx_local + ventana)
                        fin_max = min(len(y), fin_ventana + post_min_muestras)
                        if fin_max > fin_ventana:
                            subida = y[fin_ventana:fin_max].max() - y[idx_local]
                        else:
                            subida = 0  # Si no hay ventana, la subida es cero (no cumple)
                        if y[idx_local] < peso_maximo:
                            if subida >= subida_minima:
                                too_close = any(abs(idx_local - idx_existente) < umbral_cercania for idx_existente in min_indices + rescates)
                                if not too_close:
                                    rescates.append(idx_local)
                                    print(f"[DEBUG - FINAL] AGREGADO idx={idx_local} subida={subida:.2f}")
                                else:
                                    print(f"[DEBUG - FINAL] IGNORADO idx={idx_local} por cercanía")
                            else:
                                print(f"[DEBUG - FINAL] NO guardado idx={idx_local} subida={subida:.2f} < {subida_minima}")
                        else:
                            print(f"[DEBUG - FINAL] DESCARTADO idx={idx_local} valor={y[idx_local]:.2f} > peso_maximo")

        min_indices = sorted(set(min_indices))
        min_tiempos = tiempos[min_indices]
        min_indices_all.append(min_indices)
        min_tiempos_all.append(min_tiempos)

        ### --- Segmentar ciclos ---
        ciclos_apoyo = []
        indices_apoyo = []
        ciclos_completos = []

        for i in range(len(min_indices) - 1):
            idx_ini = min_indices[i]
            idx_fin = min_indices[i + 1]
            ciclo_completo = serie.iloc[idx_ini:idx_fin]
            ciclos_completos.append(ciclo_completo)

            datos = ciclo_completo.values.ravel()
            max_valor = datos.max()
            threshold = threshold_ratio * max_valor
            indices_validos = np.where(datos > threshold)[0]

            if len(indices_validos) > 0:
                rel_fin = indices_validos[-1]

                # Protege: si queda igual
                if rel_fin == 0:
                    rel_fin = 1
                if rel_fin > len(ciclo_completo):
                    rel_fin = len(ciclo_completo)

                apoyo = ciclo_completo.iloc[:rel_fin]
                if len(apoyo) == 0:
                    apoyo = ciclo_completo

                t_ini = apoyo.index[0]
                t_fin = apoyo.index[-1]
            else:
                apoyo = ciclo_completo
                t_ini = ciclo_completo.index[0]
                t_fin = ciclo_completo.index[-1]

            ciclos_apoyo.append(apoyo)
            indices_apoyo.append((t_ini, t_fin))

        ciclos_apoyo_all.append(ciclos_apoyo)
        indices_apoyo_all.append(indices_apoyo)
        ciclos_completos_all.append(ciclos_completos)


    return min_indices_all, min_tiempos_all, ciclos_apoyo_all, indices_apoyo_all, ciclos_completos_all


# Para el pie derecho
min_indices_der, min_tiempos_der, ciclos_der, indices_apoyo_der, ciclos_completos_der = segmentar_ciclos_vgrf(sums_der)

# Para el pie izquierdo
min_indices_izq, min_tiempos_izq, ciclos_izq, indices_apoyo_izq, ciclos_completos_izq = segmentar_ciclos_vgrf(sums_izq)

def segmentar_ciclos_vgrf_desde_picos(series_list,
                                      frecuencia_hz=100,
                                      subida_minima=3,
                                      ventana_retroceso_s=0.4,
                                      peso_maximo=5,
                                      min_separacion_s=0.7,
                                      min_prominence_factor=0.01,
                                      min_peak_height=0.07,
                                      threshold_ratio=0.03):
    """
    Detección de mínimos válidos (contacto inicial) y segmentación de ciclos de marcha,
    combinando cambio de pendiente (derivada) + validación de subida posterior,
    y barrido de rescate para zonas sin mínimos claros.
    """
    min_indices_all = []
    min_tiempos_all = []
    ciclos_apoyo_all = []
    indices_apoyo_all = []
    ciclos_completos_all = []

    ventana_retroceso = int(ventana_retroceso_s * frecuencia_hz)
    min_separacion = int(min_separacion_s * frecuencia_hz)

    for serie in series_list:
        y = serie.values.ravel()
        tiempos = np.array(serie.index)
        dy = np.diff(y, prepend=y[0])

        max_val = np.max(y)
        min_prominence = max_val * min_prominence_factor
        min_height = max_val * min_peak_height

        peaks, props = find_peaks(y, prominence=min_prominence, height=min_height)
        print(f"[INFO] Detectados {len(peaks)} picos.")

        min_indices = []

        for pico in peaks:
            ini = max(0, pico - ventana_retroceso)
            fin = pico

            encontrado = False
            for idx in reversed(range(ini + 1, fin)):
                if dy[idx - 1] < 0 and dy[idx] >= 0:
                    if y[idx] < peso_maximo:
                        subida = y[pico] - y[idx]
                        if subida >= subida_minima:
                            if len(min_indices) == 0 or (idx - min_indices[-1]) >= min_separacion:
                                min_indices.append(idx)
                                encontrado = True
                                tiempo_minimo = tiempos[idx]
                                print(f"[OK] Mínimo desde pico {pico} en idx={idx}, tiempo={tiempo_minimo} subida={subida:.2f}")
                                break

            if not encontrado:
                nueva_ventana = int(0.6 * frecuencia_hz)
                ini_rescate = max(0, pico - nueva_ventana)
                fin_rescate = fin
                dy_rescate = np.diff(y[ini_rescate:fin_rescate], prepend=y[ini_rescate])

                for j in reversed(range(1, len(dy_rescate))):
                    if dy_rescate[j - 1] < 0 and dy_rescate[j] >= 0:
                        idx_rescate = ini_rescate + j
                        subida = y[pico] - y[idx_rescate]
                        if y[idx_rescate] < peso_maximo and subida >= subida_minima:
                            if len(min_indices) == 0 or (idx_rescate - min_indices[-1]) >= min_separacion:
                                min_indices.append(idx_rescate)
                                tiempo_rescate = tiempos[idx_rescate]
                                print(f"[RESCATE OK] Derivada idx={idx_rescate}, tiempo={tiempo_rescate} subida={subida:.2f}")
                                break

        min_indices = sorted(set(min_indices))
        min_tiempos = tiempos[min_indices]
        min_indices_all.append(min_indices)
        min_tiempos_all.append(min_tiempos)

        ### --- Segmentar ciclos ---
        ciclos_apoyo = []
        indices_apoyo = []
        ciclos_completos = []

        for i in range(len(min_indices) - 1):
            idx_ini = min_indices[i]
            idx_fin = min_indices[i + 1]
            ciclo_completo = serie.iloc[idx_ini:idx_fin]
            ciclos_completos.append(ciclo_completo)

            datos = ciclo_completo.values.ravel()
            max_valor = datos.max()
            threshold = threshold_ratio * max_valor
            indices_validos = np.where(datos > threshold)[0]

            if len(indices_validos) > 0:
                rel_fin = indices_validos[-1]

                # Protege: si queda igual
                if rel_fin == 0:
                    rel_fin = 1
                if rel_fin > len(ciclo_completo):
                    rel_fin = len(ciclo_completo)

                apoyo = ciclo_completo.iloc[:rel_fin]
                if len(apoyo) == 0:
                    apoyo = ciclo_completo

                t_ini = apoyo.index[0]
                t_fin = apoyo.index[-1]
            else:
                apoyo = ciclo_completo
                t_ini = ciclo_completo.index[0]
                t_fin = ciclo_completo.index[-1]

            ciclos_apoyo.append(apoyo)
            indices_apoyo.append((t_ini, t_fin))

        ciclos_apoyo_all.append(ciclos_apoyo)
        indices_apoyo_all.append(indices_apoyo)
        ciclos_completos_all.append(ciclos_completos)


    return min_indices_all, min_tiempos_all, ciclos_apoyo_all, indices_apoyo_all, ciclos_completos_all


'''min_indices_der, min_tiempos_der, ciclos_der, indices_apoyo_der, ciclos_completos_der = segmentar_ciclos_vgrf_desde_picos(sums_der)

# Para el pie izquierdo
min_indices_izq, min_tiempos_izq, ciclos_izq, indices_apoyo_izq, ciclos_completos_izq = segmentar_ciclos_vgrf_desde_picos(sums_izq)'''



def segmentar_ciclos_vgrf_desde_picos(series_list,
                                      frecuencia_hz=100,
                                      subida_minima=3,
                                      ventana_retroceso_s=0.4,
                                      peso_maximo=5,
                                      min_separacion_s=0.7,
                                      min_prominence_factor=0.01,
                                      min_peak_height=0.07,
                                      threshold_ratio=0.03):
    """
    Detecta mínimos hacia atrás desde picos detectados internamente.
    El rescate también usa derivada.
    Además, mejora la detección del fin de apoyo combinando umbral y diferencia controlada con el mínimo inicial.
    """
    min_indices_all = []
    min_tiempos_all = []
    ciclos_apoyo_all = []
    indices_apoyo_all = []
    ciclos_completos_all = []

    ventana_retroceso = int(ventana_retroceso_s * frecuencia_hz)
    min_separacion = int(min_separacion_s * frecuencia_hz)

    for serie in series_list:
        y = serie.values.ravel()
        tiempos = np.array(serie.index)
        dy = np.diff(y, prepend=y[0])

        max_val = np.max(y)
        min_prominence = max_val * min_prominence_factor
        min_height = max_val * min_peak_height

        peaks, props = find_peaks(y, prominence=min_prominence, height=min_height)
        print(f"[INFO] Detectados {len(peaks)} picos.")

        min_indices = []

        for pico in peaks:
            ini = max(0, pico - ventana_retroceso)
            fin = pico

            encontrado = False
            for idx in reversed(range(ini + 1, fin)):
                if dy[idx - 1] < 0 and dy[idx] >= 0:
                    if y[idx] < peso_maximo:
                        subida = y[pico] - y[idx]
                        if subida >= subida_minima:
                            if len(min_indices) == 0 or (idx - min_indices[-1]) >= min_separacion:
                                min_indices.append(idx)
                                encontrado = True
                                tiempo_minimo = tiempos[idx]
                                print(f"[OK] Mínimo desde pico {pico} en idx={idx}, tiempo={tiempo_minimo} subida={subida:.2f}")
                                break

            if not encontrado:
                nueva_ventana = int(0.7 * frecuencia_hz)
                ini_rescate = max(0, pico - nueva_ventana)
                fin_rescate = fin
                dy_rescate = np.diff(y[ini_rescate:fin_rescate], prepend=y[ini_rescate])

                for j in reversed(range(1, len(dy_rescate))):
                    if dy_rescate[j - 1] < 0 and dy_rescate[j] >= 0:
                        idx_rescate = ini_rescate + j
                        subida = y[pico] - y[idx_rescate]
                        if y[idx_rescate] < peso_maximo and subida >= subida_minima:
                            if len(min_indices) == 0 or (idx_rescate - min_indices[-1]) >= min_separacion:
                                min_indices.append(idx_rescate)
                                tiempo_rescate = tiempos[idx_rescate]
                                print(f"[RESCATE OK] Derivada idx={idx_rescate}, tiempo={tiempo_rescate} subida={subida:.2f}")
                                break

        min_indices = sorted(set(min_indices))
        min_tiempos = tiempos[min_indices]
        min_indices_all.append(min_indices)
        min_tiempos_all.append(min_tiempos)

        ciclos_apoyo = []
        indices_apoyo = []
        ciclos_completos = []

        for i in range(len(min_indices) - 1):
            idx_ini = min_indices[i]
            idx_fin = min_indices[i + 1]

            ciclo_completo = serie.iloc[idx_ini:idx_fin]
            ciclos_completos.append(ciclo_completo)

            datos = ciclo_completo.values.ravel()
            # 1️⃣ Empieza desde el final del ciclo
            ultimo_idx = len(datos) - 1

            # 2️⃣ Define ventana de retroceso (p.ej. 200 ms)
            ventana_subida = int(0.4 * frecuencia_hz)
            inicio_busqueda = max(0, ultimo_idx - ventana_subida)

            # 3️⃣ Define un umbral mínimo de pendiente (por ejemplo: +0.5 kg por muestra)
            pendiente_min = 0.4  # Ajustable según escala

            rel_fin = ultimo_idx
            for k in reversed(range(inicio_busqueda + 1, ultimo_idx)):
                delta = abs(datos[k] - datos[k - 1])
                print(f'el delta para {k} es {delta}')
                if delta > pendiente_min:
                    rel_fin = k - 1
                    break 

            # Failsafe
            if rel_fin <= 0:
                rel_fin = 1

            apoyo = ciclo_completo.iloc[:rel_fin]
            if len(apoyo) == 0:
                apoyo = ciclo_completo

            t_ini = apoyo.index[0]
            t_fin = apoyo.index[-1]

            ciclos_apoyo.append(apoyo)
            indices_apoyo.append((t_ini, t_fin))

        ciclos_apoyo_all.append(ciclos_apoyo)
        indices_apoyo_all.append(indices_apoyo)
        ciclos_completos_all.append(ciclos_completos)

    return min_indices_all, min_tiempos_all, ciclos_apoyo_all, indices_apoyo_all, ciclos_completos_all

'''# Para el pie derecho
min_indices_der, min_tiempos_der, ciclos_der, indices_apoyo_der, ciclos_completos_der = segmentar_ciclos_vgrf_desde_picos(sums_der)

# Para el pie izquierdo
min_indices_izq, min_tiempos_izq, ciclos_izq, indices_apoyo_izq, ciclos_completos_izq = segmentar_ciclos_vgrf_desde_picos(sums_izq)'''


############## DETECCIÓN DE PICOS Y VALLES EN CADA CICLO, CLASIFICACIÓN DE VÁLIDOS/INVÁLIDOS ########################
def improved_vGRF_detection(ciclos_por_pasada, indices_por_pasada, ciclos_completos_por_pasada,
                                time_threshold=0.5, 
                                min_prominence_factor=0.05,
                                min_peak_height=0.1,
                                support_ratio_threshold=(0.6, 0.8)):
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

        tiempos_inicio_apoyo = [i[0] for i in indices_ciclos if i[0] is not None]
        tiempos_fin_apoyo = [i[1] for i in indices_ciclos if i[1] is not None]
        duraciones = [fin - ini for ini, fin in zip(tiempos_inicio_apoyo, tiempos_fin_apoyo)]
        
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

                # --- FILTRO: solo valles mayores a 30 kg ---
                valleys_filtrados = []
                for v in valleys:
                    valor_valle = y[v]
                    if valor_valle >= 30:
                        valleys_filtrados.append(v)

                is_valid = True
                razones = []

                if not ((1 - time_threshold) * median_dur < duracion < (1 + time_threshold) * median_dur):
                    is_valid = False
                    razones.append("Duración anómala")

                if len(peaks) < 2:
                    is_valid = False
                    razones.append(f"Solo {len(peaks)} pico(s)")

                if len(valleys_filtrados) < 1:
                    is_valid = False
                    razones.append("Sin valle central relevante")

                if len(valleys_filtrados) > 1:
                    is_valid = False
                    razones.append(f"{len(valleys_filtrados)} valles detectados")

                duracion_apoyo = duracion
                duracion_completo = ciclo_completo.index[-1] - ciclo_completo.index[0]
                
                proporcion_apoyo = duracion_apoyo / duracion_completo if duracion_completo > 0 else 0
                
                if len(ciclo_apoyo) == 0 or len(ciclo_completo) == 0:
                    # No tiene sentido calcular duración
                    proporcion_apoyo = 0    

                if not (support_ratio_threshold[0] <= proporcion_apoyo <= support_ratio_threshold[1]):
                    is_valid = False
                    razones.append(f"Proporción de apoyo fuera de rango ({proporcion_apoyo:.2f})")

            except Exception as e:
                is_valid = False
                peaks = []
                valleys_filtrados = []
                razones = [f"Error: {str(e)}"]

            if is_valid:
                ciclos_validos.append(j)

            ciclo_resultados.append({
                "cycle_num": j,
                "is_valid": is_valid,
                "peaks": [ciclo_apoyo.index[p] for p in peaks] if len(peaks) else [],
                "valleys": [ciclo_apoyo.index[v] for v in valleys_filtrados] if len(valleys_filtrados) else [],
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

results_der = improved_vGRF_detection(ciclos_der, indices_apoyo_der, ciclos_completos_der, 
                                        time_threshold=0.5,  # Aumentar para permitir más variación en duración
                                        min_prominence_factor=0.05,  # Reducir para detectar picos menos prominentes
                                        min_peak_height=0.1, # Reducir para incluir picos más pequeños
                                        support_ratio_threshold=(0.6, 0.77)) 
results_izq = improved_vGRF_detection(ciclos_izq, indices_apoyo_izq, ciclos_completos_izq,
                                        time_threshold=0.5,  
                                        min_prominence_factor=0.05,                                          
                                        min_peak_height=0.1,
                                        support_ratio_threshold=(0.6, 0.77)) 


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


plot_with_diagnosis(sums_der, results_der, indices_apoyo_der, min_tiempos=min_tiempos_der)
plot_with_diagnosis(sums_izq, results_izq, indices_apoyo_izq, min_tiempos=min_tiempos_izq)

plt.show()
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
    indices_apoyo_izq[0], [d['is_valid'] for d in results_izq[0]['details']], lado='L'
)

copx_der_list, copy_der_list = calcular_COP_por_ciclos_validos(
    filt_der[0], sums_der[0], coord_df_der,
    indices_apoyo_der[0], [d['is_valid'] for d in results_der[0]['details']], lado='R'
)

'''# === Graficar ===
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

plt.tight_layout()'''



############### CORRECCIÓN y GRAFICO DE SENSORES Y CURVA vGRF ################ 
'''def graficar_suma_por_fases_con_sensores_v2(df_sensores, resultados, indices_ciclos, lado="R"):

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

        sensores_por_fase = {
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
graficar_suma_por_fases_con_sensores_v2(filt_izq[0], results_izq, indices_ciclos_izq, lado="L")'''


def graficar_suma_por_fases_normalizada(df_sensores_list, resultados, indices_ciclos, peso_kg=52, lado="R"):
    """
    Dibuja y devuelve un gráfico por cada pasada.
    """
    sufijo = "Derecha" if lado.upper() == "R" else "Izquierda"

    colores_fases = {
        'loading_response': 'red',
        'midstance': 'green',
        'terminal_stance': 'blue',
        'pre_swing': 'orange'
    }

    resultados_df = []

    for pasada_idx, (df_sensores, resultado_paso, indices_paso) in enumerate(zip(df_sensores_list, resultados, indices_ciclos)):
        
        plt.figure(figsize=(14, 7))
        matriz_ciclos = []

        for ciclo_info, indices_ciclo in zip(resultado_paso['details'], indices_paso):
            if not ciclo_info['is_valid']:
                continue

            inicio, fin = indices_ciclo[0], indices_ciclo[1]
            datos_ciclo = df_sensores.loc[inicio:fin].copy()

            suma_total = (datos_ciclo.sum(axis=1) / peso_kg)

            peaks = ciclo_info['peaks']
            valleys = ciclo_info['valleys']

            if len(peaks) < 2 or len(valleys) < 1:
                continue

            p1, p2 = peaks[0], peaks[1]
            v = valleys[0]

            segmentos = {
                'loading_response': (inicio, p1),
                'midstance': (p1, v),
                'terminal_stance': (v, p2),
                'pre_swing': (p2, fin)
            }

            x_original = np.linspace(0, 100, len(suma_total))
            x_interp = np.linspace(0, 100, 100)

            f_interp = interp1d(x_original, suma_total.values, kind='linear')
            ciclo_interp = f_interp(x_interp)
            matriz_ciclos.append(ciclo_interp)

            for fase, (start, end) in segmentos.items():
                idx_start = datos_ciclo.index.get_loc(start)
                idx_end = datos_ciclo.index.get_loc(end)

                x_fase = x_original[idx_start:idx_end+1]
                y_fase = suma_total.iloc[idx_start:idx_end+1].values

                plt.plot(x_fase, y_fase, '--', color=colores_fases[fase], alpha=0.4)

            idx_p1 = datos_ciclo.index.get_loc(p1)
            idx_p2 = datos_ciclo.index.get_loc(p2)
            idx_v = datos_ciclo.index.get_loc(v)

            plt.plot(x_original[idx_p1], suma_total.iloc[idx_p1], 'ro')
            plt.plot(x_original[idx_p2], suma_total.iloc[idx_p2], 'ro')
            plt.plot(x_original[idx_v],  suma_total.iloc[idx_v],  'bo')

        if matriz_ciclos:
            matriz = np.vstack(matriz_ciclos)
            promedio = matriz.mean(axis=0)
            std = matriz.std(axis=0)

            plt.plot(x_interp, promedio, '-', color='black', alpha=0.7, linewidth=2, label='Promedio')
            plt.fill_between(x_interp, promedio - std, promedio + std, color='black', alpha=0.2, label='±1 SD')

            df_resultado = pd.DataFrame({
                'Ciclo (%)': x_interp,
                'Promedio (%BW)': promedio,
                'STD (%BW)': std
            })
        else:
            df_resultado = pd.DataFrame()

        plt.title(f"Todos los ciclos válidos - Pie {sufijo} (Pasada {pasada_idx+1}, normalizados, promedio ± STD)")
        plt.xlabel("Porcentaje del ciclo de apoyo (%)")
        plt.ylabel("vGRF (% BW)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        resultados_df.append(df_resultado)

    return resultados_df


desvio_ciclos_der = graficar_suma_por_fases_normalizada(filt_der, results_der, indices_apoyo_der, lado="R")
desvio_ciclos_izq = graficar_suma_por_fases_normalizada(filt_izq, results_izq, indices_apoyo_izq, lado="L")


plt.show()

'''
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
'''


