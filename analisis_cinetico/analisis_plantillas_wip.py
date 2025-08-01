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
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import beta
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splprep, splev
from matplotlib import cm
import matplotlib.patheffects as pe
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
    

def suavizar_saturacion_df_spline_direccion(df, margen=3, delta=2.0, saturacion = 25, lado='D'):
    """
    Suaviza mesetas que se quedan en el máximo local de cada sensor.
    - Detecta saturación automática usando el máximo de la señal.
    - S1–S4 (antepié) → forma ascendente
    - S5–S8 (retropié) → forma descendente
    """
    df_out = df.copy()

    for col in df.columns:
        serie = df[col]
        valor_max = serie.max()

    
        # Encontrar zonas donde se queda saturado
        mask = serie >= saturacion

        if not np.any(mask):
            continue

        # Identificar sensor
        sensor_name = col.split("_")[-1]  # Ej: 'Izquierda_S3' → 'S3'
        sensor_num = int(sensor_name[1:])

        es_antepie = sensor_num in [3, 4]

        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if mask.iloc[0]:
            starts = np.insert(starts, 0, 0)
        if mask.iloc[-1]:
            ends = np.append(ends, len(serie))

        serie_suave = serie.copy()

        for s, e in zip(starts, ends):
            i0 = max(s - margen, 0)
            i1 = min(e + margen, len(serie) - 1)
            idx_parche = np.arange(i0, i1 + 1)
            t_parche = serie.index[idx_parche]

            y0 = serie.iloc[i0]
            y1 = serie.iloc[i1]

            if es_antepie:
                y_mid = max(y0, y1) + delta

            
            if es_antepie:
                x = [t_parche[0], t_parche[len(t_parche)//2], t_parche[-1]]
                y = [y0, y_mid, y1]
            else:
                x = [t_parche[0], t_parche[-1]]
                y = [y0, y1]
            
            interpolador = PchipInterpolator(x, y)
            serie_suave.loc[t_parche] = interpolador(t_parche)

        df_out[col] = serie_suave

    return df_out



################ PROCESAMIENTO BASE DE DATOS CRUDOS #######################

def procesar_plantillas(datos_derecha, datos_izquierda):
    def limpiar_tiempo_anomalo(df_list, frecuencia_hz=100):
        dfs_reamostrados = []
        tiempos_corridos_detectados = []
        muestras_faltantes_detectadas = []
        muestras_faltantes_ubicaciones = []  # ✅ nuevo: lista de listas con tiempos
        muestras_totales = []

        intervalo = 1 / frecuencia_hz
        tolerancia = intervalo * 0.1

        for df in df_list:
            df_copy = df.copy()
            df_copy['Tiempo'] = df_copy['Tiempo'] / 1000.0  # pasar a segundos
            tiempo_real = df_copy['Tiempo'].values

            tiempo_inicial = tiempo_real[0]
            tiempo_final = tiempo_real[-1]
            tiempos_ideales = np.arange(tiempo_inicial, tiempo_final + intervalo, intervalo)

            tiempos_corridos = 0
            muestras_faltantes = 0
            ubicaciones_faltantes = []  # ✅ guarda los tiempos ideales donde falta

            df_copy = df_copy.set_index('Tiempo')

            df_nuevo = []

            for t in tiempos_ideales:
                diffs = np.abs(df_copy.index - t)
                if len(diffs) == 0:
                    muestras_faltantes += 1
                    ubicaciones_faltantes.append(t)
                    fila = df_copy[df_copy.index < t].iloc[-1]
                else:
                    idx_min = np.argmin(diffs)
                    diff_min = diffs[idx_min]

                    if diff_min <= tolerancia:
                        if diff_min > 1e-6:
                            tiempos_corridos += 1
                        fila = df_copy.iloc[idx_min]
                    else:
                        muestras_faltantes += 1
                        ubicaciones_faltantes.append(t)
                        fila = df_copy[df_copy.index < t].iloc[-1]

                fila_dict = fila.to_dict()
                fila_dict['Tiempo'] = round(t, 3)
                df_nuevo.append(fila_dict)

            df_final = pd.DataFrame(df_nuevo)
            for col in df_final.columns:
                if col != 'Tiempo':
                    df_final[col] = df_final[col].astype(int)

            dfs_reamostrados.append(df_final)
            tiempos_corridos_detectados.append(tiempos_corridos)
            muestras_faltantes_detectadas.append(muestras_faltantes)
            muestras_faltantes_ubicaciones.append(ubicaciones_faltantes)  # ✅ guardar ubicaciones
            muestras_totales.append(len(df_final))

        return (
            dfs_reamostrados,
            muestras_faltantes_detectadas,
            muestras_totales,
            muestras_faltantes_ubicaciones  # ✅ nuevo output
        )



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
    raw_der_final, valores_ausentes_der, muestras_totales_der, ubicacion_muestras_der = limpiar_tiempo_anomalo(raw_der, frecuencia_hz=100)
    raw_izq_final, valores_ausentes_izq, muestras_totales_izq, ubicaciones_muestras_izq = limpiar_tiempo_anomalo(raw_izq, frecuencia_hz=100)
    
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
            #df_filtered[column] = np.clip(df[column], 0, 25)

        lowpass_der.append(df_filtered)

    # Aplicar filtro a los DataFrames de la izquierda
    for i, df in enumerate(dataframes_izq):
        df_filtered = df.copy()  # Crear una copia del DataFrame para almacenar los datos filtrados

        # Aplicar el filtro a cada columna de sensores
        for column in df.columns:
            df_filtered[column] = apply_lowpass_filter(df[column], cutoff_frequency, sampling_rate)
            #df_filtered[column] = np.clip(df[column], 0, 25)

        lowpass_izq.append(df_filtered)
        
    # Parámetros del filtro Savitzky-Golay
    window_length = 11  # Tamaño de la ventana del filtro (debe ser un número impar)
    polyorder = 3    # Orden del polinomio usado en el filtro

    filt_der = []
    filt_izq = []

    # === Pie IZQUIERDO: spline primero, luego filtro ===
    for df in dataframes_izq:
        # 1️⃣ Suavizar mesetas de saturación con spline
        df_suave = suavizar_saturacion_df_spline_direccion(df, margen=2, delta=2, saturacion = 25, lado='I')
        
        # 2️⃣ Filtro Savitzky-Golay después para suavizar todo
        df_filtered = df_suave.copy()
        for column in df_filtered.columns:
            df_filtered[column] = savgol_filter(df_filtered[column], window_length=window_length, polyorder=polyorder)
        
        filt_izq.append(df_filtered)

    # === Pie DERECHO: spline primero, luego filtro ===
    for df in dataframes_der:
        # 1️⃣ Suavizar mesetas de saturación con spline
        df_suave = suavizar_saturacion_df_spline_direccion(df, margen=1, delta=1, saturacion = 25, lado='D')
        
        # 2️⃣ Filtro Savitzky-Golay después para suavizar todo
        df_filtered = df_suave.copy()
        for column in df_filtered.columns:
            df_filtered[column] = savgol_filter(df_filtered[column], window_length=window_length, polyorder=polyorder)

        filt_der.append(df_filtered)
        
    # === Sumas finales ===
    sums_der = []
    sums_izq = []

    for filtered_df in filt_der:
        sum_der = filtered_df.sum(axis=1)
        sums_der.append(sum_der)

    for filtered_df in filt_izq:
        sum_izq = filtered_df.sum(axis=1)
        sums_izq.append(sum_izq)
        

    # Asegurar formato consistente
    sums_der = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in sums_der]
    sums_izq = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in sums_izq]

    
    return filt_der, filt_izq, sums_der, sums_izq, dataframes_der, dataframes_izq, mV_der, mV_izq, ubicacion_muestras_der, ubicaciones_muestras_izq

#FILT: 8 sensores en kg
#SUMS: suma de sensores en kg
#DATAFRAMES: 8 sensores en kg antes del filtrado
#mV: 8 sensores en mV

filt_der, filt_izq, sums_der, sums_izq, dataframes_der, lowpass_izq, mV_der, mV_izq, ubicaciones_muestras_der, ubicaciones_muestras_izq = procesar_plantillas(raw_der, raw_izq)



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

for i, (sum_der, sum_izq, filt_d, filt_i, mV_d, mV_i) in enumerate(zip(sums_der, sums_izq, filt_der, filt_izq, mV_der, mV_izq)):
    tf_der = sum_der.index[-1]
    tf_izq = sum_izq.index[-1]

    dif = tf_der - tf_izq

    sum_der = sum_der.copy()
    filt_d = filt_d.copy()
    mV_d = mV_d.copy()

    sum_der.index = sum_der.index - dif
    filt_d.index = filt_d.index - dif
    mV_d.index = mV_d.index - dif

    ti = sum_izq.index[0]


    # Filtrar sumas, datos crudos y mV
    sums_der[i] = subset(sum_der, ti, tf_izq)
    sums_izq[i] = subset(sum_izq, ti, tf_izq)

    filt_der[i] = subset(filt_d, ti, tf_izq)
    filt_izq[i] = subset(filt_i, ti, tf_izq)

    mV_der[i] = subset(mV_d, ti, tf_izq)
    mV_izq[i] = subset(mV_i, ti, tf_izq)
    


############### CÁLCULO DE MÍNIMOS DE LA SUMA PARA EXTRAER CICLOS #####################
def correccion_baseline_general(
    data, 
    modo='sensores',  # 'sensores' o 'suma'
    cero_negativos=True
):
    """
    Aplica corrección de baseline (offset mínimo):
    - Puede recibir una lista de DataFrames o un solo DataFrame.
    - 'modo' define si es multicolumna ('sensores') o monocola ('suma').
    - Si cero_negativos=True, fuerza a cero todo valor negativo tras corrección.

    Returns:
        - Corregido (mismo tipo: lista o único)
        - Offset(s)
    """
    # Si es lista → siempre igual
    if isinstance(data, list):
        resultados_corr = []
        offsets = []

        for df in data:
            if modo == 'sensores':
                offset = df.min()
                corr = df - offset
            else:  # suma única
                fuerza = df.iloc[:, 0]
                offset = fuerza.min()
                corr = pd.DataFrame(fuerza - offset, index=df.index)

            if cero_negativos:
                corr[corr < 0] = 0

            resultados_corr.append(corr)
            offsets.append(offset)

        return resultados_corr, offsets

    # Si es único DataFrame
    else:
        df = data
        if modo == 'sensores':
            offset = df.min()
            corr = df - offset
        else:
            fuerza = df.iloc[:, 0]
            offset = fuerza.min()
            corr = pd.DataFrame(fuerza - offset, index=df.index)

        if cero_negativos:
            corr[corr < 0] = 0

        return corr, offset

filt_der, offsets_sens_der = correccion_baseline_general(filt_der, modo='sensores')
filt_izq, offsets_sens_izq = correccion_baseline_general(filt_izq, modo='sensores')
sums_der, offsets_sum_der= correccion_baseline_general(sums_der, modo='suma')
sums_izq, offsets_sum_izq = correccion_baseline_general(sums_izq, modo='suma')



def segmentar_ciclos_vgrf_desde_picos(series_list,
                                      frecuencia_hz=100,
                                      subida_minima=5,
                                      ventana_retroceso_s=0.4,
                                      peso_maximo=5,
                                      min_separacion_s=0.7,
                                      min_prominence_factor=0.01,
                                      min_peak_height=0.07,
                                      threshold_ratio=0.03,
                                      filt_list=None,
                                      lado='I',
                                      aporte_max_rel=0.03):
    """
    Detecta mínimos y segmenta ciclos de marcha.
    Para cada sensor ajusta su nivel para que el aporte relativo en la región no válida sea ≤ aporte_max_rel.
    Limita además la distancia máxima permitida entre CI y su pico asociado.
    """
    min_indices_all = []
    min_tiempos_all = []
    ciclos_apoyo_all = []
    indices_apoyo_all = []
    ciclos_completos_all = []
    suma_corregida_all = []
    filt_corr_list = []

    ventana_retroceso = int(ventana_retroceso_s * frecuencia_hz)
    min_separacion = int(min_separacion_s * frecuencia_hz)
    max_dist_pico_CI = int(0.6 * frecuencia_hz)  # Máx distancia CI–pico: 0.5 s

    if lado == 'I':
        sensores = [f'Izquierda_S{i}' for i in range(1, 9)]
    elif lado == 'D':
        sensores = [f'Derecha_S{i}' for i in range(1, 9)]
    else:
        raise ValueError("El parámetro 'lado' debe ser 'I' o 'D'.")

    for i_passada, serie in enumerate(series_list):
        y = serie.values.ravel()
        tiempos = np.array(serie.index)
        dy = np.diff(y, prepend=y[0])

        max_val = np.max(y)
        min_prominence = max_val * min_prominence_factor
        min_height = max_val * min_peak_height

        peaks, _ = find_peaks(y, prominence=min_prominence, height=min_height)

        min_indices = []

        for pico in peaks:
            ini = max(0, pico - ventana_retroceso)
            min_ini = max(0, pico - max_dist_pico_CI)
            #ini = max(ini, min_ini)  # Aplica restricción de distancia
            fin = pico

            encontrado = False
            for idx in reversed(range(ini + 1, fin)):
                if dy[idx - 1] < 0 and dy[idx] >= 0:
                    if y[idx] < peso_maximo:
                        subida = y[pico] - y[idx]
                        if subida >= subida_minima:
                            if (pico - idx) <= max_dist_pico_CI:
                                if len(min_indices) == 0 or (idx - min_indices[-1]) >= min_separacion:
                                    min_indices.append(idx)
                                    encontrado = True
                                    break

            if not encontrado:
                nueva_ventana = int(0.5 * frecuencia_hz)
                ini_rescate = max(0, pico - nueva_ventana)
                ini_rescate = max(ini_rescate, min_ini)  # Aplica restricción en rescate
                fin_rescate = fin
                dy_rescate = np.diff(y[ini_rescate:fin_rescate], prepend=y[ini_rescate])

                for j in reversed(range(1, len(dy_rescate))):
                    if dy_rescate[j - 1] < 0 and dy_rescate[j] >= 0:
                        idx_rescate = ini_rescate + j
                        subida = y[pico] - y[idx_rescate]
                        if y[idx_rescate] < peso_maximo and subida >= subida_minima:
                            if (pico - idx_rescate) <= max_dist_pico_CI:
                                if len(min_indices) == 0 or (idx_rescate - min_indices[-1]) >= min_separacion:
                                    min_indices.append(idx_rescate)
                                    break

        min_indices = sorted(set(min_indices))
        min_tiempos = tiempos[min_indices]
        min_indices_all.append(min_indices)
        min_tiempos_all.append(min_tiempos)

        ciclos_apoyo = []
        indices_apoyo = []
        ciclos_completos = []

        df_passada = filt_list[i_passada].copy() if filt_list is not None else None

        for i in range(len(min_indices) - 1):
            idx_ini = min_indices[i]
            idx_fin = min_indices[i + 1]

            y_segmento = y[idx_ini:idx_fin]
            peaks_ciclo, _ = find_peaks(y_segmento, prominence=min_prominence, height=min_height)

            if len(peaks_ciclo) >= 1:
                idx_rel_pico1 = peaks_ciclo[0]
                idx_pico1 = idx_ini + idx_rel_pico1
            else:
                idx_pico1 = idx_ini + int((idx_fin - idx_ini) * 0.3)

            if len(peaks_ciclo) >= 2:
                idx_rel_pico2 = peaks_ciclo[1]
                idx_pico2 = idx_ini + idx_rel_pico2
            else:
                idx_pico2 = idx_ini + int((idx_fin - idx_ini) * 0.7)

            if df_passada is not None:
                for sensor in sensores:
                    sensor_num = int(sensor.split('_')[-1][1:])

                    if sensor_num in [1, 2, 3, 4]:
                        reg_ini = idx_ini
                        reg_fin = idx_pico1
                    elif sensor_num in [5, 6, 7, 8]:
                        reg_ini = idx_pico2
                        reg_fin = idx_fin
                    else:
                        continue

                    aporte = df_passada[sensor].iloc[reg_ini:reg_fin].mean()
                    suma_total = df_passada[sensores].iloc[reg_ini:reg_fin].sum(axis=1).mean()

                    aporte_rel = aporte / suma_total if suma_total > 0 else 0
                    if aporte_rel > aporte_max_rel:
                        aporte_deseado = aporte_max_rel * suma_total
                        factor = aporte_deseado / aporte if aporte > 0 else 1
                        desplazamiento = aporte * (1 - factor)

                        corregido = (df_passada[sensor].iloc[idx_ini:idx_fin] - desplazamiento).clip(lower=0)
                        df_passada.loc[df_passada.index[idx_ini:idx_fin], sensor] = corregido

                suma_ciclo = df_passada[sensores].iloc[idx_ini:idx_fin].sum(axis=1).to_frame()
                ciclo_completo = suma_ciclo

                datos = ciclo_completo.values.ravel()
                max_valor = datos.max()
                threshold = threshold_ratio * max_valor
                indices_validos = np.where(datos > threshold)[0]

                if len(indices_validos) > 0:
                    rel_fin = indices_validos[-1]

                    if rel_fin == 0:
                        rel_fin = 1

                    idx_fin_apoyo = idx_ini + rel_fin

                    # SOLO aplica recorte por subida máxima si está después del 2do pico
                    if idx_fin_apoyo > idx_pico2:
                        ventana_chequeo = int(0.05 * frecuencia_hz)  # 50 ms
                        subida_maxima = 3  # por ejemplo

                        while rel_fin > ventana_chequeo:
                            idx_actual = rel_fin
                            idx_prev = rel_fin - ventana_chequeo
                            subida = datos[idx_actual] - datos[idx_prev]

                            if subida >= subida_maxima:
                                # Subida grande => todavía estás en apoyo => retrocede más
                                rel_fin = idx_prev
                            else:
                                # Subida pequeña => señal plana => corto aquí
                                break

                    if rel_fin > (idx_fin - idx_ini):
                        rel_fin = idx_fin - idx_ini

                    apoyo = ciclo_completo.iloc[:rel_fin]
                    if len(apoyo) == 0:
                        apoyo = ciclo_completo

                    t_ini = apoyo.index[0]
                    t_fin = apoyo.index[-1]

                else:
                    apoyo = ciclo_completo
                    t_ini = ciclo_completo.index[0]
                    t_fin = ciclo_completo.index[-1]

                ciclos_completos.append(ciclo_completo)
                ciclos_apoyo.append(apoyo)
                indices_apoyo.append((t_ini, t_fin))

        if df_passada is not None:
            suma_corregida = df_passada[sensores].sum(axis=1).to_frame()
            suma_corregida_all.append(suma_corregida)
            filt_corr_list.append(df_passada)

        ciclos_apoyo_all.append(ciclos_apoyo)
        indices_apoyo_all.append(indices_apoyo)
        ciclos_completos_all.append(ciclos_completos)

    return (
        min_indices_all,
        min_tiempos_all,
        ciclos_apoyo_all,
        indices_apoyo_all,
        ciclos_completos_all,
        suma_corregida_all,
        filt_corr_list
    )




min_indices_izq, min_tiempos_izq, ciclos_izq, indices_apoyo_izq, ciclos_completos_izq, sums_izq, filt_izq = segmentar_ciclos_vgrf_desde_picos(
    sums_izq,
    subida_minima=5,
    ventana_retroceso_s=0.4,
    peso_maximo=5,
    min_separacion_s=0.7,
    min_prominence_factor=0.03,
    min_peak_height=0.07,
    threshold_ratio=0.04,
    filt_list=filt_izq, 
    lado='I',
    aporte_max_rel=0.03)


min_indices_der, min_tiempos_der, ciclos_der, indices_apoyo_der, ciclos_completos_der, sums_der, filt_der = segmentar_ciclos_vgrf_desde_picos(
    sums_der,
    subida_minima=5,
    ventana_retroceso_s=0.4,
    peso_maximo=5,
    min_separacion_s=0.7,
    min_prominence_factor=0.03,
    min_peak_height=0.07,
    threshold_ratio=0.04,
    filt_list=filt_der, 
    lado='D',
    aporte_max_rel=0.03)






############## DETECCIÓN DE PICOS Y VALLES EN CADA CICLO, CLASIFICACIÓN DE VÁLIDOS/INVÁLIDOS ########################

def improved_vGRF_detection(ciclos_por_pasada, indices_por_pasada, ciclos_completos_por_pasada,
                            time_threshold=0.5, 
                            min_prominence_factor=0.05,
                            min_peak_height=0.1,
                            support_ratio_threshold=(0.60, 0.78), 
                            peso_kg=52):
    """
    Detección de patrones vGRF usando ciclos ya recortados, con evaluación de proporción de fase de apoyo.

    Args:
        ciclos_por_pasada: Lista de listas de pd.Series, una por ciclo ya recortado (fase de apoyo)
        indices_por_pasada: Lista de (inicio, fin) en tiempo real de cada apoyo
        ciclos_completos_por_pasada: Lista de listas de pd.Series con los ciclos sin recorte
        Otros: parámetros de detección

    Returns:
        Lista de resultados por pasada, con ciclos válidos/invalidos y diagnóstico.
    """
    resultados = []

    for ciclos_apoyo, indices_ciclos, ciclos_completos in zip(ciclos_por_pasada, indices_por_pasada, ciclos_completos_por_pasada):
        if len(ciclos_apoyo) < 2:
            resultados.append({"valid": [], "invalid": [], "details": []})
            continue

        # Calcular duración total de cada ciclo completo
        duraciones_completas = []
        for ciclo in ciclos_completos:
            if len(ciclo.index) > 0:
                duraciones_completas.append(ciclo.index[-1] - ciclo.index[0])
            else:
                duraciones_completas.append(0)

        median_dur = np.median(duraciones_completas)
        ciclo_resultados = []
        ciclos_validos = []

        for j, (ciclo_apoyo, ciclo_completo) in enumerate(zip(ciclos_apoyo, ciclos_completos)):
            y = ciclo_apoyo.values.flatten()
            max_val = np.max(y)
            min_prominence = max_val * min_prominence_factor
            min_height = max_val * min_peak_height

            try:
                peaks, props = find_peaks(y, prominence=min_prominence, height=min_height)
                valleys, _ = find_peaks(-y, prominence=min_prominence)

                valleys_filtrados = []
                for v in valleys:
                    valor_valle = y[v]
                    if valor_valle >= peso_kg * 0.60:
                        valleys_filtrados.append(v)

                is_valid = True
                razones = []

                duracion_apoyo = indices_ciclos[j][1] - indices_ciclos[j][0]
                duracion_completa = duraciones_completas[j]
                proporcion_apoyo = duracion_apoyo / duracion_completa if duracion_completa > 0 else 0

                if not ((1 - time_threshold) * median_dur < duracion_completa < (1 + time_threshold) * median_dur):
                    is_valid = False
                    razones.append("Duración anómala")

                if len(peaks) < 2:
                    is_valid = False
                    razones.append(f"Solo {len(peaks)} pico(s)")

                if len(peaks) >= 2:
                    primer_pico = y[peaks[0]]
                    segundo_pico = y[peaks[1]]
                    if primer_pico < peso_kg or segundo_pico < peso_kg:
                        is_valid = False
                        razones.append("Pico(s) menor al peso")

                if len(valleys_filtrados) < 1:
                    is_valid = False
                    razones.append("Sin valle central relevante")

                if len(valleys_filtrados) > 1:
                    is_valid = False
                    razones.append(f"{len(valleys_filtrados)} valles detectados")

                if len(ciclo_apoyo) == 0 or len(ciclo_completo) == 0:
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
                "start_index": indices_ciclos[j][0],
                "end_index": indices_ciclos[j][1],
                "peaks": [ciclo_apoyo.index[p] for p in peaks] if len(peaks) else [],
                "valleys": [ciclo_apoyo.index[v] for v in valleys_filtrados] if len(valleys_filtrados) else [],
                "exclusion_reasons": razones,
                "duration": duracion_completa,        # ✅ ciclo completo
                "support_ratio": proporcion_apoyo     # ✅ proporción apoyo / total
            })

        resultados.append({
            "valid": ciclos_validos,
            "invalid": [j for j in range(len(ciclos_apoyo)) if j not in ciclos_validos],
            "details": ciclo_resultados
        })

    return resultados


results_der = improved_vGRF_detection(ciclos_der, indices_apoyo_der, ciclos_completos_der, 
                                        time_threshold=0.5,  # Aumentar para permitir más variación en duración
                                        min_prominence_factor=0.05,  # Reducir para detectar picos menos prominentes
                                        min_peak_height=0.1, # Reducir para incluir picos más pequeños
                                        support_ratio_threshold=(0.60, 0.80),
                                        peso_kg=52) 
results_izq = improved_vGRF_detection(ciclos_izq, indices_apoyo_izq, ciclos_completos_izq,
                                        time_threshold=0.5,  
                                        min_prominence_factor=0.05,                                          
                                        min_peak_height=0.1,
                                        support_ratio_threshold=(0.60, 0.80),
                                        peso_kg=52) 




import matplotlib.pyplot as plt

# Asumo que tenés 5 pasadas
for i in range(5):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Señal derecha
    axes[0].plot(sums_der[i].index, sums_der[i].values, color='b')
    axes[0].set_title(f"Pasada {i+1} - Pie Derecho")
    axes[0].set_ylabel("Señal")

    # Señal izquierda
    axes[1].plot(sums_izq[i].index, sums_izq[i].values, color='g')
    axes[1].set_title(f"Pasada {i+1} - Pie Izquierdo")
    axes[1].set_ylabel("Señal")
    axes[1].set_xlabel("Muestras")

    plt.tight_layout()
    plt.show()





def plot_with_diagnosis(senales_completas, results, indices_ciclos, min_tiempos=None):
    """
    Muestra la señal completa con sombreado en las fases de apoyo:
    - Verde para ciclos válidos
    - Rojo para inválidos
    También muestra picos, valles y los mínimos globales (si se proveen).
    Escala la señal a porcentaje del peso corporal (BW).
    """
    BW = 52  # tu peso en kg

    for i, (df, res, recortes) in enumerate(zip(senales_completas, results, indices_ciclos)):
        fig, ax = plt.subplots(figsize=(12, 4))
        col_name = df.columns[0]
        x = df.index

        # Escalar a %BW
        y = (df[col_name] / BW)

        ax.plot(x, y, color='gray', alpha=0.8, label='vGRF (%BW)')

        # Flags para mostrar solo una vez
        shown_peak = False
        shown_valley = False
        shown_minimo = False

        for ciclo in res['details']:
            num = ciclo['cycle_num']
            if num >= len(recortes):
                continue

            inicio, fin = recortes[num]
            if inicio is None or fin is None:
                continue

            color = 'green' if ciclo['is_valid'] else 'red'
            recorte_x = df.loc[inicio:fin].index
            recorte_y = y.loc[inicio:fin]

            ax.fill_between(recorte_x, 0, recorte_y, color=color, alpha=0.2)

            # Picos y valles
            for peak in ciclo['peaks']:
                if peak in df.index:
                    ax.scatter(
                        peak, (df.loc[peak, col_name] / BW),
                        color='#1f77b4', marker='x', s=40,
                        label='Pico' if not shown_peak else ""
                    )
                    shown_peak = True

            for valley in ciclo['valleys']:
                if valley in df.index:
                    ax.scatter(
                        valley, (df.loc[valley, col_name] / BW),
                        color='#ff7f0e', marker='o', s=20,
                        label='Valle' if not shown_valley else ""
                    )
                    shown_valley = True
            
            if not ciclo['is_valid'] and len(ciclo['exclusion_reasons']) > 0:
                mid_point = inicio + (fin - inicio) / 2
                '''ax.text(mid_point, y.max() * 0.9,
                        "\n".join(ciclo['exclusion_reasons']),
                        ha='center', va='top', fontsize=7)'''

        # Mínimos globales
        if min_tiempos is not None:
            tiempos_minimos = min_tiempos[i]
            for t_min in tiempos_minimos:
                if t_min in df.index:
                    ax.scatter(
                        t_min, (df.loc[t_min, col_name] / BW),
                        color='#9467bd', marker='o', s=20,
                        label='Mínimo' if not shown_minimo else ""
                    )
                    shown_minimo = True

        ax.set_title(f'Pasada {i+1} - Ciclos válidos: {len(res["valid"])}')
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("vGRF")
        ax.legend(loc='upper right')
        plt.tight_layout()


plot_with_diagnosis(sums_der, results_der, indices_apoyo_der, min_tiempos=min_tiempos_der)
plot_with_diagnosis(sums_izq, results_izq, indices_apoyo_izq, min_tiempos=min_tiempos_izq)
plt.show()

############# GRÁFICO IZQUIERDA Y DERECHA INTERCALADOS, SE DIFERENCIA VALIDEZ DE CADA CICLO ####################
def plot_izquierda_derecha(sums_der_subset, sums_izq_subset,
                           min_indices_der, min_indices_izq,
                           results_der, results_izq):
    """
    Grafica la señal completa de ambos pies coloreada por ciclo completo:
    - Verde para ciclos válidos
    - Rojo para inválidos
    Diferencia pie derecho (líneas más fuertes) e izquierdo (más opacos).
    """
    BW = 52
    
    for i, (df_der, df_izq, mins_der, mins_izq, res_der, res_izq) in enumerate(
        zip(sums_der_subset, sums_izq_subset, min_indices_der, min_indices_izq, results_der, results_izq)
    ):
        fig, ax = plt.subplots(figsize=(14, 5))
        col_der = df_der.columns[0]
        col_izq = df_izq.columns[0]

        # Flags para mostrar cada etiqueta una sola vez
        shown_der_valid = False
        shown_der_invalid = False
        shown_izq_valid = False
        shown_izq_invalid = False

        # Pie derecho: línea más fuerte
        for j, ciclo in enumerate(res_der['details']):
            if j + 1 >= len(mins_der):
                continue
            idx_inicio = df_der.index[mins_der[j]]
            idx_fin = df_der.index[mins_der[j+1]]
            tramo = df_der.loc[idx_inicio:idx_fin, col_der] / BW 

            if ciclo['is_valid']:
                label = 'Derecho Válido' if not shown_der_valid else None
                shown_der_valid = True
                color = 'green'
            else:
                label = 'Derecho Inválido' if not shown_der_invalid else None
                shown_der_invalid = True
                color = 'red'

            ax.plot(
                tramo.index, tramo.values,
                color=color,
                alpha=0.9,
                linewidth=1.5,
                label=label
            )

        # Pie izquierdo: línea más suave
        for j, ciclo in enumerate(res_izq['details']):
            if j + 1 >= len(mins_izq):
                continue
            idx_inicio = df_izq.index[mins_izq[j]]
            idx_fin = df_izq.index[mins_izq[j+1]]
            tramo = df_izq.loc[idx_inicio:idx_fin, col_izq] / BW 

            if ciclo['is_valid']:
                label = 'Izquierdo Válido' if not shown_izq_valid else None
                shown_izq_valid = True
                color = 'green'
            else:
                label = 'Izquierdo Inválido' if not shown_izq_invalid else None
                shown_izq_invalid = True
                color = 'red'

            ax.plot(
                tramo.index, tramo.values,
                color=color,
                alpha=0.4,
                linewidth=1.5,
                label=label
            )

        ax.set_title(f'vGRF - Ciclos Válidos/Inválidos')
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("vGRF (% BW)")
        ax.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()

plot_izquierda_derecha(
    sums_der,
    sums_izq,
    min_indices_der,
    min_indices_izq,
    results_der,
    results_izq
)


def extraer_fases(results, señales, lado="der"):
    for i, (pasada, señal_df) in enumerate(zip(results, señales), start=1):
        rows = []
        tiempo_final = 0  # por pasada
        print(f"Tipo de señal_df en pasada {i}: {type(señal_df)}")

        for det in pasada.get("details", []):
            if det.get("is_valid") is True:
                ciclo = det["cycle_num"]
                apoyo_inicio = float(det["start_index"])
                apoyo_fin = float(det["end_index"])
                duracion_total = float(det["duration"])
                duracion_apoyo = apoyo_fin - apoyo_inicio
                duracion_balanceo = duracion_total - duracion_apoyo
                balanceo_inicio = apoyo_fin
                balanceo_fin = balanceo_inicio + duracion_balanceo

                rows.append([i, ciclo, apoyo_inicio, apoyo_fin, balanceo_inicio, balanceo_fin])

                # Actualizar tiempo_final con el fin del balanceo
                tiempo_final = max(tiempo_final, balanceo_fin)
        
        # Obtener el último tiempo del DataFrame
        tiempo_df_final = señal_df.index[-1]

        # Comparar y guardar el mayor
        tiempo_final = max(tiempo_final, tiempo_df_final)

        if rows:
            df = pd.DataFrame(rows, columns=[
                "Pasada", "Ciclo", "Apoyo_inicio", "Apoyo_fin", "Balanceo_inicio", "Balanceo_fin"
            ])

            # Agregar fila con tiempo final
            df.loc[len(df)] = ["", "TIEMPO_FINAL", "", "", "", tiempo_final]

            df.to_csv(f"fases_validas_{lado}_{i}.csv", index=False)



extraer_fases(results_der, sums_der, lado="der")
extraer_fases(results_izq, sums_izq, lado="izq")



############### CORRECCIÓN y GRAFICO DE SENSORES Y CURVA vGRF ################
def graficar_suma_por_fases_normalizada(
    df_sensores_list, resultados, indices_ciclos,
    peso_kg=52, lado="R",
    RMS_ABSOLUTO=0.15, MAE_ABSOLUTO=0.12
):
    """
    Dibuja:
    - Un gráfico por pasada con picos, valles, fases
    - Labels únicos para fases
    - Devuelve tablas de promedio + métricas
    Con flag de RMS/MAE alto basado en umbral global.
    """
    sufijo = "Derecha" if lado.upper() == "R" else "Izquierda"

    colores_fases = {
        'loading_response': '#9467bd',
        'midstance': '#d62728',
        'terminal_stance': '#e377c2',
        'pre_swing': '#2ca02c'
    }

    resultados_df = []
    resumen_metricas = []

    for pasada_idx, (df_sensores, resultado_paso, indices_paso) in enumerate(zip(df_sensores_list, resultados, indices_ciclos)):
        plt.figure(figsize=(14, 7))
        matriz_ciclos = []

        pico_label_usado = False
        valle_label_usado = False

        fases_usadas = {fase: False for fase in colores_fases.keys()}

        for ciclo_info, indices_ciclo in zip(resultado_paso['details'], indices_paso):
            if not ciclo_info['is_valid']:
                continue

            inicio, fin = indices_ciclo[0], indices_ciclo[1]
            datos_ciclo = df_sensores.loc[inicio:fin].copy()

            suma_total = datos_ciclo.sum(axis=1).to_frame().iloc[:, 0] / peso_kg

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

                plt.plot(
                    x_fase, y_fase, '--',
                    color=colores_fases[fase], linewidth=1.5, alpha=0.5,
                    label=fase.replace("_", " ").capitalize() if not fases_usadas[fase] else ""
                )
                fases_usadas[fase] = True

            idx_p1 = datos_ciclo.index.get_loc(p1)
            idx_p2 = datos_ciclo.index.get_loc(p2)
            idx_v = datos_ciclo.index.get_loc(v)

            for p in [idx_p1, idx_p2]:
                plt.scatter(x_original[p], suma_total.iloc[p],
                            marker='x', color='#1f77b4', s=40,
                            label='Pico' if not pico_label_usado else "")
                pico_label_usado = True

            plt.scatter(x_original[idx_v], suma_total.iloc[idx_v],
                        marker='o', color='#ff7f0e', s=30,
                        label='Valle' if not valle_label_usado else "")
            valle_label_usado = True

        if matriz_ciclos:
            matriz = np.vstack(matriz_ciclos)
            promedio = matriz.mean(axis=0)
            std = matriz.std(axis=0)

            plt.plot(x_interp, promedio, '-', color='black',
                     alpha=0.7, linewidth=2, label='Promedio')
            plt.fill_between(x_interp, promedio - std, promedio + std,
                             color='gray', alpha=0.2, label='±1 SD')

            df_promedio = pd.DataFrame({
                'Ciclo (%)': x_interp,
                'Promedio (%BW)': promedio,
                'STD (%BW)': std
            })

            ciclos_metricas = []
            for idx_ciclo, ciclo in enumerate(matriz):
                desvio = ciclo - promedio
                rms = np.sqrt(np.mean(desvio ** 2))
                mae = np.mean(np.abs(desvio))
                ciclos_metricas.append({
                    'Pasada': pasada_idx + 1,
                    'Ciclo': idx_ciclo + 1,
                    'RMS': rms,
                    'MAE': mae,
                    'RMS_ALTO': rms > RMS_ABSOLUTO,
                    'MAE_ALTO': mae > MAE_ABSOLUTO
                })

            df_metricas = pd.DataFrame(ciclos_metricas)

        else:
            df_promedio = pd.DataFrame()
            df_metricas = pd.DataFrame()

        plt.title(f"Pie {sufijo} - Ciclo de apoyo normalizado - Pasada {pasada_idx+1}")
        plt.xlabel("Porcentaje del ciclo de apoyo (%)")
        plt.ylabel("vGRF (% BW)")
        plt.legend(loc='upper right')
        plt.tight_layout()

        resultados_df.append(df_promedio)
        resumen_metricas.append(df_metricas)

    resumen_total = pd.concat(resumen_metricas, ignore_index=True)
    return resultados_df, resumen_metricas, resumen_total


promedio_der, metricas_der, tabla_der = graficar_suma_por_fases_normalizada(
    filt_der, results_der, indices_apoyo_der, lado="R")

promedio_izq, metricas_izq, tabla_izq = graficar_suma_por_fases_normalizada(
    filt_izq, results_izq, indices_apoyo_izq, lado="L")


def plot_promedios_comparados(promedios_list, lado="R"):
    """
    Grafica todos los promedios de ciclo de apoyo de una pierna en un mismo gráfico.
    """
    sufijo = "Derecha" if lado.upper() == "R" else "Izquierda"
    plt.figure(figsize=(12, 6))

    for idx, df_prom in enumerate(promedios_list, 1):
        if df_prom.empty:
            continue

        x = df_prom['Ciclo (%)']
        y = df_prom['Promedio (%BW)']
        plt.plot(x, y, label=f'Pasada {idx}', alpha=0.8)

    plt.title(f"Pie {sufijo} - Comparación de promedios normalizados")
    plt.xlabel("Porcentaje del ciclo de apoyo (%)")
    plt.ylabel("vGRF (% BW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
plot_promedios_comparados(promedio_der, lado="R")
plot_promedios_comparados(promedio_izq, lado="L")


def graficar_suma_por_fases_con_sensores_v2(filt_list, results_list, indices_ciclos_list, lado="R"):
    """
    Para cada pasada:
    - Muestra la suma por fases coloreadas.
    - Muestra cada sensor individual.
    """
    sufijo = "Derecha" if lado.upper() == "R" else "Izquierda"

    for i_passada, (df_sensores, resultados, indices_ciclos) in enumerate(zip(filt_list, results_list, indices_ciclos_list)):
        
        for ciclo_info, indices_ciclo in zip(resultados['details'], indices_ciclos):
            if not ciclo_info['is_valid']:
                continue

            inicio, fin = indices_ciclo[0], indices_ciclo[1]
            datos_ciclo = df_sensores.loc[inicio:fin].copy()
            suma_total = datos_ciclo.sum(axis=1)

            peaks = ciclo_info['peaks']
            valleys = ciclo_info['valleys']
            
            if len(peaks) < 2 or len(valleys) < 1:
                print(f"[Pasada {i_passada+1}] Ciclo {ciclo_info['cycle_num']}: Faltan picos/valles.")
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

            plt.figure(figsize=(14, 7))
            for fase, (start, end) in segmentos.items():
                plt.axvspan(start, end, alpha=0.2, color=colores_fases[fase], label=fase)

            plt.plot(suma_total.index, suma_total.values, '--', label="Suma original")

            colormap = plt.colormaps['tab10']
            for i, sensor in enumerate(datos_ciclo.columns):
                plt.plot(datos_ciclo.index, datos_ciclo[sensor], color=colormap(i), alpha=0.6, label=sensor)
            
            plt.plot(p1, suma_total.loc[p1], 'ro')
            plt.plot(p2, suma_total.loc[p2], 'ro')
            plt.plot(v,  suma_total.loc[v], 'bo')

            plt.title(f"{sufijo} - Pasada {i_passada+1} - Ciclo {ciclo_info['cycle_num']}")
            plt.xlabel("Índice de muestra")
            plt.ylabel("Señal")
            plt.legend()
            plt.tight_layout()

    


graficar_suma_por_fases_con_sensores_v2(filt_der, results_der, indices_apoyo_der, lado="R")
graficar_suma_por_fases_con_sensores_v2(filt_izq, results_izq, indices_apoyo_izq, lado="L")

coord_sensores = pd.DataFrame([
    ['S1', 6.4, 152.1],
    ['S2', 0, 119.4],
    ['S3', 28.7, 124.1],
    ['S4', 56.8, 119.6],
    ['S5', 54.4, 63.7],
    ['S6', 6.4, -13.9],
    ['S7', 23.4, -29.8],
    ['S8', 41.3, -15.3]
], columns=["Sensor", "X", "Y"]).set_index("Sensor")


def graficar_promedio_por_pie_2D(filt_df_list, results, coord_sensores, pie="Derecho"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib import cm
    from scipy.interpolate import splprep, splev
    import matplotlib.patheffects as pe

    fases = ["loading_response", "midstance", "terminal_stance", "pre_swing"]
    sensores_orden = ["S4", "S3", "S1", "S2", "S5", "S6", "S7", "S8"]

    coord_plot = coord_sensores.copy()

    # === Offsets contorno para DERECHO ===
    offsets_contorno_der = {
        "S1": (0, 5),
        "S2": (-15, 0),
        "S3": (0, 20),
        "S4": (-15, 20),
        "S5": (-35, 0),
        "S6": (-15, -5),
        "S7": (0, -10),
        "S8": (9, -5)
    }

    for pasada_idx, (resultado, pasada_df) in enumerate(zip(results, filt_df_list)):

        acumulador = {fase: [] for fase in fases}

        for detalle in resultado["details"]:
            if not detalle["is_valid"]:
                continue

            tiempo_inicio = detalle.get("start_index")
            tiempo_fin = detalle.get("end_index")
            peaks = detalle.get("peaks", [])
            valleys = detalle.get("valleys", [])

            if tiempo_inicio is None or tiempo_fin is None or len(peaks) < 2 or len(valleys) < 1:
                continue

            try:
                tiempo_inicio = float(tiempo_inicio)
                pico1 = float(peaks[0])
                valle = float(valleys[0])
                pico2 = float(peaks[1])
            except Exception:
                continue

            segmentos = {
                "loading_response": (tiempo_inicio, pico1),
                "midstance": (pico1, valle),
                "terminal_stance": (valle, pico2),
                "pre_swing": (pico2, float(tiempo_fin))
            }

            for fase, (t0, t1) in segmentos.items():
                segmento_df = pasada_df[(pasada_df.index >= t0) & (pasada_df.index <= t1)]
                sumas = segmento_df.sum()
                total = sumas.sum()
                if total > 0:
                    porcentajes = (sumas / total * 100)
                    acumulador[fase].append(porcentajes)

        promedio_fases = {
            fase: pd.concat(valores, axis=1).mean(axis=1) if valores else pd.Series([0]*8, index=pasada_df.columns)
            for fase, valores in acumulador.items()
        }

        max_global = max([p.max() for p in promedio_fases.values()])

        X = coord_plot["X"]
        Y = coord_plot["Y"]
        xc_ref = np.mean(X)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Aporte promedio por sensor (%) | Pie {pie} | Pasada {pasada_idx+1}", fontsize=14)

        cmap = cm.get_cmap('jet')

        for i, (fase, ax) in enumerate(zip(fases, axes)):
            porcentajes = promedio_fases[fase]

            plantilla_X = []
            plantilla_Y = []

            for sid in sensores_orden:
                x, y = X[sid], Y[sid]

                if pie == "Izquierdo":
                    x = 2*xc_ref - x

                posibles = [sid, f"{pie}_{sid}", f"Derecha_{sid}", f"Izquierda_{sid}"]
                valor = None
                for clave in posibles:
                    if clave in porcentajes.index:
                        valor = porcentajes[clave]
                        break
                if valor is None:
                    valor = 0

                radio = 5 + (valor / max_global) * 15
                color = cmap(valor / max_global)

                shadow = Circle((x, y), radius=radio + 3, color=color, alpha=0.2, zorder=1)
                ax.add_patch(shadow)

                circ = Circle((x, y), radius=radio, color=color, alpha=0.8, ec=color, lw=0.5)
                ax.add_patch(circ)

                ax.text(
                    x, y, f"{valor:.1f}%",
                    ha='center', va='center',
                    fontsize=7, color='white', weight='bold',
                    zorder=3,
                    path_effects=[pe.withStroke(linewidth=1, foreground='black')]
                )

                plantilla_X.append(x)
                plantilla_Y.append(y)

            xc, yc = np.mean(plantilla_X), np.mean(plantilla_Y)
            factor_expand = 1.2

            plantilla_X_exp = []
            plantilla_Y_exp = []

            for sid, xi, yi in zip(sensores_orden, plantilla_X, plantilla_Y):
                dx, dy = offsets_contorno_der[sid]
                if pie == "Izquierdo":
                    dx = -dx
                x_exp = xc + (xi - xc) * factor_expand + dx
                y_exp = yc + (yi - yc) * factor_expand + dy
                plantilla_X_exp.append(x_exp)
                plantilla_Y_exp.append(y_exp)

            x_closed = np.append(plantilla_X_exp, plantilla_X_exp[0])
            y_closed = np.append(plantilla_Y_exp, plantilla_Y_exp[0])

            tck, u = splprep([x_closed, y_closed], s=0, per=True)
            unew = np.linspace(0, 1.0, 300)
            out = splev(unew, tck)

            ax.plot(out[0], out[1], lw=1.5, color='black')

            ax.set_title(r"$\mathit{" + fase.replace("_", r"\ ") + "}$")
            ax.set_facecolor("white")
            ax.set_xlim(X.min() - 40, X.max() + 40)
            ax.set_ylim(Y.min() - 50, Y.max() + 50)
            ax.set_aspect('equal')
            ax.axis('off')

        # === Barra de colores global para todo el gráfico ===
        norm = plt.Normalize(vmin=0, vmax=max_global)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Ubicar la barra fuera a la derecha
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
        cbar.set_label('Aporte (%)', fontsize=10)

        plt.tight_layout()
        plt.show()




graficar_promedio_por_pie_2D(filt_der, results_der, coord_sensores, pie='Derecho')
graficar_promedio_por_pie_2D(filt_izq, results_izq, coord_sensores, pie='Izquierdo')


# === Coordenadas de sensores ===
coord_df = pd.DataFrame([
    ['S1', 6.4, 182.0],
    ['S2', 0.0, 149.3],
    ['S3', 28.7, 154.0],
    ['S4', 56.8, 149.5],
    ['S5', 54.4, 93.6],
    ['S6', 6.4, 16.0],
    ['S7', 23.4, 0.1],
    ['S8', 41.3, 14.6]
], columns=["Sensor", "X", "Y"]).set_index("Sensor")



# Reflejo en X para el pie izquierdo
coord_df_izq = coord_df.copy()
coord_df_izq["X"] = -coord_df_izq["X"]

# Offsets para que (0,0) no quede pegado al borde
offset_izq = abs(coord_df_izq["X"].min()) + 10
offset_der = abs(coord_df["X"].min()) + 10

coord_df_izq["X"] += offset_izq
coord_df_der = coord_df.copy()
coord_df_der["X"] += offset_der


def calcular_COP_normalizado_todas_pasadas(
    lista_fuerza, lista_vgrf, coord_df, lista_indices, lista_validos, lado,
    n_points=100, umbral_global=0.07, umbral_sensor=0.07, min_points=10
):
    """
    Devuelve:
      - copx_todas: lista de listas (COP X por ciclo)
      - copy_todas: lista de listas (COP Y por ciclo)
      - fuerza_interp_todas: lista de listas de DF (1 DF por ciclo)
      - vgrf_interp_todas: lista de listas de Series (1 Serie por ciclo)
    """
    copx_todas = []
    copy_todas = []
    fuerza_interp_todas = []
    vgrf_interp_todas = []

    sufijo = 'Derecha' if lado.upper() == 'R' else 'Izquierda'
    coord_actual = coord_df.copy()
    coord_actual.index = [f"{sufijo}_{s}" for s in coord_actual.index]

    for pasada_idx, (fuerza_df, vgrf_df, indices_ciclos, validos) in enumerate(
        zip(lista_fuerza, lista_vgrf, lista_indices, lista_validos)
    ):
        copx_pasada = []
        copy_pasada = []
        fuerza_pasada = []
        vgrf_pasada = []

        for ciclo_idx, ((ini, fin), es_valido) in enumerate(zip(indices_ciclos, validos)):
            if not es_valido:
                continue

            ciclo_fuerza = fuerza_df[ini:fin]
            ciclo_vgrf = vgrf_df[ini:fin]

            coordenadas = coord_actual.loc[ciclo_fuerza.columns][['X', 'Y']].to_numpy()
            matriz_fuerza = ciclo_fuerza.to_numpy()
            vector_vgrf = ciclo_vgrf.to_numpy().flatten()

            # === Filtro de fuerza mínima por sensor ===
            matriz_fuerza[matriz_fuerza < umbral_sensor] = 0

            # === COP ===
            vector_vgrf_cop = vector_vgrf.copy()
            umbral_vgrf = umbral_global * np.nanmax(vector_vgrf)
            vector_vgrf_cop[vector_vgrf_cop < umbral_vgrf] = np.nan

            cop_x_raw = np.sum(matriz_fuerza * coordenadas[:, 0], axis=1) / vector_vgrf_cop
            cop_y_raw = np.sum(matriz_fuerza * coordenadas[:, 1], axis=1) / vector_vgrf_cop

            mask = ~np.isnan(cop_x_raw) & ~np.isnan(cop_y_raw)
            if np.sum(mask) < min_points:
                continue

            # === Interpolación normalizada con recorte ===
            x_old = np.linspace(0, 100, len(vector_vgrf))
            x_new = np.linspace(0, 100, n_points)

            f_x = interp1d(x_old[mask], cop_x_raw[mask],
                           kind='linear', fill_value=np.nan, bounds_error=False)
            f_y = interp1d(x_old[mask], cop_y_raw[mask],
                           kind='linear', fill_value=np.nan, bounds_error=False)

            copx_interp = pd.Series(f_x(x_new))
            copy_interp = pd.Series(f_y(x_new))

            # === ⚡ BLOQUE CLAVE: recorte fuera del rango confiable ===
            x_min = x_old[mask].min()
            x_max = x_old[mask].max()

            copx_interp[x_new < x_min] = np.nan
            copx_interp[x_new > x_max] = np.nan
            copy_interp[x_new < x_min] = np.nan
            copy_interp[x_new > x_max] = np.nan

            copx_pasada.append(copx_interp)
            copy_pasada.append(copy_interp)

            # === Interpolar fuerzas y vGRF ===
            fuerzas_interp = []
            for i in range(matriz_fuerza.shape[1]):
                f_s = interp1d(x_old, matriz_fuerza[:, i],
                               kind='linear', fill_value=np.nan, bounds_error=False)
                fuerzas_interp.append(f_s(x_new))
            fuerzas_interp = np.vstack(fuerzas_interp).T

            f_vgrf = interp1d(x_old, vector_vgrf,
                              kind='linear', fill_value=np.nan, bounds_error=False)
            vgrf_interp = f_vgrf(x_new)

            df_fuerza = pd.DataFrame(fuerzas_interp, columns=ciclo_fuerza.columns)
            s_vgrf = pd.Series(vgrf_interp, name="vGRF")

            fuerza_pasada.append(df_fuerza)
            vgrf_pasada.append(s_vgrf)

        copx_todas.append(copx_pasada)
        copy_todas.append(copy_pasada)
        fuerza_interp_todas.append(fuerza_pasada)
        vgrf_interp_todas.append(vgrf_pasada)

    return copx_todas, copy_todas, fuerza_interp_todas, vgrf_interp_todas




# === Calcular COP ===
copx_izq_all, copy_izq_all, filt_izq, sums_izq = calcular_COP_normalizado_todas_pasadas(
    filt_izq, sums_izq, coord_df_izq,
    indices_apoyo_izq, [[d['is_valid'] for d in res['details']] for res in results_izq],
    lado='L',
    n_points = 100
)

copx_der_all, copy_der_all, filt_der, sums_der = calcular_COP_normalizado_todas_pasadas(
    filt_der, sums_der, coord_df_der,
    indices_apoyo_der, [[d['is_valid'] for d in res['details']] for res in results_der],
    lado='R',
    n_points = 100
)


# === Config común ===
pies = ['Izquierdo', 'Derecho']
coord_dfs = [coord_df_izq, coord_df_der]
sensor_colores = ['blue', 'yellow', 'grey', 'pink', 'green', 'orange', 'purple', 'red']
colores = plt.colormaps['tab10']
margen = 20

# === Recorre todas las pasadas ===
for pasada_idx, (copx_izq, copy_izq, copx_der, copy_der) in enumerate(
    zip(copx_izq_all, copy_izq_all, copx_der_all, copy_der_all)
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    copx_all = [copx_izq, copx_der]
    copy_all = [copy_izq, copy_der]

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

        # Trayectorias COP por ciclo
        for i, (cop_x, cop_y) in enumerate(zip(copx_list, copy_list)):
            ax.plot(cop_x, cop_y, linestyle='--', color=colores(i % 10), alpha=0.6)

        # Promedio COP
        if copx_list:
            copx_prom = pd.concat(copx_list, axis=1).mean(axis=1)
            copy_prom = pd.concat(copy_list, axis=1).mean(axis=1)
            ax.plot(copx_prom, copy_prom, color='black', linewidth=2, label='COP promedio')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'Pie {pie} - Pasada {pasada_idx + 1}')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
