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





def suavizar_saturacion_df_spline_direccion(df, saturacion, margen=1, delta=2.0):
    """
    Suaviza mesetas saturadas y fuerza forma ascendente o descendente
    según el sensor: S1–S4 (antepié) → ascendente; S5–S8 (retropié) → descendente.
    """
    df_out = df.copy()

    for col in df.columns:
        serie = df[col]
        mask = serie >= saturacion

        if not np.any(mask):
            continue

        # Detectar sensor (S1, S2, ...)
        sensor_name = col.split("_")[-1]  # 'Izquierda_S3' -> 'S3'
        sensor_num = int(sensor_name[1:])  # 'S3' -> 3

        # Decidir forma
        es_antepie = sensor_num in [1, 2, 3, 4]
        es_retropie = sensor_num in [5, 6, 7, 8]

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
                y_mid = max(y0, y1) + delta  # Forzar ascendente
            else:
                y_mid = (y0 + y1) / 2  # Default neutro
            

            x = [t_parche[0], t_parche[len(t_parche)//2], t_parche[-1]]
            y = [y0, y_mid, y1]

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
        muestras_totales = []
        intervalo = 1 / frecuencia_hz
        tolerancia = intervalo * 0.1

        for df in df_list:
            df_copy = df.copy()
            df_copy['Tiempo'] = df_copy['Tiempo'] / 1000.0
            tiempo_real = df_copy['Tiempo'].values

            tiempo_inicial = tiempo_real[0]
            tiempo_final = tiempo_real[-1]
            tiempos_ideales = np.arange(tiempo_inicial, tiempo_final + intervalo, intervalo)

            tiempos_corridos = 0
            muestras_faltantes = 0

            df_copy = df_copy.set_index('Tiempo')

            df_nuevo = []

            for t in tiempos_ideales:
                diffs = np.abs(df_copy.index - t)
                if len(diffs) == 0:
                    muestras_faltantes += 1
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
            muestras_totales.append(len(df_final))  # <-- Guarda total real de muestras

        return dfs_reamostrados, muestras_faltantes_detectadas, muestras_totales


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
    raw_der_final, valores_ausentes_der, muestras_totales_der = limpiar_tiempo_anomalo(raw_der, frecuencia_hz=100)
    raw_izq_final, valores_ausentes_izq, muestras_totales_izq = limpiar_tiempo_anomalo(raw_izq, frecuencia_hz=100)
    
    print(valores_ausentes_der, muestras_totales_der)
    print(valores_ausentes_izq, muestras_totales_izq)
    
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

    saturacion_valor = 25.9  # Valor máximo real para detectar meseta

    # === Pie IZQUIERDO: spline primero, luego filtro ===
    for df in dataframes_izq:
        # 1️⃣ Suavizar mesetas de saturación con spline
        df_suave = suavizar_saturacion_df_spline_direccion(df, saturacion=saturacion_valor, margen=1, delta=3.0)
        
        # 2️⃣ Filtro Savitzky-Golay después para suavizar todo
        df_filtered = df_suave.copy()
        for column in df_filtered.columns:
            df_filtered[column] = savgol_filter(df_filtered[column], window_length=window_length, polyorder=polyorder)
            df_filtered[column] = np.maximum(df_filtered[column], 0)

        filt_izq.append(df_filtered)

    # === Pie DERECHO: spline primero, luego filtro ===
    for df in dataframes_der:
        # 1️⃣ Suavizar mesetas de saturación con spline
        df_suave = suavizar_saturacion_df_spline_direccion(df, saturacion=saturacion_valor, margen=1, delta=2.0)
        
        # 2️⃣ Filtro Savitzky-Golay después para suavizar todo
        df_filtered = df_suave.copy()
        for column in df_filtered.columns:
            df_filtered[column] = savgol_filter(df_filtered[column], window_length=window_length, polyorder=polyorder)
            df_filtered[column] = np.maximum(df_filtered[column], 0)

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

# GRÁFICOS FILT (kg)
# Pie DERECHO 
der = mV_der
if len(der) == 0:
    print("No hay datos de PIE DERECHO para graficar.")
else:
    fig_der_kg, axes = plt.subplots(len(der), 1, figsize=(12, 2.5 * len(der)))
    if len(der) == 1:
        axes = [axes]
    
    for i, df in enumerate(der):
        cols = [col for col in df.columns if col != 'Tiempo']
        df[cols].plot(ax=axes[i])
        axes[i].set_title(f'Pie Derecho - Pasada {i+1}', pad=10)
        axes[i].set_xlabel("Tiempo (s)")
        axes[i].set_ylabel("Voltaje (mV)")
        axes[i].legend()
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
   
izq = mV_izq
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
        axes[i].set_title(f'Pie Izquierdo - Pasada {i+1}', pad=10)
        axes[i].set_xlabel("Tiempo (s)")
        axes[i].set_ylabel("Voltaje (mV)")
        axes[i].legend()
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    

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
        #print(f"[INFO] Detectados {len(peaks)} picos.")

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
                                #print(f"[OK] Mínimo desde pico {pico} en idx={idx}, tiempo={tiempo_minimo} subida={subida:.2f}")
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
                                #print(f"[RESCATE OK] Derivada idx={idx_rescate}, tiempo={tiempo_rescate} subida={subida:.2f}")
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


min_indices_der, min_tiempos_der, ciclos_der, indices_apoyo_der, ciclos_completos_der = segmentar_ciclos_vgrf_desde_picos(sums_der, threshold_ratio=0.04)

# Para el pie izquierdo
min_indices_izq, min_tiempos_izq, ciclos_izq, indices_apoyo_izq, ciclos_completos_izq = segmentar_ciclos_vgrf_desde_picos(sums_izq, threshold_ratio=0.04)


############## DETECCIÓN DE PICOS Y VALLES EN CADA CICLO, CLASIFICACIÓN DE VÁLIDOS/INVÁLIDOS ########################
def improved_vGRF_detection(ciclos_por_pasada, indices_por_pasada, ciclos_completos_por_pasada,
                                time_threshold=0.5, 
                                min_prominence_factor=0.05,
                                min_peak_height=0.1,
                                support_ratio_threshold=(0.60, 0.78)):
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
                "start_index": indices_ciclos[j][0],
                "end_index": indices_ciclos[j][1],
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
                                        support_ratio_threshold=(0.60, 0.78)) 
results_izq = improved_vGRF_detection(ciclos_izq, indices_apoyo_izq, ciclos_completos_izq,
                                        time_threshold=0.5,  
                                        min_prominence_factor=0.05,                                          
                                        min_peak_height=0.1,
                                        support_ratio_threshold=(0.60, 0.78)) 


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
                ax.text(mid_point, y.max() * 0.9,
                        "\n".join(ciclo['exclusion_reasons']),
                        ha='center', va='top', fontsize=7)

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

            datos_ciclo_corr, _ = correccion_baseline_general(
                datos_ciclo.sum(axis=1).to_frame(), modo='suma')
            suma_total = datos_ciclo_corr.iloc[:, 0] / peso_kg

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
    
'''plot_promedios_comparados(promedio_der, lado="R")
plot_promedios_comparados(promedio_izq, lado="L")
'''
#plt.close('all')
#plt.show()


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
def calcular_COP_normalizado_todas_pasadas(
    lista_fuerza, lista_vgrf, coord_df, lista_indices, lista_validos, lado, n_points=100
):
    """
    Devuelve listas anidadas: [[ciclos pasada 1], [ciclos pasada 2], ...]
    Cada ciclo interpolado a n_points.
    Con control de NaNs y prints de debug.
    """
    copx_todas = []
    copy_todas = []

    sufijo = 'Derecha' if lado.upper() == 'R' else 'Izquierda'
    coord_actual = coord_df.copy()
    coord_actual.index = [f"{sufijo}_{s}" for s in coord_actual.index]

    for pasada_idx, (fuerza_df, vgrf_df, indices_ciclos, validos) in enumerate(
        zip(lista_fuerza, lista_vgrf, lista_indices, lista_validos)
    ):
        copx_pasada = []
        copy_pasada = []

        for ciclo_idx, ((ini, fin), es_valido) in enumerate(zip(indices_ciclos, validos)):
            if not es_valido:
                continue

            ciclo_fuerza = fuerza_df[ini:fin]
            ciclo_vgrf = vgrf_df[ini:fin]

            coordenadas = coord_actual.loc[ciclo_fuerza.columns][['X', 'Y']].to_numpy()
            matriz_fuerza = ciclo_fuerza.to_numpy()
            vector_vgrf = ciclo_vgrf.to_numpy().flatten()

            # Reemplaza ceros por NaN para evitar división por cero
            vector_vgrf[vector_vgrf == 0] = np.nan

            cop_x_raw = np.sum(matriz_fuerza * coordenadas[:, 0], axis=1) / vector_vgrf
            cop_y_raw = np.sum(matriz_fuerza * coordenadas[:, 1], axis=1) / vector_vgrf

            '''# Debug rápido
            print(f"[Pasada {pasada_idx+1} | Ciclo {ciclo_idx+1}]")
            print(f"  vGRF: min={np.nanmin(vector_vgrf):.2f}, max={np.nanmax(vector_vgrf):.2f}")
            print(f"  COP X: min={np.nanmin(cop_x_raw):.2f}, max={np.nanmax(cop_x_raw):.2f}")
            print(f"  COP Y: min={np.nanmin(cop_y_raw):.2f}, max={np.nanmax(cop_y_raw):.2f}")'''

            # Filtra NaNs antes de interpolar
            mask = ~np.isnan(cop_x_raw) & ~np.isnan(cop_y_raw)
            if np.sum(mask) < 5:
                print(f"  [WARN] Muy pocos puntos válidos, ciclo ignorado.")
                continue

            # Interpolación solo con puntos válidos
            x_old = np.linspace(0, 100, len(cop_x_raw))
            x_new = np.linspace(0, 100, n_points)

            f_x = interp1d(x_old[mask], cop_x_raw[mask], kind='linear', fill_value='extrapolate')
            f_y = interp1d(x_old[mask], cop_y_raw[mask], kind='linear', fill_value='extrapolate')

            copx_interp = pd.Series(f_x(x_new))
            copy_interp = pd.Series(f_y(x_new))

            copx_pasada.append(copx_interp)
            copy_pasada.append(copy_interp)

        copx_todas.append(copx_pasada)
        copy_todas.append(copy_pasada)

    return copx_todas, copy_todas


# === Calcular COP ===
copx_izq_all, copy_izq_all = calcular_COP_normalizado_todas_pasadas(
    filt_izq, sums_izq, coord_df_izq,
    indices_apoyo_izq, [[d['is_valid'] for d in res['details']] for res in results_izq],
    lado='L'
)

copx_der_all, copy_der_all = calcular_COP_normalizado_todas_pasadas(
    filt_der, sums_der, coord_df_der,
    indices_apoyo_der, [[d['is_valid'] for d in res['details']] for res in results_der],
    lado='R'
)

# === Config común ===
pies = ['Izquierdo', 'Derecho']
coord_dfs = [coord_df_izq, coord_df_der]
sensor_colores = ['blue', 'yellow', 'grey', 'pink', 'green', 'orange', 'purple', 'red']
colores = plt.colormaps['tab10']
margen = 20

'''# === Recorre todas las pasadas ===
for pasada_idx, (copx_izq, copy_izq, copx_der, copy_der) in enumerate(zip(copx_izq_all, copy_izq_all, copx_der_all, copy_der_all)):

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
        ax.set_title(f'Pie {pie} - Pasada {pasada_idx + 1}')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
    '''
    
def graficar_promedio_por_pie(filt_df, results, coord_sensores, pie="Derecho"):
    fases = ["loading_response", "midstance", "terminal_stance", "pre_swing"]
    acumulador = {fase: [] for fase in fases}

    for resultado in results:
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
                tiempo_fin = float(tiempo_fin)
                pico1 = float(peaks[0])
                valle = float(valleys[0])
                pico2 = float(peaks[1])
            except Exception as e:
                continue

            segmentos = {
                "loading_response": (tiempo_inicio, pico1),
                "midstance": (pico1, valle),
                "terminal_stance": (valle, pico2),
                "pre_swing": (pico2, tiempo_fin)
            }

            for fase, (t0, t1) in segmentos.items():
                segmento_df = filt_df[(filt_df.index >= t0) & (filt_df.index <= t1)]
                sumas = segmento_df.sum()
                total = sumas.sum()
                if total > 0:
                    porcentajes = (sumas / total * 100)
                    acumulador[fase].append(porcentajes)

    # Promediar por fase
    promedio_fases = {
        fase: pd.concat(valores, axis=1).mean(axis=1) if valores else pd.Series([0]*8, index=filt_df.columns)
        for fase, valores in acumulador.items()
    }

    # Graficar
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(f"Aporte promedio por sensor - Pie {pie}", fontsize=16)
    fig.patch.set_facecolor("white")
    

    for i, fase in enumerate(fases, 1):
        porcentajes = promedio_fases[fase]
        sensores = porcentajes.index
        sensor_id = lambda s: s.replace("Derecha_", "").replace("Izquierda_", "")
        x = [coord_sensores.loc[sensor_id(s), "X"] for s in sensores]
        y = [coord_sensores.loc[sensor_id(s), "Y"] for s in sensores]
        z = [0] * len(sensores)
        dz = [porcentajes[s] for s in sensores]
        


        ax = fig.add_subplot(1, 4, i, projection='3d')
        ax.bar3d(x, y, z, dx=5, dy=5, dz=dz, shade=True)

        # Estética
        ax.set_facecolor("white")
        ##ax.set_axis_off()   si quiero apagar todo
        ax.grid(False)
        ax.xaxis.line.set_color((1, 1, 1, 0))  # línea eje X
        ax.yaxis.line.set_color((1, 1, 1, 0))  # línea eje Y
        ax.zaxis.line.set_color((1, 1, 1, 0))  # línea eje Z
        
        ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))  # XY plane (suelo)
        ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0)) 
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

        # Etiquetas sobre cada barra
        for xi, yi, zi in zip(x, y, dz):
            ax.text(xi, yi, zi + 2, f"{zi:.1f}%", fontsize=5, ha='left', va='bottom')


        ax.set_xlim(-70, 120) 
        ax.set_ylim(-35, 155)
        ax.set_zlim(0, 60)
       

        ax.set_title(fase.replace("_", " ").title())
  

    plt.tight_layout()
    plt.show()


graficar_promedio_por_pie(filt_der[0], results_der, coord_sensores, pie="Derecho")
graficar_promedio_por_pie(filt_izq[0], results_izq, coord_sensores, pie="Izquierdo")


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


'''
graficar_suma_por_fases_con_sensores_v2(filt_der[0], results_der, indices_apoyo_der, lado="R")
graficar_suma_por_fases_con_sensores_v2(filt_izq[0], results_izq, indices_apoyo_izq, lado="L")
'''


def exportar_tabla_latex_metricas(df_metricas, pie="Derecho", archivo="tabla_metricas.tex"):
    """
    Exporta un DataFrame de métricas con RMS y MAE
    resaltando en negrita los valores altos.
    """

    with open(archivo, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\renewcommand{\\arraystretch}{1.2}\n")
        f.write("\\begin{tabular}{|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Pasada} & \\textbf{Ciclo} & \\textbf{RMS} & \\textbf{MAE} \\\\\n")
        f.write("\\hline\n")

        for _, row in df_metricas.iterrows():
            pasada = int(row['Pasada'])
            ciclo = int(row['Ciclo'])

            if row['RMS_ALTO']:
                rms = f"\\textbf{{{row['RMS']:.4f}}}"
            else:
                rms = f"{row['RMS']:.4f}"

            if row['MAE_ALTO']:
                mae = f"\\textbf{{{row['MAE']:.4f}}}"
            else:
                mae = f"{row['MAE']:.4f}"

            f.write(f"{pasada} & {ciclo} & {rms} & {mae} \\\\\n")
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{Valores de RMS y MAE de cada ciclo por pasada para el pie {pie}.}}\n")
        f.write(f"\\label{{tab:rms_mae_pie_{pie.lower()}}}\n")
        f.write("\\end{table}\n")

    print(f"Tabla LaTeX exportada a {archivo}")


# Supongamos que resumen_total es todo
# O filtrás solo uno:
derecho = tabla_der[tabla_der['Pasada'] <= 5]
izquierdo = tabla_izq[tabla_izq['Pasada'] <= 5]

# Exportar pie derecho:
exportar_tabla_latex_metricas(derecho, pie="Derecho", archivo="tabla_rms_mae_derecho.tex")

# Exportar pie izquierdo:
exportar_tabla_latex_metricas(izquierdo, pie="Izquierdo", archivo="tabla_rms_mae_izquierdo.tex")

