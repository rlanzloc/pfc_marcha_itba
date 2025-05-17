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
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.signal import butter, filtfilt, savgol_filter
import os

plt.ioff()

############## CARGA DE DATOS ###################3
# Lista de sensores y pies (ajusta esto según tus datos)
sensor_pie_list = [
    'Derecha_S1', 'Derecha_S2', 'Derecha_S3', 'Derecha_S4', 
    'Derecha_S5', 'Derecha_S6', 'Derecha_S7', 'Derecha_S8',
    'Izquierda_S1', 'Izquierda_S2', 'Izquierda_S3', 'Izquierda_S4', 
    'Izquierda_S5', 'Izquierda_S6', 'Izquierda_S7', 'Izquierda_S8'
]

# Carpeta donde están los archivos
folder_path = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/pasadas/pasadas_sin_proteccion"

# Función sigmoide y su inversa
def sigmoid_decay(x, a, b, c, d):
    return (a - d) / (1 + (x/c)**b) + d

def inverse_sigmoid(y, a, b, c, d):
    """Inversa de la sigmoide decreciente f(x) = (a-d)/(1 + (x/c)^b) + d"""
    return c * ((a - y) / (y - d))**(1/b)

# Función para detectar delimitador
def detect_delimiter(file_path, sample_size=5):
    delimiters = [',', ';', '\t']
    with open(file_path, 'r') as f:
        sample = [f.readline() for _ in range(sample_size)]
    for delimiter in delimiters:
        counts = [line.count(delimiter) for line in sample]
        if all(count == counts[0] for count in counts) and counts[0] > 0:
            return delimiter
    return ','

# Función para aplicar filtro pasa bajos
def apply_lowpass_filter(data, cutoff_freq, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return np.maximum(filtered_data, 0)

parametros_df = pd.read_csv('C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/parametros_sigmoides.csv')

# Función principal de procesamiento
def procesar_plantillas(folder_path, parametros_df):
    # Leer archivos CSV
    dfs = []
    archivos_csv = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    for f in archivos_csv:
        file_path = os.path.join(folder_path, f)
        delimiter = detect_delimiter(file_path)
        df = pd.read_csv(file_path, delimiter=delimiter)
        df = df.drop(columns=["Hora"], errors="ignore")
        dfs.append(df)
    
    variables = [os.path.splitext(f)[0] for f in archivos_csv]
    print(variables)
    raw_izq = [df for name, df in zip(variables, dfs) if "izquierda" in name.lower()]
    raw_der = [df for name, df in zip(variables, dfs) if "derecha" in name.lower()]

    # Funciones auxiliares
    def limpiar_valores_anomalos(df_list, valor_maximo=1023):
        dfs_limpios = []
        for df in df_list:
            columnas_sensores = [col for col in df.columns if col != 'Tiempo']
            df_copy = df.copy()
            df_copy[columnas_sensores] = df_copy[columnas_sensores].where(
                df_copy[columnas_sensores] <= valor_maximo, np.nan)
            dfs_limpios.append(df_copy)
        return dfs_limpios

    def interp(df_list, frecuencia_hz=100):
        dfs_interpolados = []
        for df in df_list:
            df_copy = df.copy()
            df_copy['Tiempo_diff'] = df_copy['Tiempo'].diff()
            
            if len(df_copy[df_copy['Tiempo_diff'] != 1/frecuencia_hz]) > 0:
                tiempo_inicial = df_copy['Tiempo'].iloc[0]
                tiempo_final = df_copy['Tiempo'].iloc[-1]
                tiempos_nuevos = np.linspace(tiempo_inicial, tiempo_final, len(df_copy))
                
                df_interpolado = df_copy.copy()
                for columna in df_copy.columns:
                    if columna != 'Tiempo':
                        df_interpolado[columna] = np.interp(
                            tiempos_nuevos, df_copy['Tiempo'], df_copy[columna])
                        df_interpolado[columna] = df_interpolado[columna].ffill().fillna(0).astype(int)
                
                df_interpolado['Tiempo'] = tiempos_nuevos
                df_interpolado.drop(columns=['Tiempo_diff'], inplace=True)
                dfs_interpolados.append(df_interpolado)
            else:
                dfs_interpolados.append(df_copy)
        return dfs_interpolados

    def preproc_df(dataframes, R):
        for df in dataframes:
            # Crear una copia del DataFrame para no modificar el original
            # Normalizar el primer valor de Tiempo a 0
            # Restar el primer valor para que el primer tiempo sea 0
            df['Tiempo'] = (df['Tiempo'] - df['Tiempo'].iloc[0])/1000
        
        mV_dataframes = []
        dataframes_transformados = []
        for i, df in enumerate(dataframes):
            df_copy = df.copy()
            df_copy.set_index('Tiempo', inplace=True)
            df_copy = df_copy.apply(lambda s: s.where((s >= 0) & (s <= 1023), None)).interpolate(limit_direction='both')
            
            # Convertir a mV
            df_mV = df_copy.map(lambda x: max(1, int((x / 1023) * 5000)) if pd.notnull(x) else 1)
            mV_dataframes.append(df_mV)
            
            def voltaje_a_resistencia(x, R):
                return ((5000 - x) * R) / x

            # Aplicar la función al DataFrame transformado
            df_processed = df_mV.apply(lambda x: voltaje_a_resistencia(x, R))
            #print(df_processed)
            
            parametros = {}
            for _, row in parametros_df.iterrows():
                sensor = row['sensor']
                parametros[sensor] = {
                    'a': row['a'],
                    'b': row['b'],
                    'c': row['c'],
                    'd': row['d']
                }

            # Función para aplicar la transformación a una columna
            def transformar_columna(columna, params):
                a = params['a']
                b = params['b']
                c = params['c']
                d = params['d']

                return columna.apply(lambda x: inverse_sigmoid(x,a,b,c,d))

            # Aplicar la transformación a cada columna del DataFrame original
            df_transformado = pd.DataFrame()
            for columna in df_processed.columns:
                if columna in parametros:
                    df_transformado[columna] = transformar_columna(df_processed[columna], parametros[columna])
                else:
                    df_transformado[columna] = df_processed[columna]
            
            dataframes_transformados.append(df_transformado)
                    
        return dataframes_transformados, mV_dataframes

    # Procesamiento principal
    raw_der_proc = limpiar_valores_anomalos(raw_der)
    raw_izq_proc = limpiar_valores_anomalos(raw_izq)
    
    raw_der_final = interp(raw_der_proc, frecuencia_hz=100)
    raw_izq_final = interp(raw_izq_proc, frecuencia_hz=100)
    
    dataframes_izq, mV_izq = preproc_df(raw_izq_final, 10000)
    dataframes_der, mV_der = preproc_df(raw_der_final, 3300)
    
    # Aplicar filtros
    sampling_rate = 100
    cutoff_frequency = 20
    
    lowpass_der = []
    for df in dataframes_der:
        df_filtered = df.copy()
        for column in df.columns:
            df_filtered[column] = apply_lowpass_filter(df[column], cutoff_frequency, sampling_rate)
        lowpass_der.append(df_filtered)
    
    lowpass_izq = []
    for df in dataframes_izq:
        df_filtered = df.copy()
        for column in df.columns:
            df_filtered[column] = apply_lowpass_filter(df[column], cutoff_frequency, sampling_rate)
        lowpass_izq.append(df_filtered)

    # Filtro Savitzky-Golay
    window_length = 11
    polyorder = 3
    
    filt_der = []
    for df in lowpass_der:
        df_filtered = df.copy()
        for column in df.columns:
            df_filtered[column] = savgol_filter(df_filtered[column], window_length=window_length, polyorder=polyorder)
            df_filtered[column] = np.maximum(df_filtered[column], 0)
        filt_der.append(df_filtered)
    
    filt_izq = []
    for df in lowpass_izq:
        df_filtered = df.copy()
        for column in df.columns:
            df_filtered[column] = savgol_filter(df_filtered[column], window_length=window_length, polyorder=polyorder)
            df_filtered[column] = np.maximum(df_filtered[column], 0)
        filt_izq.append(df_filtered)

    # Calcular sumas
    sums_der = [df.sum(axis=1) for df in filt_der]
    sums_izq = [df.sum(axis=1) for df in filt_izq]
    
    return sums_der, sums_izq, filt_der, filt_izq, dataframes_der, dataframes_izq, raw_der_final, raw_izq_final


sums_der, sums_izq, filt_der, filt_izq, dataframes_der, dataframes_izq, raw_der_final, raw_izq_final = procesar_plantillas(folder_path, parametros_df)


#print(dataframes_der)
#print(dataframes_izq)

##################### CÁLCULO DE % BW CON MEDICIONES APOYANDO UN SOLO PIE #######################
'''
def calculo_bw(sums_der, sums_izq):
    # Iterar sobre los DataFrames en sums_der
    for i, df in enumerate(sums_der):
        if i == len(sums_der) - 1:
            prom_der_4 = df[13.00:20.00].mean()  # BW derecho
            
        if i == len(sums_der) - 2:
            prom_der_3 = df[12.00:20.00].mean()  # BW derecho

        if i == len(sums_der) - 3:
            prom_der_2 = df[6.00:14.00].mean()  # BW derecho


    # Iterar sobre los DataFrames en sums_izq
    for i, df in enumerate(sums_izq):
        if i == len(sums_izq) - 1:
            prom_izq_4 = df[13.00:20.00].mean()  # BW derecho
        
        if i == len(sums_izq) - 2:
            prom_izq_3 = df[10.00:18.00].mean()  # BW derecho

        if i == len(sums_izq) - 3:
            prom_izq_2 = df[2.00:10.00].mean()  # BW derecho
    grf_der = []
    grf_izq = []
    
    print(prom_der_2, prom_der_3, prom_der_4)
    print(prom_izq_2, prom_izq_3, prom_izq_4)
    
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
    sums_izq[i] = subset(sum_izq, ti, tf_izq)  # Filtra sum_izq y actualiza sums_izq

###################### CALCULO SUMS NORMALIZADO POR BW #####################
grf_der, grf_izq = calculo_bw(sums_der, sums_izq)
'''
from analisis_cinetico_BW_calibracion_indiv import sums_der as sums_der_V, sums_izq as sums_izq_V, filt_der as filt_der_V, filt_izq as filt_izq_V 

# ==================================================
# GRÁFICOS DE FUERZA (KG)
# ==================================================

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
    
    
# Tomar solo las últimas 2 pasadas de sums_der
sums_der_BW = []
for df in sums_der:
    df_new = df/51
    sums_der_BW.append(df_new) 

sums_der_last2 = sums_der_BW

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
    
# Tomar solo las últimas 2 pasadas de sums_izq
sums_izq_BW = []
for df in sums_izq:
    df_new = df/51
    sums_izq_BW.append(df_new) 

sums_izq_last2 = sums_izq_BW

# Evitar error si no hay datos
if not sums_izq_last2:
    print("No hay datos para graficar (sums_izq tiene menos de 2 pasadas).")
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
        pasada.plot(ax=axes[i], label="Izquierda", color='g')  # Cambiado a verde para izquierda
        axes[i].set_title(f'Suma de fuerzas (N) - Pasada N°{i + 1}')  # Numeración desde 1
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valores')
        axes[i].legend()

    plt.tight_layout()
    

# Definir las pasadas que quieres graficar (ejemplo: últimas 3 pasadas)
sums_der_pasadas = sums_der  # DataFrames de fuerza derecha
sums_der_V_pasadas = sums_der_V  # DataFrames de velocidad derecha
sums_izq_pasadas = sums_izq # DataFrames de fuerza izquierda 
sums_izq_V_pasadas = sums_izq_V # DataFrames de velocidad izquierda

# Calcular el número máximo de pasadas entre ambos lados
num_pasadas = max(len(sums_der_pasadas), len(sums_izq_pasadas))

# Evitar error si no hay datos
if num_pasadas == 0:
    print("No hay datos para graficar.")
else:
    # Crear figura para la derecha
    fig_der, axes_der = plt.subplots(nrows=num_pasadas, ncols=1, 
                                   figsize=(10, num_pasadas * 3), 
                                   squeeze=False)
    axes_der = axes_der.flatten()
    
    # Crear figura para la izquierda
    fig_izq, axes_izq = plt.subplots(nrows=num_pasadas, ncols=1, 
                                   figsize=(10, num_pasadas * 3), 
                                   squeeze=False)
    axes_izq = axes_izq.flatten()

    # Iterar sobre las pasadas seleccionadas
    for i in range(num_pasadas):
        # Gráfico de la derecha
        if i < len(sums_der_pasadas):
            sums_der_pasadas[i].plot(ax=axes_der[i], label="Derecha_R", color='orange')
            if i < len(sums_der_V_pasadas):  # Verificar que exista la pasada correspondiente en V
                sums_der_V_pasadas[i].plot(ax=axes_der[i], label="Derecha_V", color='r')
            axes_der[i].set_title(f'Suma de fuerzas Derecha (N) - Pasada N°{i+1}')
            axes_der[i].set_xlabel('Tiempo')
            axes_der[i].set_ylabel('Valores')
            axes_der[i].legend()
        
        # Gráfico de la izquierda
        if i < len(sums_izq_pasadas):
            sums_izq_pasadas[i].plot(ax=axes_izq[i], label="Izquierda_R", color='g')
            if i < len(sums_izq_V_pasadas):  # Verificar que exista la pasada correspondiente en V
                sums_izq_V_pasadas[i].plot(ax=axes_izq[i], label="Izquierda_V", color='b')
            axes_izq[i].set_title(f'Suma de fuerzas Izquierda (N) - Pasada N°{i+1}')
            axes_izq[i].set_xlabel('Tiempo')
            axes_izq[i].set_ylabel('Valores')
            axes_izq[i].legend()

    plt.tight_layout()
    plt.show()
   
    

# Configuración
sensor_numbers = range(1, 9)  # Sensores del 1 al 8
num_pasadas = 1

# Función para graficar un pie (derecho o izquierdo)
def plot_pie_sensors(pie_type, data_list, figsize=(20, 16)):
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(f'Comparación de sensores - Pie {pie_type}', fontsize=16)
    axs = axs.flatten()
    
    for sensor_idx, sensor_num in enumerate(sensor_numbers):
        for pasada_idx, df_pasada in enumerate(data_list):
            # Obtener los datos del sensor específico
            sensor_data = df_pasada.iloc[:, sensor_idx]  # Asumiendo que los sensores están en columnas ordenadas
            
            # Graficar cada pasada
            axs[sensor_idx].plot(sensor_data, 
                                label=f'Pasada {pasada_idx + 1}',
                                alpha=0.7,
                                linewidth=2)
            
        axs[sensor_idx].set_title(f'Sensor {sensor_num}')
        axs[sensor_idx].set_ylabel('Valor crudo')
        axs[sensor_idx].legend()
        axs[sensor_idx].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Generar gráficos para ambos pies
plot_pie_sensors("Derecho", raw_der_final[0:])
#plot_pie_sensors("Izquierdo", raw_izq_final)

'''
# Configuración
sensor_numbers = range(1, 9)  # Sensores del 1 al 8
num_pasadas = 1  # Últimas 3 pasadas

def plot_comparison_metodos(pie_type, filt_R_list, filt_V_list, num_last_pasadas=3):
    # Tomar las últimas pasadas
    last_pasadas = range(len(filt_R_list)-num_last_pasadas, len(filt_R_list))
    
    for pasada_idx in last_pasadas:
        if pasada_idx >= 0:
            fig, axs = plt.subplots(4, 2, figsize=(20, 16))
            fig.suptitle(f'Comparación métodos - Pie {pie_type} - Pasada {pasada_idx + 1}', fontsize=16)
            axs = axs.flatten()
            
            df_R = filt_R_list[pasada_idx]
            df_V = filt_V_list[pasada_idx]
            
            for sensor_idx, sensor_num in enumerate(sensor_numbers):
                sensor_R = df_R.iloc[:, sensor_idx]
                sensor_V = df_V.iloc[:, sensor_idx]
                
                # Graficar con etiquetas directas
                axs[sensor_idx].plot(sensor_R, label='Con R', color='blue', alpha=0.7, linewidth=2)
                axs[sensor_idx].plot(sensor_V, label='Con V', color='red', alpha=0.7, linewidth=2)
                
                axs[sensor_idx].set_title(f'Sensor {sensor_num}')
                axs[sensor_idx].set_ylabel('Valor en kg')
                axs[sensor_idx].legend()
                axs[sensor_idx].grid(True)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
plt.show()
# Llamada a la función con nombres explícitos
plot_comparison_metodos("Derecho", filt_der, filt_der_V, num_last_pasadas=3)
plot_comparison_metodos("Izquierdo", filt_izq, filt_izq_V, num_last_pasadas=3)
'''
'''
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
   

# Definir cantidad de pasadas
num_pasadas = max(len(BW_der_list), len(BW_izq_list))

# Evitar error si no hay datos
if num_pasadas == 0:
    print("No data to plot.")
else:
    # Crear figura y subplots (una fila por pasada, dos señales en cada subplot)
    fig, axes = plt.subplots(nrows=num_pasadas, ncols=1, figsize=(10, num_pasadas * 3), squeeze=False)
    axes = axes.flatten()  # Convertir en un array 1D para acceso fácil

    # Iterar sobre las pasadas
    for i in range(num_pasadas):
        if i < len(BW_der_list):
            BW_der_list[i].plot(ax=axes[i], label="Derecha", color='b')

        if i < len(BW_izq_list):
            BW_izq_list[i].plot(ax=axes[i], label="Izquierda", color='g')

        axes[i].set_title(f'Suma de fuerzas (N) - Pasada N°{i+1}')
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valores')
        #axes[i].set_xlim(14, 24)  # Ajusta si es necesario
        axes[i].legend()

    plt.tight_layout()



# Definir cantidad de pasadas
num_pasadas = max(len(sums_der_subset), len(sums_der_subset))

# Evitar error si no hay datos
if num_pasadas == 0:
    print("No data to plot.")
else:
    # Crear figura y subplots (una fila por pasada, dos señales en cada subplot)
    fig, axes = plt.subplots(nrows=num_pasadas, ncols=1, figsize=(10, num_pasadas * 3), squeeze=False)
    axes = axes.flatten()  # Convertir en un array 1D para acceso fácil

    # Iterar sobre las pasadas
    for i in range(num_pasadas):
        if i < len(sums_der_subset):
            sums_der_subset[i].plot(ax=axes[i], label="Derecha", color='b')

        if i < len(sums_izq_subset):
            sums_izq_subset[i].plot(ax=axes[i], label="Izquierda", color='g')

        axes[i].set_title(f'Suma de fuerzas (N) - Pasada N°{i+1}')
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valores')
        #axes[i].set_ylim(0, 70)  # Ajusta si es necesario
        axes[i].legend()

    plt.tight_layout()


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
'''
