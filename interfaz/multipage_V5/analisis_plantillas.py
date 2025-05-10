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
from scipy.interpolate import interp1d


def procesar_archivo_csv_plantillas(folder_path):
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
        
        mV_dataframes = []

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

        return mV_dataframes
    
    def subset(dfs, t_inicio, t_fin):
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
        if isinstance(dfs, list):
            # Si es una lista, aplica el filtrado a cada DataFrame en la lista
            sums_subset = [df[(df.index >= t_inicio) & (df.index <= t_fin)] for df in dfs]
        else:
            # Si es un solo DataFrame, aplica el filtrado directamente
            sums_subset = dfs[(dfs.index >= t_inicio) & (dfs.index <= t_fin)]
        
        return sums_subset
    
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
    
    # Aplicar las funciones a los DataFrames de las listas raw_der y raw_izq
    raw_der_proc = limpiar_valores_anomalos(raw_der)
    raw_izq_proc = limpiar_valores_anomalos(raw_izq)

    # Aplicar las funciones a los DataFrames de las listas raw_der y raw_izq
    raw_der_final = interp(raw_der_proc, frecuencia_hz=100)
    raw_izq_final = interp(raw_izq_proc, frecuencia_hz=100)

    mV_der = preproc_df(raw_der_final)  
    mV_izq = preproc_df(raw_izq_final)
    
    for i, (der, izq) in enumerate(zip(mV_der, mV_izq)):
        tf_der = der.index[-1]  # Último tiempo de sum_der
        tf_izq = izq.index[-1]  # Último tiempo de sum_izq
        
        dif = tf_der - tf_izq  # Diferencia entre los últimos tiempos
        der.index = der.index - dif  # Ajusta los índices de sum_der
        ti = izq.index[0]  # Primer tiempo de sum_izq
        
        # Aplica subset y actualiza las listas originales
        mV_der[i] = subset(der, ti, tf_izq)  # Filtra sum_der y actualiza sums_der
        mV_izq[i] = subset(izq, ti, tf_izq)  # Filtra sum_izq y actualiza sums_izq

    return mV_der, mV_izq, raw_der_final, raw_izq_final
