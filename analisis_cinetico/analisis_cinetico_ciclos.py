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

sums_der_subset = subset(sums_der, t_inicio, t_fin)
sums_izq_subset = subset(sums_izq, t_inicio, t_fin)

min_indices_subset_der, min_indices_subset_izq, min_tiempos_subset_der, min_tiempos_subset_izq = contacto_inicial(sums_der_subset, sums_izq_subset)

excluded_cycles_der, remaining_cycles_der, exclude_duration_der, exclude_shape_der = find_cycles_to_exclude(sums_der_subset, min_indices_subset_der, threshold_time=0.25)
excluded_cycles_izq, remaining_cycles_izq, exclude_duration_izq, exclude_shape_izq = find_cycles_to_exclude(sums_izq_subset, min_indices_subset_izq, threshold_time=0.25)


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
