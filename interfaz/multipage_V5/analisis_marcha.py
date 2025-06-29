import kineticstoolkit.lab as ktk
import matplotlib
import matplotlib.pyplot as plt
import scipy.spatial.transform as transform
import kineticstoolkit.external.icp as icp
from kineticstoolkit.typing_ import ArrayLike, check_param
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from scipy.signal import argrelextrema


def procesar_archivo_c3d( filename ):

    """## Funciones"""

    # Función para calcular el vector perpendicular (normal) al plano
    def calcular_vector_perpendicular(P1, P2, P3):
        def cross(v1, v2):
            """Cross on series of vectors of length 4."""
            c = v1.copy()
            c[:, 0:3] = np.cross(v1[:, 0:3], v2[:, 0:3])
            return c
        # Calculamos los vectores en el plano
        V1 = np.subtract(P2, P1)
        V2 = np.subtract(P3, P1)

        # Producto vectorial entre V1 y V2 para obtener el vector normal (perpendicular)
        return  cross(V1, V2)

    # Define una función para aplicar el filtro de paso bajo
    def low_pass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs  # Frecuencia de Nyquist
        normal_cutoff = cutoff / nyquist  # Frecuencia de corte normalizada
        b, a = butter(order, normal_cutoff, btype="low", analog=False)  # Diseña el filtro
        # Aplica el filtro
        return filtfilt(b, a, data)

    def find_cycles_to_exclude(heel_y, time, min_indices, threshold_time, threshold_shape,threshold_shape1):
        """
        Encuentra los ciclos que deben ser excluidos basados en la duración y la forma de la curva.

        Parámetros:
        - heel_y: Array con los datos de la coordenada Y del talón.
        - time: Array con los tiempos correspondientes a los datos.
        - min_indices: Índices de los mínimos locales (eventos de contacto inicial).
        - threshold_time: Umbral de diferencia en la duración del ciclo (porcentaje).
        - threshold_shape: Umbral de diferencia en la forma de la curva (distancia euclidiana).

        Retorna:
        - excluded_cycles: Lista de índices de ciclos a excluir (numerados desde 1).
        """
        # Calcular la duración de cada ciclo
        cycle_durations = [time[min_indices[i + 1]] - time[min_indices[i]] for i in range(len(min_indices) - 1)]
        avg_duration = np.mean(cycle_durations)

        # Calcular la diferencia en la duración de cada ciclo respecto al promedio
        duration_diffs = np.abs((cycle_durations - avg_duration) / avg_duration)

        # Encontrar ciclos con duración muy diferente al promedio
        exclude_duration = np.where(duration_diffs > threshold_time)[0]


        # Obtener todos los ciclos
        all_cycles = [heel_y[min_indices[i]:min_indices[i + 1]] for i in range(len(min_indices) - 1)]

        # Filtrar los ciclos que no fueron excluidos por tiempo
        remaining_cycles = [i for i in range(len(cycle_durations)) if i not in exclude_duration]

        # Función para excluir por forma
        def exclude_by_shape(cycles_indices, all_cycles, threshold_shape):
            # Calcular la forma promedio de los ciclos restantes
            avg_cycle = np.mean([np.interp(np.linspace(0, 100, num=100), np.linspace(0, 100, num=len(all_cycles[i])), all_cycles[i]) for i in cycles_indices], axis=0)

            # Calcular la diferencia en la forma de cada ciclo respecto al promedio
            shape_diffs = []
            for i in cycles_indices:
                cycle = all_cycles[i]
                normalized_cycle = np.interp(np.linspace(0, 100, num=100), np.linspace(0, 100, num=len(cycle)), cycle)
                shape_diff = np.linalg.norm(normalized_cycle - avg_cycle)
                shape_diffs.append(shape_diff)


            # Encontrar ciclos con forma muy diferente al promedio
            exclude_shape = [cycles_indices[i] for i in np.where(np.array(shape_diffs) > threshold_shape)[0]]
            return exclude_shape, avg_cycle

        # Primera exclusión por forma
        exclude_shape_1, avg_cycle_1 = exclude_by_shape(remaining_cycles, all_cycles, threshold_shape)

        # Filtrar los ciclos que no fueron excluidos en la primera exclusión por forma
        remaining_cycles_after_shape_1 = [i for i in remaining_cycles if i not in exclude_shape_1]

        # Segunda exclusión por forma
        exclude_shape_2, avg_cycle_2 = exclude_by_shape(remaining_cycles_after_shape_1, all_cycles, threshold_shape1)
        not_excluded_cycles = [i for i in remaining_cycles_after_shape_1 if i not in exclude_shape_2]

        # Combinar todas las exclusiones y ajustar la numeración (sumar 1)
        excluded_cycles = list(set(exclude_duration).union(set(exclude_shape_1)).union(set(exclude_shape_2)))
        excluded_cycles = [i + 1 for i in excluded_cycles]  # Ajustar la numeración
        not_excluded_cycles = [i + 1 for i in not_excluded_cycles]  # Ajustar la numeración


        return excluded_cycles, not_excluded_cycles

    # Función para excluir ciclos
    def exclude_cycles(min_indices, min_times, excluded_cycles):
        """
        Excluye los ciclos especificados en excluded_cycles de min_indices y min_times.
        Devuelve un vector de pares [inicio, fin] que representan los ciclos no excluidos.
        """
        # Lista para almacenar los ciclos no excluidos
        filtered_cycles = []
        filtered_cycles_times = []

        # Iterar sobre los ciclos
        for i in range(len(min_indices) - 1):  # -1 porque cada ciclo necesita un par de índices
            cycle_number = i + 1  # Los ciclos comienzan desde 1
            if cycle_number not in excluded_cycles:
                # Agregar el par [inicio, fin] que define el ciclo
                inicio = min_indices[i]
                fin = min_indices[i + 1]

                inicio_t = min_times[i]
                fin_t = min_times[i + 1]
                filtered_cycles.append([inicio, fin])
                filtered_cycles_times.append([inicio_t, fin_t])

        # Convertir a un array de numpy
        filtered_cycles = np.array(filtered_cycles)
        filtered_cycles_times = np.array(filtered_cycles_times)


        return filtered_cycles, filtered_cycles_times
    
    # Función para calcular pasos dentro de un segmento
    def calculate_steps(segment, filtered_indices_right, filtered_indices_left, time, heel_x_right, heel_x_left):
        """
        Calcula los pasos dentro de un segmento.
        Devuelve una lista de tuplas (tiempo_inicio, tiempo_fin, tiempo_paso, longitud_paso).
        """
        steps = []

        # Crear un único vector ordenado con los índices de inicio y fin de los ciclos no excluidos
        all_cycles = []

        # Agregar ciclos derechos
        for cycle in filtered_indices_right:
            inicio, fin = cycle
            if inicio in segment and fin in segment:  # Verificar que el ciclo esté dentro del segmento
                all_cycles.append((time[inicio], "right"))
                all_cycles.append((time[fin], "right"))

        # Agregar ciclos izquierdos
        for cycle in filtered_indices_left:
            inicio, fin = cycle
            if inicio in segment and fin in segment:  # Verificar que el ciclo esté dentro del segmento
                all_cycles.append((time[inicio], "left"))
                all_cycles.append((time[fin], "left"))

        # Ordenar los ciclos por tiempo de inicio y eliminar duplicados
        all_cycles.sort(key=lambda x: x[0])  # Ordenar por tiempo
        unique_cycles = []
        seen_times = set()
        for cycle in all_cycles:
            if cycle[0] not in seen_times:  # Eliminar duplicados
                unique_cycles.append(cycle)
                seen_times.add(cycle[0])


        # Recorrer el vector ordenado para identificar los pasos
        for i in range(len(unique_cycles) - 1):
            inicio_tiempo, inicio_pie = unique_cycles[i]
            siguiente_tiempo, siguiente_pie = unique_cycles[i + 1]

            # Verificar si el paso es válido (alterna entre pie derecho e izquierdo)
            if inicio_pie != siguiente_pie:
                # Calcular el tiempo de paso
                tiempo_paso = siguiente_tiempo - inicio_tiempo

                # Calcular la longitud de paso (diferencia en la coordenada x)
                if inicio_pie == "right":
                    # Extraer el índice como escalar usando [0]
                    indice_inicio = np.where(time == inicio_tiempo)[0][0]
                    indice_fin = np.where(time == siguiente_tiempo)[0][0]
                    longitud_paso = abs(heel_x_right[indice_fin] - heel_x_left[indice_inicio])
                else:
                    # Extraer el índice como escalar usando [0]
                    indice_inicio = np.where(time == inicio_tiempo)[0][0]
                    indice_fin = np.where(time == siguiente_tiempo)[0][0]
                    longitud_paso = abs(heel_x_left[indice_fin] - heel_x_right[indice_inicio])

                # Agregar el paso a la lista
                steps.append((inicio_tiempo, siguiente_tiempo, tiempo_paso, longitud_paso))

        return steps
    
    def eliminar_ultimo_paso(filtered_indices_right, filtered_indices_left,
                        filtered_times_right, filtered_times_left,
                        final_cycles_right, final_cycles_left,
                        excluded_cycles_right, excluded_cycles_left,
                        time, segmentos):
        # Inicializar listas para resultados
        indices_in_segment_right = []
        indices_in_segment_left = []
        times_in_segment_right = []
        times_in_segment_left = []
        cycle_in_segment_right = []
        cycle_in_segment_left = []

        # Procesar cada segmento
        for segmento in segmentos:
            seg_start = time[segmento[0]]
            seg_end = time[segmento[-1]]

            # Pasos derechos en este segmento
            right_in_seg = []
            for k in range(len(filtered_times_right)):
                if filtered_times_right[k][0] >= seg_start and filtered_times_right[k][1] <= seg_end:
                    right_in_seg.append({
                        'indices': filtered_indices_right[k],
                        'times': filtered_times_right[k],
                        'cycle': final_cycles_right[k],
                        'type': 'right'
                    })

            # Pasos izquierdos en este segmento
            left_in_seg = []
            for k in range(len(filtered_times_left)):
                if filtered_times_left[k][0] >= seg_start and filtered_times_left[k][1] <= seg_end:
                    left_in_seg.append({
                        'indices': filtered_indices_left[k],
                        'times': filtered_times_left[k],
                        'cycle': final_cycles_left[k],
                        'type': 'left'
                    })

            # Combinar y ordenar todos los pasos por tiempo de inicio
            todos_los_pasos = right_in_seg + left_in_seg
            todos_los_pasos.sort(key=lambda x: x['times'][0])

            # Conservar siempre el primer paso (nunca eliminarlo)
            # Eliminar solo el último paso si hay 2 o más pasos
            for i, paso in enumerate(todos_los_pasos):
                if i == len(todos_los_pasos) - 1 and len(todos_los_pasos) >= 2:
                    # Excluir el último paso si hay al menos 2 pasos
                    if paso['type'] == 'right':
                        excluded_cycles_right.append(paso['cycle'])
                    else:
                        excluded_cycles_left.append(paso['cycle'])
                else:
                    # Conservar todos los demás pasos (incluido el primero)
                    if paso['type'] == 'right':
                        indices_in_segment_right.append(paso['indices'])
                        times_in_segment_right.append(paso['times'])
                        cycle_in_segment_right.append(paso['cycle'])
                    else:
                        indices_in_segment_left.append(paso['indices'])
                        times_in_segment_left.append(paso['times'])
                        cycle_in_segment_left.append(paso['cycle'])

        # Eliminar duplicados y ordenar ciclos excluidos
        excluded_cycles_right = sorted(list(set(excluded_cycles_right)))
        excluded_cycles_left = sorted(list(set(excluded_cycles_left)))

        return (indices_in_segment_right, indices_in_segment_left,
                times_in_segment_right, times_in_segment_left,
                cycle_in_segment_right, cycle_in_segment_left,
                excluded_cycles_right, excluded_cycles_left)

    def procesar_articulacion(articulacion, nombre, ylim, min_indices):
        euler_angles = ktk.geometry.get_angles(articulacion, "ZYX", degrees=True)
        angles = ktk.TimeSeries(time=markers.time)
        angles.data["Z"] = euler_angles[:, 0]
        angles.data["Y"] = euler_angles[:, 1]
        angles.data["X"] = euler_angles[:, 2]

        # Define parámetros del filtro
        sampling_rate = 100  # Frecuencia de muestreo en Hz (ajusta según tus datos)
        cutoff_frequency = 6  # Frecuencia de corte en Hz
        order = 4  # Orden del filtro

        # Aplicar filtro
        angles.data["Z"] = low_pass_filter(angles.data["Z"], cutoff_frequency, sampling_rate, order)
        angles.data["Y"] = low_pass_filter(angles.data["Y"], cutoff_frequency, sampling_rate, order)
        angles.data["X"] = low_pass_filter(angles.data["X"], cutoff_frequency, sampling_rate, order)

        # Agrega información adicional a los datos
        angles = angles.add_data_info("Dorsiflexion", "Unit", "deg")
        angles = angles.add_data_info("Int/ Ext Rotation", "Unit", "deg")
        angles = angles.add_data_info("Eversion", "Unit", "deg")

        # Normalización y promediado
        normalized_Z, normalized_Y, normalized_X = [], [], []



        gait_cycles_angles = []
        curves_dict = {"Z": [], "Y": [], "X": []}

        for i in min_indices:
            start_idx = i[0]
            end_idx = i[1]


            cycle_angles = {
                    "time": angles.time[start_idx:end_idx],
                    "Z": angles.data["Z"][start_idx:end_idx],
                    "Y": angles.data["Y"][start_idx:end_idx],
                    "X": angles.data["X"][start_idx:end_idx]
            }

            cycle_length = len(cycle_angles["time"])
            normalized_time = np.linspace(0, 100, num=cycle_length)
            cycle_angles["Normalized_time"] = normalized_time

            num_points = 100
            fixed_time = np.linspace(0, 100, num=num_points)
            Z_interpolated = np.interp(fixed_time, normalized_time, cycle_angles["Z"])
            Y_interpolated = np.interp(fixed_time, normalized_time, cycle_angles["Y"])
            X_interpolated = np.interp(fixed_time, normalized_time, cycle_angles["X"])


            normalized_Z.append(Z_interpolated)
            normalized_Y.append(Y_interpolated)
            normalized_X.append(X_interpolated)
            curves_dict["Z"].append(Z_interpolated)
            curves_dict["Y"].append(Y_interpolated)
            curves_dict["X"].append(X_interpolated)

            cycle_angles["Fixed_time"] = fixed_time
            cycle_angles["Z_interpolated"] = Z_interpolated
            cycle_angles["Y_interpolated"] = Y_interpolated
            cycle_angles["X_interpolated"] = X_interpolated
            gait_cycles_angles.append(cycle_angles)

        if len(normalized_Z) > 0:
            if len(normalized_Z) == 1:  # Si solo hay un ciclo no excluido
                average_Z = normalized_Z[0]
                average_Y = normalized_Y[0]
                average_X = normalized_X[0]
            else:  # Si hay múltiples ciclos
                average_Z = np.mean(normalized_Z, axis=0)
                average_Y = np.mean(normalized_Y, axis=0)
                average_X = np.mean(normalized_X, axis=0)


        # Devolver las curvas no excluidas y los promedios
        return curves_dict, average_Z, average_Y, average_X
    
    def calcular_angulo_pelvis_pie_derecho(markers, ASIS_der, ASIS_izq, RHeel, RToeIn, IC_right, filtered_indices_right):
        """
        Calcula pelvic obliquity, pelvic rotation y foot progression angle en base a ciclos válidos,
        ajustando la dirección de marcha por ciclo.

        Retorna:
        - angulos_normalizados: diccionario con curvas por ciclo normalizadas
        - señales_completas: diccionario con las señales completas (no recortadas)
        """
        n_frames = markers.time.shape[0]

        # Inicializar arrays con NaN
        obliquity_deg = np.full(n_frames, np.nan)
        pelvic_rotation_deg = np.full(n_frames, np.nan)
        foot_progression_angle_deg = np.full(n_frames, np.nan)

        # Procesar cada ciclo válido
        for start, end in filtered_indices_right:
            # Dirección de marcha en plano XZ (usando IC o RHeel como referencia)
            pelvis_dir = ASIS_der[start:end, [0,1, 2 ]] - ASIS_izq[start:end, [0,1, 2]]

            dir_z = pelvis_dir[2][0]

            # ---- Pelvic Obliquity (plano YZ) ----
            pelvis_yz = ASIS_der[start:end, [1, 2]] - ASIS_izq[start:end, [1, 2]]
            if dir_z > 0:
                obliquity_rad = np.arctan2(pelvis_yz[:, 0], pelvis_yz[:, 1]) # Y contra Z
            else:
                obliquity_rad = np.arctan2(pelvis_yz[:, 0], -pelvis_yz[:, 1]) # Y contra Z
                obliquity_deg[start:end] = np.rad2deg(obliquity_rad)

            # ---- Pelvic Rotation (ángulo pelvis vs dirección de marcha en plano XZ) ----
            pelvis_xz = ASIS_der[start:end, [0, 2]] - ASIS_izq[start:end, [0, 2]]
            if dir_z > 0:
                rotation_rad = np.arctan2(pelvis_xz[:, 0], pelvis_xz[:, 1] ) # X contra Z
            else:
                rotation_rad = np.arctan2(-pelvis_xz[:, 0], -pelvis_xz[:, 1] ) # X contra Z
                pelvic_rotation_deg[start:end] = np.rad2deg(rotation_rad)

            # ---- Foot Progression (ángulo pie vs dirección de marcha en plano XZ) ----

            foot_vector = RToeIn[start:end, [0, 2]] - RHeel[start:end, [0, 2]]
            if dir_z > 0:
                ang_rad = np.arctan2(foot_vector[:, 1], foot_vector[:, 0] ) # X contra Z
            else:
                ang_rad = np.arctan2(-foot_vector[:, 1],-foot_vector[:, 0] ) # X contra Z

            foot_progression_angle_deg[start:end] = np.rad2deg(ang_rad)

        # --- Normalización por ciclo ---
        def normalizar_por_ciclo(serie, ciclos):
            normalizadas = []
            for c in ciclos:
                start, end = c
                ciclo = serie[start:end]
                if np.isnan(ciclo).all():
                    continue  # Salta ciclos completamente vacíos
                t_norm = np.linspace(0, 100, num=len(ciclo))
                interp = np.interp(np.linspace(0, 100, 100), t_norm, ciclo)
                normalizadas.append(interp)
            return np.array(normalizadas)


        angulos_normalizados = {
            "Pelvic_Obliquity": normalizar_por_ciclo(obliquity_deg, filtered_indices_right),
            "Pelvic_Rotation": normalizar_por_ciclo(pelvic_rotation_deg, filtered_indices_right),
            "Foot_Progression": normalizar_por_ciclo(foot_progression_angle_deg, filtered_indices_right)
        }

        señales_completas = {
            "Pelvic_Obliquity": obliquity_deg,
            "Pelvic_Rotation": pelvic_rotation_deg,
            "Foot_Progression": foot_progression_angle_deg
        }

        return angulos_normalizados, señales_completas

    def calcular_angulo_pelvis_pie_izquierdo(markers, ASIS_der, ASIS_izq, LHeel, LToeIn, IC_left, filtered_indices_left):
        """
        Calcula pelvic obliquity, pelvic rotation y foot progression angle para el pie izquierdo,
        ajustando la dirección de marcha por ciclo.

        Retorna:
        - angulos_normalizados: diccionario con curvas por ciclo normalizadas
        - señales_completas: diccionario con las señales completas (no recortadas)
        """
        n_frames = markers.time.shape[0]

        # Inicializar arrays con NaN
        obliquity_deg = np.full(n_frames, np.nan)
        pelvic_rotation_deg = np.full(n_frames, np.nan)
        foot_progression_angle_deg = np.full(n_frames, np.nan)

        # Procesar cada ciclo válido
        for start, end in filtered_indices_left:
            # Dirección de marcha en plano XZ (usando IC o LHeel como referencia)
            pelvis_dir = ASIS_der[start:end, [0, 1, 2]] - ASIS_izq[start:end, [0, 1, 2]]
            dir_z = pelvis_dir[2][0]

            # ---- Pelvic Obliquity (plano YZ) ----
            pelvis_yz = ASIS_der[start:end, [1, 2]] - ASIS_izq[start:end, [1, 2]]
            if dir_z < 0:
                
                obliquity_rad = np.arctan2(pelvis_yz[:, 0], pelvis_yz[:, 1])  # Y contra Z
            else:
                obliquity_rad = np.arctan2(pelvis_yz[:, 0], -pelvis_yz[:, 1])  # Y contra Z
            obliquity_deg[start:end] = -np.rad2deg((obliquity_rad))

            

            # ---- Pelvic Rotation (ángulo pelvis vs dirección de marcha en plano XZ) ----
            pelvis_xz = ASIS_der[start:end, [0, 2]] - ASIS_izq[start:end, [0, 2]]
            if dir_z < 0:
                rotation_rad = np.arctan2(pelvis_xz[:, 0], pelvis_xz[:, 1])  # X contra Z
            else:
                rotation_rad = np.arctan2(-pelvis_xz[:, 0], -pelvis_xz[:, 1])  # X contra Z
            pelvic_rotation_deg[start:end] = -np.rad2deg((rotation_rad))

            # ---- Foot Progression (ángulo pie vs dirección de marcha en plano XZ) ----
            foot_vector = LToeIn[start:end, [0, 2]] - LHeel[start:end, [0, 2]]
            
            if dir_z < 0:
                ang_rad = np.arctan2(foot_vector[:, 1], foot_vector[:, 0])  # X contra Z
            else:
                ang_rad =  np.arctan2(-foot_vector[:, 1], -foot_vector[:, 0])  # X contra Z
            foot_progression_angle_deg[start:end] =  np.rad2deg((ang_rad))
            
    


        # --- Normalización por ciclo ---
        def normalizar_por_ciclo(serie, ciclos):
            normalizadas = []
            for c in ciclos:
                start, end = c
                ciclo = serie[start:end]
                if np.isnan(ciclo).all():
                    continue  # Salta ciclos completamente vacíos
                t_norm = np.linspace(0, 100, num=len(ciclo))
                interp = np.interp(np.linspace(0, 100, 100), t_norm, ciclo)
                normalizadas.append(interp)
            return np.array(normalizadas)

        

        angulos_normalizados = {
            "Pelvic_Obliquity": normalizar_por_ciclo(obliquity_deg, filtered_indices_left),
            "Pelvic_Rotation": normalizar_por_ciclo(pelvic_rotation_deg, filtered_indices_left),
            "Foot_Progression": normalizar_por_ciclo(foot_progression_angle_deg, filtered_indices_left)
        }

        señales_completas = {
            "Pelvic_Obliquity": obliquity_deg,
            "Pelvic_Rotation": pelvic_rotation_deg,
            "Foot_Progression": foot_progression_angle_deg
        }

        return angulos_normalizados, señales_completas




    """# Cargamos los datos"""


    markers = ktk.read_c3d(filename)["Points"]
    time = markers.time 

    #LAS DEFINICIONES LAS HACEMOS SIGUENDO EL PAPER
    #   Y  ^
    #      |
    #      |
    #   Z  o-------> X
    #
    #COMO VIENEN EN EL DATA SET
    #           Z  ^
    #              |
    #              |
    #   X  <------ o Y

    #ROTACION DE EJES
    import copy

    # Crear una copia de markers para trabajar con la estructura completa
    markers_copy = copy.deepcopy(markers)

    # Lista de marcadores que deseas procesar
    keys = markers.data.keys()
    # Convert dict_keys to a list
    marcadores = list(keys)

    # Creación de las transformaciones T1 y T2
    T1 = ktk.geometry.create_transforms(
        seq="x",  # Rotación alrededor del eje x
        angles=[90],
        translations=[[0, 0, 0]],
        degrees=True,
    )

    T2 = ktk.geometry.create_transforms(
        seq="z",  # Rotación alrededor del eje z
        angles=[180],
        translations=[[0, 0, 0]],
        degrees=True,
    )

    # Aplicar transformaciones a todos los marcadores en la copia
    for marcador in marcadores:
        # Extraer los datos del marcador actual en la copia
        rotation_1 = markers_copy.data[marcador]
        # Aplicar T1
        rotation_1 = ktk.geometry.matmul(T1, rotation_1)
        # Aplicar T2
        rotation_2 = ktk.geometry.matmul(T2, rotation_1)
        # Actualizar el marcador en la copia con las nuevas rotaciones
        markers_copy.data[marcador] = rotation_2

    markers.data = markers_copy.data

    """## Tibia"""

    #-------------------------------------------------------------------------------------------------
    # DEFINIMOS COORDENADAS LOCALES CON REFERENCIAS ANATOMICAS PARA EL TOBILLO


    # Lado derecho
    IC_right = 0.5 * (markers.data["Rashel:RKneeOut"] + markers.data["Rashel:RKneeIn"])
    MM_right = markers.data["Rashel:RAnkleIn"]
    LM_right = markers.data["Rashel:RAnkleOut"]
    IM_right = 0.5 * (markers.data["Rashel:RAnkleIn"] + markers.data["Rashel:RAnkleOut"])
    LC_right = markers.data["Rashel:RKneeOut"]
    MC_right = markers.data["Rashel:RKneeIn"]

    # Lado izquierdo
    IC_left = 0.5 * (markers.data["Rashel:LKneeOut"] + markers.data["Rashel:LKneeIn"])
    MM_left = markers.data["Rashel:LAnkleIn"]
    LM_left = markers.data["Rashel:LAnkleOut"]
    IM_left = 0.5 * (markers.data["Rashel:LAnkleIn"] + markers.data["Rashel:LAnkleOut"])
    LC_left = markers.data["Rashel:LKneeOut"]
    MC_left = markers.data["Rashel:LKneeIn"]

    # Calcular el vector normal y el vector con origen
    # Calcular vectores perpendiculares para ambos lados
    perp_torsional_right = calcular_vector_perpendicular(MM_right, IC_right, LM_right)
    perp_frontal_right = calcular_vector_perpendicular(LC_right, IM_right, MC_right)

    perp_torsional_left = calcular_vector_perpendicular(LM_left, IC_left, MM_left)
    perp_frontal_left = calcular_vector_perpendicular(MC_left , IM_left, LC_left)


    #TIBIA Lado derecho
    origen_tibia_right = IM_right
    Z_tibia_right = LM_right - MM_right
    Y_tibia_right = IC_right - IM_right
    X_tibia_right = perp_torsional_right

    frames = ktk.TimeSeries(time=markers.time)
    frames.data["Tibia_Right"] = ktk.geometry.create_frames(origin=origen_tibia_right, z=Z_tibia_right, xz=X_tibia_right)


    origen_tibia_right_knee = IC_right
    frames.data["TibiaRodilla_Right"] = ktk.geometry.create_frames(origin=origen_tibia_right_knee, z=Z_tibia_right, xz=X_tibia_right)


    #TIBIA Lado izquierdo
    origen_tibia_left = IM_left
    Z_tibia_left =  MM_left - LM_left
    Y_tibia_left = IC_left - IM_left
    X_tibia_left = perp_torsional_left

    frames.data["Tibia_Left"] = ktk.geometry.create_frames(origin=origen_tibia_left, z=Z_tibia_left, xz= -X_tibia_left)

    origen_tibia_left_knee = IC_left

    frames.data["TibiaRodilla_Left"] = ktk.geometry.create_frames(origin=origen_tibia_left_knee, z=Z_tibia_left, xz= -X_tibia_left)

    """## Tobillo"""

    #TOBILLO Lado derecho
    origen_tobillo_right = IM_right
    Y_tobillo_right = IC_right - IM_right
    X_tobillo_right = perp_frontal_right
    X_tobillo1_right = markers.data["Rashel:RToeIn"] - markers.data["Rashel:RHeel"]


    frames.data["Calcaneus_Right"] = ktk.geometry.create_frames(origin=origen_tobillo_right, x=X_tobillo1_right, xy=Y_tobillo_right)


    #TOBILLO Lado izquierdo
    origen_tobillo_left = IM_left
    Y_tobillo_left =   IC_left - IM_left
    X_tobillo_left = perp_frontal_left
    X_tobillo1_left =  markers.data["Rashel:LToeIn"] - markers.data["Rashel:LHeel"]

    frames.data["Calcaneus_Left"] = ktk.geometry.create_frames(origin=origen_tobillo_left, x=-X_tobillo1_left, xy=Y_tobillo_left)

    """## Femur"""

    ASIS_der = markers.data['Rashel:WaistRFront']
    ASIS_izq = markers.data['Rashel:WaistLFront']
    mid_ASIS = 0.5*(ASIS_der + ASIS_izq)
    mid_PSIS = markers.data['Rashel:WaistBack']
    mid_FE_right = 0.5 * (markers.data["Rashel:RKneeOut"] + markers.data["Rashel:RKneeIn"])

    mid_FE_left = 0.5 * (markers.data["Rashel:LKneeOut"] + markers.data["Rashel:LKneeIn"])



    Rleg_length = np.linalg.norm(markers.data['Rashel:RAnkleOut'][0,0:3] - ASIS_der[0,0:3])
    Lleg_length = np.linalg.norm(markers.data['Rashel:LAnkleOut'][0,0:3] - ASIS_izq[0,0:3])
    ASIS_distance = np.linalg.norm(ASIS_der[0,0:3] - ASIS_izq[0,0:3])
    X_dis = 0.1288*Rleg_length - 0.04856
    C = 0.115*Rleg_length - 0.0153
    theta = 28.4*np.pi/180.0
    beta = 18.0*np.pi/180.0
    Rmarker = 0
    S = 1



    hip_Y=	(-X_dis -Rmarker)*np.sin(beta) - C*np.cos(theta)*np.cos(beta)
    hip_X = (-X_dis-Rmarker)*np.cos(beta)+ C*np.cos(theta)*np.sin(beta)
    hip_Z= S*(C*np.sin(theta)-0.5*ASIS_distance)
    hip_XYZ = np.array([hip_X, hip_Y, hip_Z, 0])




    # Supongamos que h es un vector con componentes x, y, z
    R = ASIS_der - mid_ASIS   #me fijo por jemplo para que direccion en z, va este vector, de ahi parto

    z_value_r = R[:, 2]  # Selecciona todos los valores de la tercera columna
    x_value_r = R[:,0]
    # Aplicar la condición a todo el array
    R_z = np.where(z_value_r < 0, 1, np.where(z_value_r > 0, -1, 0))
    R_x = np.where(z_value_r < 0, -1, np.where(z_value_r > 0, 1, 0))
    R[:] = 1  #hago una matriz de unos para poder multiplicar por  hip_XYZ
    R[:, 0] = R_x
    R[:, 2] = R_z

    origen_hip_right  = mid_ASIS +  R * hip_XYZ

    L = ASIS_der - mid_ASIS

    z_value = L[:, 2]  # Selecciona todos los valores de la tercera columna
    x_value = L[:,0]
    # Aplicar la condición a todo el array
    L_z = np.where(z_value < 0, -1, np.where(z_value > 0, 1, 0))
    L_x = np.where(z_value < 0, -1, np.where(z_value > 0, 1, 0))
    L[:] = 1  #hago una matriz de unos para poder multiplicar por  hip_XYZ
    L[:, 0] = L_x
    L[:, 2] = L_z

    hip_XYZ_left = np.array([hip_X, hip_Y, hip_Z, 0])
    origen_hip_left = mid_ASIS +  L* hip_XYZ_left



    #FEMUR Lado Derecho

    origen_femur_right = IC_right
    Y_femur_right = origen_hip_right - mid_FE_right
    YZ_femur_right = markers.data["Rashel:RKneeOut"] - markers.data["Rashel:RKneeIn"]

    frames.data["FemurRodilla_Right"] = ktk.geometry.create_frames(origin=origen_femur_right, y=Y_femur_right, yz=YZ_femur_right)




    origen_femur_right = origen_hip_right


    frames.data["Femur_Right"] = ktk.geometry.create_frames(origin=origen_femur_right, y=Y_femur_right, yz=YZ_femur_right)


    #FEMUR Lado Izquiero

    origen_femur_left = IC_left
    Y_femur_left = origen_hip_left - mid_FE_left
    YZ_femur_left =  markers.data["Rashel:LKneeIn"] - markers.data["Rashel:LKneeOut"]

    frames.data["FemurRodilla_Left"] = ktk.geometry.create_frames(origin=origen_femur_left, y=-Y_femur_left, yz=YZ_femur_left)

    origen_femur_left = origen_hip_left


    frames.data["Femur_Left"] = ktk.geometry.create_frames(origin=origen_femur_left, y=-Y_femur_left, yz=YZ_femur_left)

    """## Cadera"""

    #CADERA  Lado derecho
    Z_hip_right = ASIS_der - ASIS_izq
    XZ_hip_right = mid_ASIS - mid_PSIS

    frames.data["Hip_Right"] = ktk.geometry.create_frames(origin=origen_hip_right, z=Z_hip_right, xz=XZ_hip_right)

    #CADERA  Lado izquierdo
    Z_hip_left = ASIS_der- ASIS_izq
    XZ_hip_left = mid_ASIS- mid_PSIS

    frames.data["Hip_Left"] = ktk.geometry.create_frames(origin=origen_hip_left, z=Z_hip_left, xz=-XZ_hip_left)

    """## Calculamos los angulos"""

    Tibia_to_calcaneus_Right = ktk.geometry.get_local_coordinates(frames.data["Calcaneus_Right"], frames.data["Tibia_Right"])
    Femur_to_tibia_Right= ktk.geometry.get_local_coordinates( frames.data["FemurRodilla_Right"], frames.data["TibiaRodilla_Right"])
    Hip_to_femur_Right = ktk.geometry.get_local_coordinates(frames.data["Femur_Right"], frames.data["Hip_Right"])

    Tibia_to_calcaneus_Left = ktk.geometry.get_local_coordinates(frames.data["Calcaneus_Left"], frames.data["Tibia_Left"])
    Femur_to_tibia_Left= ktk.geometry.get_local_coordinates( frames.data["FemurRodilla_Left"], frames.data["TibiaRodilla_Left"])
    Hip_to_femur_Left = ktk.geometry.get_local_coordinates(frames.data["Femur_Left"], frames.data["Hip_Left"])

    # --- Procesamiento de ciclos de marcha ---
    heel_r = markers.data["Rashel:RHeel"][:, 1]
    heel_l = markers.data["Rashel:LHeel"][:, 1]
    heel_r = low_pass_filter(heel_r, cutoff=15, fs=100, order=3)
    heel_l = low_pass_filter(heel_l, cutoff=15, fs=100, order=3)
    time = markers.time  # Tiempo correspondiente a los datos

    # Detección de ciclos
    min_indices_right = argrelextrema(heel_r, np.less, order=20)[0]
    min_indices_left = argrelextrema(heel_l, np.less, order=20)[0]

    # Exclusión de ciclos anómalos
    excluded_cycles_right, final_cycles_right = find_cycles_to_exclude(heel_r, markers.time, min_indices_right, 0.5, 0.4, 0.4)
    excluded_cycles_left, final_cycles_left = find_cycles_to_exclude(heel_l, markers.time, min_indices_left, 0.5, 0.4, 0.4)

    # Filtrar ciclos
    filtered_indices_right, filtered_times_right = exclude_cycles(min_indices_right, markers.time[min_indices_right], excluded_cycles_right)
    filtered_indices_left, filtered_times_left = exclude_cycles(min_indices_left, markers.time[min_indices_left], excluded_cycles_left)
    

    # Segmentación de la marcha
    dir = ASIS_der - mid_ASIS
    z_value_dir = dir[:, 2]
    cambios_de_signo = np.where(np.diff(np.sign(z_value_dir)) != 0)[0] + 1
    segmentos = np.split(np.arange(len(markers.time)), cambios_de_signo)

    # Eliminar último paso de cada segmento
    filtered_indices_right, filtered_indices_left, filtered_times_right, filtered_times_left, \
    final_cycles_right, final_cycles_left, excluded_cycles_right, excluded_cycles_left = eliminar_ultimo_paso(
        filtered_indices_right, filtered_indices_left, filtered_times_right, filtered_times_left,
        final_cycles_right, final_cycles_left, excluded_cycles_right, excluded_cycles_left,
        markers.time, segmentos
    )

    # Procesar todas las articulaciones
    curvas_tobillo_derecho, _, _, _ = procesar_articulacion(Tibia_to_calcaneus_Right, "Tobillo Derecho", (-30, 30), filtered_indices_right)
    curvas_tobillo_izquierdo, _, _, _ = procesar_articulacion(Tibia_to_calcaneus_Left, "Tobillo Izquierdo", (-30, 30), filtered_indices_left)
    
    # Procesar todas las articulaciones
    curvas_rodilla_derecha, _, _, _ = procesar_articulacion(Femur_to_tibia_Right, "Rodilla Derecha", (-20, 70), filtered_indices_right)
    curvas_rodilla_izquierda, _, _, _ = procesar_articulacion(Femur_to_tibia_Left, "Rodilla Izquierda", (-20, 70), filtered_indices_left)

    # Procesar todas las articulaciones
    curvas_cadera_derecha, _, _, _ = procesar_articulacion(Hip_to_femur_Right, "Cadera Derecha", (-20, 60), filtered_indices_right)
    curvas_cadera_izquierda, _, _, _ = procesar_articulacion(Hip_to_femur_Left, "Cadera Izquierda", (-20, 60), filtered_indices_left)

    RHeel = markers.data["Rashel:RHeel"]
    RToeIn = markers.data["Rashel:RToeIn"]
    LHeel = markers.data["Rashel:LHeel"]
    LToeIn = markers.data["Rashel:LToeIn"]

    angulos_norm_der, _ = calcular_angulo_pelvis_pie_derecho(
            markers, ASIS_der, ASIS_izq, RHeel, RToeIn, IC_right, filtered_indices_right
        )
    angulos_norm_izq, _ = calcular_angulo_pelvis_pie_izquierdo(
            markers, ASIS_der, ASIS_izq, LHeel, LToeIn, IC_left, filtered_indices_left
        )
    

    def filtrar_ciclos_validos(angulo_dict):
        obli = angulo_dict["Pelvic_Obliquity"]
        rot = angulo_dict["Pelvic_Rotation"]
        foot = angulo_dict["Foot_Progression"]
        
        # Asegura que se mantengan solo los ciclos donde todas las curvas están bien definidas
        ciclos_validos = [
            (o, r, f)
            for o, r, f in zip(obli, rot, foot)
            if o is not None and r is not None and f is not None
            and not np.isnan(o).all()
            and not np.isnan(r).all()
            and not np.isnan(f).all()
        ]

        return pd.DataFrame({
            'Pelvic_Obliquity': [c[0] for c in ciclos_validos],
            'Pelvic_Rotation': [c[1] for c in ciclos_validos],
            'Foot_Progression': [c[2] for c in ciclos_validos]
        })


    df_pelvis_der = filtrar_ciclos_validos(angulos_norm_der)
    df_pelvis_izq = filtrar_ciclos_validos(angulos_norm_izq)











    def calculate_steps_pie(segment, filtered_indices_right, filtered_indices_left, time,
                        heel_x_right, heel_x_left, heel_z_right, heel_z_left):
        """
        Calcula los pasos dentro de un segmento.
        Devuelve una lista de tuplas (indice_inicio, indice_fin, tiempo_paso, longitud_paso, ancho_paso_z, tipo_paso).
        - tipo_paso: "derecho" (inicia con izquierdo) o "izquierdo" (inicia con derecho).
        - ancho_paso_z: distancia en el eje Z entre los pies al inicio del paso.
        """
        steps = []
        all_cycles = []

        # Agregar ciclos derechos (inicio y fin)
        for cycle in filtered_indices_right:
            inicio, fin = cycle
            if inicio in segment and fin in segment:
                all_cycles.append((inicio, "right"))
                all_cycles.append((fin, "right"))

        # Agregar ciclos izquierdos (inicio y fin)
        for cycle in filtered_indices_left:
            inicio, fin = cycle
            if inicio in segment and fin in segment:
                all_cycles.append((inicio, "left"))
                all_cycles.append((fin, "left"))

        # Ordenar por tiempo (usando los índices para acceder a los tiempos)
        all_cycles.sort(key=lambda x: time[x[0]])
        unique_cycles = []
        seen_indices = set()
        for cycle in all_cycles:
            if cycle[0] not in seen_indices:
                unique_cycles.append(cycle)
                seen_indices.add(cycle[0])

        # Identificar pasos (alternancia right -> left o left -> right)
        for i in range(len(unique_cycles) - 1):
            current_idx, current_pie = unique_cycles[i]
            next_idx, next_pie = unique_cycles[i + 1]

            # Paso izquierdo: right -> left
            if current_pie == "right" and next_pie == "left":
                tiempo_paso = time[next_idx] - time[current_idx]
                longitud_paso = abs(heel_x_left[next_idx] - heel_x_right[current_idx])
                ancho_paso_z = abs(heel_z_left[next_idx] - heel_z_right[current_idx])
                steps.append((current_idx, next_idx, tiempo_paso, longitud_paso, ancho_paso_z, "izquierdo"))

            # Paso derecho: left -> right
            elif current_pie == "left" and next_pie == "right":
                tiempo_paso = time[next_idx] - time[current_idx]
                longitud_paso = abs(heel_x_right[next_idx] - heel_x_left[current_idx])
                ancho_paso_z = abs(heel_z_right[next_idx] - heel_z_left[current_idx])
                steps.append((current_idx, next_idx, tiempo_paso, longitud_paso, ancho_paso_z, "derecho"))

        return steps
    
    # Calcular promedios
    def calcular_promedios(pasos):
        if not pasos:
            return 0.0, 0.0, 0.0
        tiempos = [paso[2] for paso in pasos]
        longitudes = [paso[3] for paso in pasos]
        anchos_z = [paso[4] for paso in pasos]

        return np.mean(tiempos), np.mean(longitudes), np.mean(anchos_z)
    
    def calcular_totales(all_steps):
        """
        Calcula:
        - Tiempo total: suma de todos los tiempos de paso individuales.
        - Longitud total: suma de todas las longitudes de paso.
        - Cantidad total de pasos.
        """
        if not all_steps:
            return 0.0, 0.0, 0

        # Aplanar la lista de pasos (todos los segmentos juntos)
        pasos_totales = [paso for segmento in all_steps for paso in segmento]

        tiempo_total = sum(paso[2] for paso in pasos_totales)  # Suma de tiempos de paso
        longitud_total = sum(paso[3] for paso in pasos_totales)  # Suma de longitudes
        cantidad_pasos = len(pasos_totales)  # Total de pasos

        return tiempo_total, longitud_total, cantidad_pasos
    
    def calculate_gait_phases(filtered_indices, min_times_toe, time, side):
        """
        Calcula los tiempos de apoyo y balanceo para cada ciclo
        """
        phases_data = []

        for i, cycle in enumerate(filtered_indices):
            inicio, fin = cycle
            ciclo_times = time[inicio:fin + 1]
            ciclo_duration = ciclo_times[-1] - ciclo_times[0]

            # Encontrar el mínimo local dentro del ciclo
            ciclo_min_times = [t for t in min_times_toe if inicio <= np.where(time == t)[0][0] <= fin]

            if len(ciclo_min_times) > 0:
                min_time = ciclo_min_times[0]

                # Calcular tiempos de balanceo (antes del mínimo) y apoyo (después del mínimo)
                swing_time = min_time - ciclo_times[0]
                stance_time = ciclo_times[-1] - min_time

                # Calcular porcentajes
                swing_percent = (swing_time / ciclo_duration) * 100
                stance_percent = (stance_time / ciclo_duration) * 100

                phases_data.append({
                    'cycle': i + 1,
                    'duration': ciclo_duration,
                    'swing_time': swing_time,
                    'stance_time': stance_time,
                    'swing_percent': swing_percent,
                    'stance_percent': stance_percent
                })

        return phases_data
    # Función para imprimir los resultados
    def print_phases_results(phases_data, side):
        

        if not phases_data:
            return


        # Calcular promedios
        avg_duration = np.mean([p['duration'] for p in phases_data])
        avg_swing = np.mean([p['swing_time'] for p in phases_data])
        avg_stance = np.mean([p['stance_time'] for p in phases_data])
        avg_swing_percent = np.mean([p['swing_percent'] for p in phases_data])
        avg_stance_percent = np.mean([p['stance_percent'] for p in phases_data])

        return avg_swing, avg_stance, avg_swing_percent, avg_stance_percent

    


    # --- Cálculo de parámetros espaciotemporales ---
    def calcular_parametros_espaciotemporales():
        """Calcula todos los parámetros espaciotemporales"""
        # Duración y longitud de ciclos
        durations_right = [t[1]-t[0] for t in filtered_times_right]
        durations_left = [t[1]-t[0] for t in filtered_times_left]
       
        heel_x_right = markers.data["Rashel:RHeel"][:, 0]
        heel_x_left = markers.data["Rashel:LHeel"][:, 0]
        lengths_right = [abs(heel_x_right[fin]-heel_x_right[inicio]) for inicio, fin in filtered_indices_right]
        lengths_left = [abs(heel_x_left[fin]-heel_x_left[inicio]) for inicio, fin in filtered_indices_left]
        
        # Cálculo de pasos
    
        #Coordenadas en z
        heel_z_right = markers.data["Rashel:RHeel"][:, 2]
        heel_z_left = markers.data["Rashel:LHeel"][:, 2]

        all_steps = []

        for i, segmento in enumerate(segmentos):
            steps = calculate_steps_pie(segmento, filtered_indices_right, filtered_indices_left, time,
                                heel_x_right, heel_x_left, heel_z_right, heel_z_left)
            all_steps.append(steps)


        # Separar pasos derechos e izquierdos
        pasos_derechos = [step for steps in all_steps for step in steps if step[5] == "derecho"]
        pasos_izquierdos = [step for steps in all_steps for step in steps if step[5] == "izquierdo"]



        tiempo_prom_derecho, longitud_prom_derecho, ancho_z_prom_derecho = calcular_promedios(pasos_derechos)
        tiempo_prom_izquierdo, longitud_prom_izquierdo, ancho_z_prom_izquierdo = calcular_promedios(pasos_izquierdos)

        tiempo_total, longitud_total, total_pasos = calcular_totales(all_steps)
        # Cálculo de velocidad y cadencia global
        if tiempo_total > 0:
            velocidad_global = longitud_total / tiempo_total  # m/s
            cadencia_global = (total_pasos / tiempo_total) * 60  # pasos/min
        else:
            velocidad_global = 0.0
            cadencia_global = 0.0

        cutoff = 15
        fs = 100

        
        toe_y_right = markers.data["Rashel:RToeIn"][:, 1]  # Coordenada Y del talón derecho
        toe_y_right = low_pass_filter(toe_y_right, cutoff, fs, order=3)

        toe_y_left = markers.data["Rashel:LToeIn"][:, 1]   # Coordenada Y del talón izquierdo
        toe_y_left = low_pass_filter(toe_y_left, cutoff, fs, order=3)

    



        # Detectar mínimos locales
        R_toe_min_indices = argrelextrema(toe_y_right, np.less, order=30)[0]
        L_toe_min_indices = argrelextrema(toe_y_left, np.less, order=30)[0]

        # Obtener los tiempos de los mínimos locales
        R_min_times_toe = time[R_toe_min_indices]
        L_min_times_toe = time[L_toe_min_indices]

        # Calcular fases para ambos pies
        right_phases = calculate_gait_phases(filtered_indices_right, R_min_times_toe, time, "derecho")
        left_phases = calculate_gait_phases(filtered_indices_left, L_min_times_toe, time, "izquierdo")

        avg_swing_der, avg_stance_der, avg_swing_percent_der, avg_stance_percent_der = print_phases_results(right_phases, "derecho")
        avg_swing_izq, avg_stance_izq, avg_swing_percent_izq, avg_stance_percent_izq = print_phases_results(left_phases, "izquierdo")





        
        return {
            "duracion_ciclo_derecho": np.mean(durations_right),
            "duracion_ciclo_izquierdo": np.mean(durations_left),
            "longitud_ciclo_derecho": np.mean(lengths_right),
            "longitud_ciclo_izquierdo": np.mean(lengths_left),
            "tiempo_paso_derecho": tiempo_prom_derecho,
            "longitud_paso_derecho": longitud_prom_derecho,
            "ancho_paso_derecho": ancho_z_prom_derecho,
            "tiempo_paso_izquierdo": tiempo_prom_izquierdo,
            "longitud_paso_izquierdo": longitud_prom_izquierdo,
            "ancho_paso_izquierdo": ancho_z_prom_izquierdo,
            "velocidad": velocidad_global,
            "cadencia": cadencia_global,
            "num_ciclos_derecho": len(filtered_indices_right),
            "num_ciclos_izquierdo": len(filtered_indices_left),
            "num_pasos": len(all_steps),
            "tiempo_balanceo_derecho": avg_swing_der, 
            "tiempo_apoyo_derecho": avg_stance_der, 
            "tiempo_balanceo_izquierdo": avg_swing_izq, 
            "tiempo_apoyo_izquierdo": avg_stance_izq, 
            "balanceo_derecho": avg_swing_percent_der, 
            "apoyo_derecho": avg_stance_percent_der, 
            "balanceo_izquierdo": avg_swing_percent_izq, 
            "apoyo_izquierdo": avg_stance_percent_izq
        }

    parametros_espaciotemporales = calcular_parametros_espaciotemporales()



    return (
        pd.DataFrame(curvas_tobillo_derecho), pd.DataFrame(curvas_tobillo_izquierdo),
        pd.DataFrame(curvas_rodilla_derecha), pd.DataFrame(curvas_rodilla_izquierda),
        pd.DataFrame(curvas_cadera_derecha), pd.DataFrame(curvas_cadera_izquierda),
        df_pelvis_der, df_pelvis_izq,
        parametros_espaciotemporales
    )