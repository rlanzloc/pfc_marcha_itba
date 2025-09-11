# === analisis_plantillas_modular.py ===

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from scipy.interpolate import interp1d
from calibracion import xx_data, yy_data
from scipy.interpolate import PchipInterpolator


def preprocesar_datos(raw_der, raw_izq, xx_data, yy_data):
    # === Reordenar sensores izquierdos S3<->S5, S6<->S8 ===
    for df in raw_izq:
        col_3 = [col for col in df.columns if col.endswith('S3')][0]
        col_5 = [col for col in df.columns if col.endswith('S5')][0]
        col_6 = [col for col in df.columns if col.endswith('S6')][0]
        col_8 = [col for col in df.columns if col.endswith('S8')][0]
        df[[col_3, col_5]] = df[[col_5, col_3]]
        df[[col_6, col_8]] = df[[col_8, col_6]]
        tiempo_col = [col for col in df.columns if col.lower().startswith('tiempo')][0]
        sensores_ordenados = sorted(
            [col for col in df.columns if col != tiempo_col],
            key=lambda x: int(x.split('S')[-1])
        )
        columnas_finales = [tiempo_col] + sensores_ordenados
        df.loc[:, :] = df[columnas_finales]

    # === Limpieza e interpolación de tiempos anómalos ===
    def limpiar_tiempo_anomalo(df_list, frecuencia_hz=100):
        dfs_reamostrados = []
        muestras_faltantes_ubicaciones = []
        intervalo = 1 / frecuencia_hz
        tolerancia = intervalo * 0.1

        for df in df_list:
            df_copy = df.copy()
            df_copy['Tiempo'] = df_copy['Tiempo'] / 1000.0
            tiempo_real = df_copy['Tiempo'].values
            tiempos_ideales = np.arange(tiempo_real[0], tiempo_real[-1] + intervalo, intervalo)
            df_copy = df_copy.set_index('Tiempo')
            df_nuevo = []
            ubicaciones_faltantes = []

            for t in tiempos_ideales:
                diffs = np.abs(df_copy.index - t)
                if len(diffs) == 0 or np.min(diffs) > tolerancia:
                    ubicaciones_faltantes.append(t)
                    fila = df_copy[df_copy.index < t].iloc[-1]
                else:
                    fila = df_copy.iloc[np.argmin(diffs)]
                fila_dict = fila.to_dict()
                fila_dict['Tiempo'] = round(t, 3)
                df_nuevo.append(fila_dict)

            df_final = pd.DataFrame(df_nuevo)
            for col in df_final.columns:
                if col != 'Tiempo':
                    df_final[col] = df_final[col].astype(int)
            dfs_reamostrados.append(df_final)
            muestras_faltantes_ubicaciones.append(ubicaciones_faltantes)

        return dfs_reamostrados, muestras_faltantes_ubicaciones

    raw_der_final, _ = limpiar_tiempo_anomalo(raw_der)
    raw_izq_final, _ = limpiar_tiempo_anomalo(raw_izq)

    # === Calibración e interpolación ===
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

    dataframes_der, mV_der = preproc_df(raw_der_final)
    dataframes_izq, mV_izq = preproc_df(raw_izq_final)

    # === Filtro pasa bajos (butter) ===
    def apply_lowpass_filter(data, cutoff_freq, sampling_rate):
        nyquist = 0.5 * sampling_rate
        b, a = butter(4, cutoff_freq / nyquist, btype='low')
        return np.maximum(filtfilt(b, a, data), 0)
    


    # === Suavizado de saturación con spline ===
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

    # === Filtro Savitzky-Golay luego de spline ===
    window_length = 11
    polyorder = 3
    sampling_rate = 100
    cutoff_frequency = 20

    filt_der = []
    filt_izq = []

    # --- IZQUIERDO: Butterworth -> spline -> SavGol ---
    for df in dataframes_izq:
        # 1) Butterworth columna a columna (con clip a 0)
        df_bw = df.copy()
        for col in df_bw.columns:
            df_bw[col] = apply_lowpass_filter(df_bw[col], cutoff_frequency, sampling_rate)

        # 2) Suavizado de mesetas con spline (sobre la salida del Butterworth)
        df_suave = suavizar_saturacion_df_spline_direccion(
            df_bw, margen=2, delta=2, saturacion=25, lado='I'
        )

        # 3) Savitzky–Golay final
        df_filtered = df_suave.copy()
        for col in df_filtered.columns:
            df_filtered[col] = savgol_filter(df_filtered[col], window_length=window_length, polyorder=polyorder)

        filt_izq.append(df_filtered)

    # --- DERECHO: Butterworth -> spline -> SavGol ---
    for df in dataframes_der:
        df_bw = df.copy()
        for col in df_bw.columns:
            df_bw[col] = apply_lowpass_filter(df_bw[col], cutoff_frequency, sampling_rate)

        df_suave = suavizar_saturacion_df_spline_direccion(
            df_bw, margen=1, delta=1, saturacion=25, lado='D'
        )

        df_filtered = df_suave.copy()
        for col in df_filtered.columns:
            df_filtered[col] = savgol_filter(df_filtered[col], window_length=window_length, polyorder=polyorder)

        filt_der.append(df_filtered)

    # === Suma de sensores ===
    sum_der = [df.sum(axis=1).to_frame(name="Suma") for df in filt_der]
    sum_izq = [df.sum(axis=1).to_frame(name="Suma") for df in filt_izq]
    sum_der[0].index = sum_der[0].index.round(3)
    sum_izq[0].index = sum_izq[0].index.round(3)

    

    return filt_der, filt_izq, sum_der, sum_izq


def correcciones(sums_der, sums_izq, filt_der, filt_izq):

    def subset(sums, t_inicio, t_fin):
        if isinstance(sums, list):
            sums_subset = [df[(df.index >= t_inicio) & (df.index <= t_fin)] for df in sums]
            sums_subset = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in sums_subset]
        else:
            sums_subset = sums[(sums.index >= t_inicio) & (sums.index <= t_fin)]
        return sums_subset

    # Sincronizar tiempos de sumas y señales entre lados
    for i, (sum_d, sum_i, filt_d, filt_i) in enumerate(zip(sums_der, sums_izq, filt_der, filt_izq)):
        tf_d = sum_d.index[-1]
        tf_i = sum_i.index[-1]
        dif = tf_d - tf_i

        sum_d = sum_d.copy()
        filt_d = filt_d.copy()

        sum_d.index = sum_d.index - dif
        filt_d.index = filt_d.index - dif

        ti = sum_i.index[0]

        # Recortar ambos lados al mismo intervalo
        sums_der[i] = subset(sum_d, ti, tf_i)
        sums_izq[i] = subset(sum_i, ti, tf_i)

        filt_der[i] = subset(filt_d, ti, tf_i)
        filt_izq[i] = subset(filt_i, ti, tf_i)

    # Corrección de baseline
    def correccion_baseline_general(data, modo='sensores', cero_negativos=True):
        if isinstance(data, list):
            resultados_corr = []
            for df in data:
                if modo == 'sensores':
                    offset = df.min()
                    corr = df - offset
                else:
                    fuerza = df.iloc[:, 0]
                    offset = fuerza.min()
                    corr = pd.DataFrame(fuerza - offset, index=df.index)
                if cero_negativos:
                    corr[corr < 0] = 0
                resultados_corr.append(corr)
            return resultados_corr
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
            return corr

    # Aplicar corrección a señales y sumas
    filt_der = correccion_baseline_general(filt_der, modo='sensores')
    filt_izq = correccion_baseline_general(filt_izq, modo='sensores')
    sums_der = correccion_baseline_general(sums_der, modo='suma')
    sums_izq = correccion_baseline_general(sums_izq, modo='suma')

    return filt_der, filt_izq, sums_der, sums_izq


def detectar_eventos(suma_list, filt_list, lado='D'):
    """
    Detecta eventos y segmenta ciclos de marcha usando el algoritmo avanzado.
    
    Parámetros:
    - suma_list: lista de DataFrames con la suma de sensores (una por pasada).
    - filt_list: lista de DataFrames con todos los sensores (una por pasada).
    - lado: 'D' o 'I' para seleccionar el pie.
    
    Retorna:
    - all_min_idx: índices de contacto inicial
    - all_min_time: tiempos de contacto inicial
    - all_apoyos: lista de ciclos de apoyo
    - all_idx_apoyos: índices de inicio y fin de apoyo
    - all_completos: lista de ciclos completos
    - all_sum_corr: suma corregida
    - all_filt_corr: sensores corregidos
    """
    frecuencia_hz = 100
    subida_minima = 5
    ventana_retroceso_s = 0.4
    peso_maximo = 5
    min_separacion_s = 0.7
    min_prominence_factor = 0.03
    min_peak_height = 0.07
    threshold_ratio = 0.04
    aporte_max_rel = 0.03

    ventana_retroceso = int(ventana_retroceso_s * frecuencia_hz)
    min_separacion = int(min_separacion_s * frecuencia_hz)
    max_dist_pico_CI = int(0.6 * frecuencia_hz)

    if lado == 'I':
        sensores = [f'Izquierda_S{i}' for i in range(1, 9)]
    elif lado == 'D':
        sensores = [f'Derecha_S{i}' for i in range(1, 9)]
    else:
        raise ValueError("El parámetro 'lado' debe ser 'I' o 'D'.")

    min_indices_all = []
    min_tiempos_all = []
    ciclos_apoyo_all = []
    indices_apoyo_all = []
    ciclos_completos_all = []
    suma_corregida_all = []
    filt_corr_list = []

    for i_passada, serie in enumerate(suma_list):
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
                ini_rescate = max(ini_rescate, min_ini)
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

            idx_pico1 = idx_ini + int((idx_fin - idx_ini) * 0.3)
            idx_pico2 = idx_ini + int((idx_fin - idx_ini) * 0.7)
            if len(peaks_ciclo) >= 1:
                idx_pico1 = idx_ini + peaks_ciclo[0]
            if len(peaks_ciclo) >= 2:
                idx_pico2 = idx_ini + peaks_ciclo[1]

            if df_passada is not None:
                for sensor in sensores:
                    sensor_num = int(sensor.split('_')[-1][1:])
                    if sensor_num in [1, 2, 3, 4]:
                        reg_ini, reg_fin = idx_ini, idx_pico1
                    else:
                        reg_ini, reg_fin = idx_pico2, idx_fin

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
                    if idx_fin_apoyo > idx_pico2:
                        ventana_chequeo = int(0.05 * frecuencia_hz)
                        subida_maxima = 3
                        while rel_fin > ventana_chequeo:
                            idx_actual = rel_fin
                            idx_prev = rel_fin - ventana_chequeo
                            subida = datos[idx_actual] - datos[idx_prev]
                            if subida >= subida_maxima:
                                rel_fin = idx_prev
                            else:
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

                ciclos_apoyo.append(apoyo)
                ciclos_completos.append(ciclo_completo)
                indices_apoyo.append((float(t_ini), float(t_fin)))

        if df_passada is not None:
            suma_corregida = df_passada[sensores].sum(axis=1).to_frame()
            suma_corregida_all.append(suma_corregida)
            filt_corr_list.append(df_passada)

        ciclos_apoyo_all.append(ciclos_apoyo)
        indices_apoyo_all.append(indices_apoyo)
        ciclos_completos_all.append(ciclos_completos)

    print(f"🧠 Mínimos detectados {lado}: {len(indices_apoyo_all)}")
    print(f"📏 Ciclos detectados completos: {len(ciclos_completos_all)}")
    print(f"🦶 Apoyos detectados: {len(ciclos_apoyo_all)}")
    
    # Validación antes de retornar
    for sublist in indices_apoyo_all:
        if not isinstance(sublist, list) or (sublist and not isinstance(sublist[0], tuple)):
            print("⚠️ Alerta: Estructura inesperada en indices_apoyo_all")


    return (
        min_indices_all,
        min_tiempos_all,
        ciclos_apoyo_all,
        indices_apoyo_all,
        ciclos_completos_all,
        suma_corregida_all,
        filt_corr_list
    )


def detectar_fases_validas(ciclos_apoyo_all, indices_all, completos_all, lado='D',
                            time_threshold=0.5, 
                            min_prominence_factor=0.05,
                            min_peak_height=0.1,
                            support_ratio_threshold=(0.60, 0.80), 
                            peso_kg=52):
    """
    Detección y validación de fases de apoyo usando reglas avanzadas.

    Parámetros:
        ciclos_apoyo_all: lista de listas de Series (fase de apoyo)
        indices_all: lista de (inicio, fin) en tiempo real de cada apoyo
        completos_all: lista de listas de Series con los ciclos completos
        lado: 'D' o 'I'
        Otros: umbrales de detección y validación

    Retorna:
        Lista de dicts por pasada con claves: 'valid', 'invalid', 'details'
    """
    resultados = []

    for ciclos_apoyo, indices_ciclos, ciclos_completos in zip(ciclos_apoyo_all, indices_all, completos_all):
        
        print(f"🔬 detectando en pasada con {len(ciclos_apoyo)} ciclos de apoyo")

        if len(ciclos_apoyo) < 2:
            resultados.append({"valid": [], "invalid": [], "details": []})
            continue

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
                    
                print(f"🔎 Ciclo {j} - duración: {duracion_completa:.3f}, picos: {len(peaks)}, valles: {len(valleys_filtrados)}")


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
                "duration": duracion_completa,
                "support_ratio": proporcion_apoyo
            })

        resultados.append({
            "valid": ciclos_validos,
            "invalid": [j for j in range(len(ciclos_apoyo)) if j not in ciclos_validos],
            "details": ciclo_resultados
        })
        
        print(f"✅ válidos: {len(ciclos_validos)} | ❌ inválidos: {len(ciclo_resultados) - len(ciclos_validos)}")


    return resultados


import traceback

def calcular_cop(filt_list, suma_list, indices_list, resultados_list, lado='D', n_points=100):
    """
    Cálculo real del Centro de Presión (COP) con interpolación y filtrado, listo para la interfaz.

    Devuelve:
        copx_all, copy_all: listas de listas de Series, una por ciclo válido por pasada
    """
    print("🔧 Iniciando calcular_cop()")

    # Coordenadas de sensores
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

    if lado.upper() == 'I':
        coord_df["X"] = -coord_df["X"]
        coord_df["X"] += abs(coord_df["X"].min()) + 10
        sufijo = 'Izquierda'
    else:
        coord_df["X"] += abs(coord_df["X"].min()) + 10
        sufijo = 'Derecha'

    coord_df.index = [f"{sufijo}_{s}" for s in coord_df.index]

    copx_all = []
    copy_all = []

    print(f"📦 len(filt_list): {len(filt_list)}")
    print(f"📦 len(suma_list): {len(suma_list)}")
    print(f"📦 len(indices_list): {len(indices_list)}")
    print(f"📦 len(resultados_list): {len(resultados_list)}")

    for pasada_i, (filt_df, suma_df, indices, resultados) in enumerate(zip(filt_list, suma_list, indices_list, resultados_list)):
        print(f"\n🔁 Pasada {pasada_i}")
        copx_pasada = []
        copy_pasada = []

        try:
            detalles = resultados[0]['details'] if isinstance(resultados, list) else resultados['details']
            valid_flags = [r['is_valid'] for r in detalles]
        except Exception as e:
            print(f"❌ Error accediendo a 'details' en pasada {pasada_i}: {e}")
            traceback.print_exc()
            continue

        for ciclo_i, ((ini, fin), es_valido) in enumerate(zip(indices, valid_flags)):
            print(f"🔬 Ciclo {ciclo_i}: ini={ini}, fin={fin}, válido={es_valido}")
            try:
                if not es_valido:
                    print("↪️ Ciclo inválido, se salta")
                    continue

                ciclo_fuerza = filt_df.loc[ini:fin]
                ciclo_vgrf = suma_df.loc[ini:fin]

                if ciclo_fuerza.empty or ciclo_vgrf.empty:
                    print("⚠️ Ciclo vacío, se salta")
                    continue
                
                print("🧪 ciclo_fuerza.columns:", ciclo_fuerza.columns.tolist())
                print("📎 coord_df.index:", coord_df.index.tolist())

                coord_map = coord_df.to_dict(orient='index')
                try:
                    coord = np.array([[coord_map[col]['X'], coord_map[col]['Y']] for col in ciclo_fuerza.columns])
                except KeyError as e:
                    print(f"❌ Error: columna no encontrada en coord_map: {e}")
                    raise
                matriz_fuerza = ciclo_fuerza.to_numpy()
                vector_vgrf = ciclo_vgrf.to_numpy().flatten()

                if len(vector_vgrf) < 2:
                    print("⚠️ vector_vgrf muy corto, se salta")
                    continue

                matriz_fuerza[matriz_fuerza < 0.07] = 0
                vgrf_cop = vector_vgrf.copy()
                vgrf_cop[vgrf_cop < 0.07 * np.nanmax(vector_vgrf)] = np.nan

                if np.all(np.isnan(vgrf_cop)):
                    print("⚠️ vgrf_cop sin datos válidos, se salta")
                    continue

                print("📈 Calculando COP crudo")
                cop_x_raw = np.sum(matriz_fuerza * coord[:, 0], axis=1) / vgrf_cop
                cop_y_raw = np.sum(matriz_fuerza * coord[:, 1], axis=1) / vgrf_cop

                mask = ~np.isnan(cop_x_raw) & ~np.isnan(cop_y_raw)
                if np.sum(mask) < 10:
                    print("⚠️ Menos de 10 puntos válidos, se salta")
                    continue

                x_old = np.linspace(0, 100, len(vector_vgrf))
                x_new = np.linspace(0, 100, n_points)

                fx = interp1d(x_old[mask], cop_x_raw[mask], kind='linear', fill_value=np.nan, bounds_error=False)
                fy = interp1d(x_old[mask], cop_y_raw[mask], kind='linear', fill_value=np.nan, bounds_error=False)

                copx_interp = pd.Series(fx(x_new))
                copy_interp = pd.Series(fy(x_new))

                # Recorte por fuera del rango confiable
                x_min, x_max = x_old[mask].min(), x_old[mask].max()
                copx_interp[x_new < x_min] = np.nan
                copx_interp[x_new > x_max] = np.nan
                copy_interp[x_new < x_min] = np.nan
                copy_interp[x_new > x_max] = np.nan

                print("✅ COP interpolado con éxito")
                copx_pasada.append(copx_interp)
                copy_pasada.append(copy_interp)

            except Exception as e:
                print(f"❌ Error en ciclo {ciclo_i} de pasada {pasada_i}: {e}")
                traceback.print_exc()
                continue

        copx_all.append(copx_pasada)
        copy_all.append(copy_pasada)

    print("🏁 Finalizó calcular_cop()\n")
    return copx_all, copy_all



