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
from analisis_plantillas_modular import *
from calibracion import xx_data, yy_data

def procesar_archivo_plantillas(df_der, df_izq):
    """
    Procesa una sola pasada por pie (derecho e izquierdo) y devuelve datos listos para graficar en Dash.
    """
    print("📥 Entrando a procesar_archivo_plantillas()")

    try:
        from analisis_plantillas_modular import (
            preprocesar_datos,
            correcciones,
            detectar_eventos,
            detectar_fases_validas,
            calcular_cop
        )

        raw_der = [df_der.drop(columns=["Hora"], errors="ignore").copy()]
        raw_izq = [df_izq.drop(columns=["Hora"], errors="ignore").copy()]
        print("✅ Datos RAW preparados")

        filt_der, filt_izq, sum_der, sum_izq = preprocesar_datos(raw_der, raw_izq, xx_data, yy_data)
        print("✅ preprocesar_datos ejecutado")

        filt_der, filt_izq, sum_der, sum_izq = correcciones(sum_der, sum_izq, filt_der, filt_izq)
        print("✅ correcciones ejecutadas")

        eventos_der = detectar_eventos(sum_der, filt_der, lado='D')
        print("✅ eventos_der detectados")

        eventos_izq = detectar_eventos(sum_izq, filt_izq, lado='I')
        print("✅ eventos_izq detectados")

        (min_indices_der, min_tiempos_der,
         ciclos_apoyo_der_all, indices_apoyo_der_all,
         ciclos_completos_der_all, sum_der_corr_list, filt_der_corr_list) = eventos_der

        (min_indices_izq, min_tiempos_izq,
         ciclos_apoyo_izq_all, indices_apoyo_izq_all,
         ciclos_completos_izq_all, sum_izq_corr_list, filt_izq_corr_list) = eventos_izq

        ciclos_apoyo_der = ciclos_apoyo_der_all[0] if ciclos_apoyo_der_all else []
        indices_apoyo_der = indices_apoyo_der_all[0] if indices_apoyo_der_all else []
        ciclos_completos_der = ciclos_completos_der_all[0] if ciclos_completos_der_all else []

        ciclos_apoyo_izq = ciclos_apoyo_izq_all[0] if ciclos_apoyo_izq_all else []
        indices_apoyo_izq = indices_apoyo_izq_all[0] if indices_apoyo_izq_all else []
        ciclos_completos_izq = ciclos_completos_izq_all[0] if ciclos_completos_izq_all else []

        sum_der_corr = sum_der_corr_list[0]
        filt_der_corr = filt_der_corr_list[0]
        sum_izq_corr = sum_izq_corr_list[0]
        filt_izq_corr = filt_izq_corr_list[0]
        print("✅ Extracción de ciclos y correcciones lista")

        resultados_der = detectar_fases_validas(
            [ciclos_apoyo_der], [indices_apoyo_der], [ciclos_completos_der], lado='D'
        )
        print("✅ Fases válidas pie derecho")

        resultados_izq = detectar_fases_validas(
            [ciclos_apoyo_izq], [indices_apoyo_izq], [ciclos_completos_izq], lado='I'
        )
        print("✅ Fases válidas pie izquierdo")

        copx_der, copy_der = calcular_cop(
            [filt_der_corr], [sum_der_corr], [indices_apoyo_der], [resultados_der], lado='D'
        )
        print("✅ COP pie derecho")

        copx_izq, copy_izq = calcular_cop(
            [filt_izq_corr], [sum_izq_corr], [indices_apoyo_izq], [resultados_izq], lado='I'
        )

        print("✅ COP pie izquierdo")

        print(f"📦 Nuevas longitudes: filt_der_corr={len([filt_der_corr])}, sum_der_corr={len([sum_der_corr])}")
        
        print("🧪 Ejemplo de índices_apoyo_der:", indices_apoyo_der[:3])
        print("🧪 Tipo de elementos:", [type(x) for x in indices_apoyo_der[:3]])


        return {
            'filt_der': filt_der_corr,
            'filt_izq': filt_izq_corr,
            'sum_der': sum_der_corr,
            'sum_izq': sum_izq_corr,
            'ciclos_apoyo_der': ciclos_apoyo_der,
            'ciclos_apoyo_izq': ciclos_apoyo_izq,
            'ciclos_completos_der': ciclos_completos_der,
            'ciclos_completos_izq': ciclos_completos_izq,
            'indices_apoyo_der': indices_apoyo_der,
            'indices_apoyo_izq': indices_apoyo_izq,
            'resultados_der': resultados_der,
            'resultados_izq': resultados_izq,
            'copx_der': copx_der,
            'copy_der': copy_der,
            'copx_izq': copx_izq,
            'copy_izq': copy_izq,
            'min_indices_der': min_indices_der,
            'min_indices_izq': min_indices_izq,
            'min_tiempos_der': min_tiempos_der,
            'min_tiempos_izq': min_tiempos_izq,
        }

    except Exception as e:
        import traceback
        print(f"❌ Error en procesar_archivo_plantillas: {e}")
        traceback.print_exc()
        raise


