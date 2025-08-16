import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from plotly.subplots import make_subplots
from matplotlib.patches import Circle
from matplotlib import cm
import matplotlib.patheffects as pe
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from plotly.subplots import make_subplots
import traceback
from plotly.colors import sample_colorscale

def generar_figuras_cinetico(data):
    BW = 52  # peso corporal fijo para normalización

    # Asegurar que los datos sean listas aunque haya un solo elemento
    def ensure_list(x):
        return x if isinstance(x, list) else [x]
    
    # Verificar si hay datos válidos
    has_valid_der = data['resultados_der'][0]['valid'] if 'resultados_der' in data else False
    has_valid_izq = data['resultados_izq'][0]['valid'] if 'resultados_izq' in data else False
    
    if not has_valid_der and not has_valid_izq:
        fig_empty = go.Figure()
        fig_empty.update_layout(title="No se detectaron ciclos válidos")
        return [fig_empty] * 6
    
    def plot_ciclos_validos_invalidos(sum_izq, sum_der,
                                    res_izq, res_der,
                                    idx_min_izq, idx_min_der):
        import plotly.graph_objects as go
        fig = go.Figure()
        BW = 52.0

        # si vienen como [[...]], desempaquetar
        def first_or_self(x):
            return x[0] if isinstance(x, list) and x and isinstance(x[0], list) else x
        idx_min_der = first_or_self(idx_min_der)
        idx_min_izq = first_or_self(idx_min_izq)

        col_der = sum_der.columns[0]; col_izq = sum_izq.columns[0]
        used = {"Der_ok": False, "Der_bad": False, "Izq_ok": False, "Izq_bad": False}

        # Derecho (opacidad 1.0)
        for det in res_der['details']:
            j = det['cycle_num']
            if j+1 >= len(idx_min_der): continue
            t0 = sum_der.index[int(idx_min_der[j])]
            t1 = sum_der.index[int(idx_min_der[j+1])]
            y = (sum_der.loc[t0:t1, col_der] / BW)
            color = '#2ca02c' if det['is_valid'] else '#d62728'
            key   = 'Der_ok'  if det['is_valid'] else 'Der_bad'
            name  = 'Der Válido'  if det['is_valid'] else 'Der Inválido'
            fig.add_trace(go.Scatter(x=y.index, y=y.values, mode='lines',
                                    line=dict(color=color, width=2.5),
                                    opacity=1.0,
                                    name=None if used[key] else name,
                                    showlegend=(not used[key])))
            used[key] = True

        # Izquierdo (opacidad 0.5)
        for det in res_izq['details']:
            j = det['cycle_num']
            if j+1 >= len(idx_min_izq): continue
            t0 = sum_izq.index[int(idx_min_izq[j])]
            t1 = sum_izq.index[int(idx_min_izq[j+1])]
            y = (sum_izq.loc[t0:t1, col_izq] / BW)
            color = '#2ca02c' if det['is_valid'] else '#d62728'
            key   = 'Izq_ok'  if det['is_valid'] else 'Izq_bad'
            name  = 'Izq Válido'  if det['is_valid'] else 'Izq Inválido'
            fig.add_trace(go.Scatter(x=y.index, y=y.values, mode='lines',
                                    line=dict(color=color, width=2.5),
                                    opacity=0.5,
                                    name=None if used[key] else name,
                                    showlegend=(not used[key])))
            used[key] = True

        #xmin = min(sum_der.index.min(), sum_izq.index.min())
        #xmax = max(sum_der.index.max(), sum_izq.index.max())
        fig.update_layout(title=dict(text="<b>Ciclos válidos e inválidos (vGRF)</b>"),
                        xaxis_title="Tiempo (s)", yaxis_title="vGRF (%BW)",
                        template="plotly_white",
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        #fig.update_xaxes(range=[xmin, xmax])
        return fig



    def plot_vgrf_por_fases(df_sensores_list, resultados, indices_ciclos,
                        peso_kg=52, lado="R"):

    

        # Normalizar entradas
        if hasattr(df_sensores_list, "sum"):  # DataFrame -> [DF]
            df_sensores_list = [df_sensores_list]
        if isinstance(resultados, dict):      # dict -> [dict]
            resultados = [resultados]

        def normalize_indices(obj):
            # [[(t0,t1), ...]]
            if isinstance(obj, list) and obj and isinstance(obj[0], list) and obj[0] and isinstance(obj[0][0], tuple):
                return obj
            # [(t0,t1), ...] -> [[...]]
            if isinstance(obj, list) and obj and isinstance(obj[0], tuple):
                return [obj]
            # (t0,t1) -> [[(t0,t1)]]
            if isinstance(obj, tuple) and len(obj) == 2:
                return [[obj]]
            # [t0, t1, t2, t3, ...] -> [[(t0,t1), (t2,t3), ...]]
            if isinstance(obj, list) and obj and isinstance(obj[0], (float, int)):
                paired = list(zip(obj[::2], obj[1::2]))
                return [paired]
            return [[]]

        indices_ciclos = normalize_indices(indices_ciclos)

        sufijo = "Derecha" if lado.upper() == "R" else "Izquierda"
        colores_fases = {
            'loading_response': '#9467bd',
            'midstance':        '#d62728',
            'terminal_stance':  '#e377c2',
            'pre_swing':        '#2ca02c'
        }

        def idx_nearest(index, t):
            try:
                return int(index.get_loc(t))
            except Exception:
                try:
                    pos = index.get_indexer([t], method='nearest')[0]
                except Exception:
                    vals = np.asarray(index.values, dtype=float)
                    pos = np.searchsorted(vals, float(t))
                if pos < 0: pos = 0
                if pos >= len(index): pos = len(index) - 1
                return int(pos)

        figuras = []

        for pasada_idx in range(min(len(df_sensores_list), len(resultados), len(indices_ciclos))):
            df_sensores    = df_sensores_list[pasada_idx]
            resultado_paso = resultados[pasada_idx]
            indices_paso   = indices_ciclos[pasada_idx]  # lista de (t_ini, t_fin)

            # si viene raro, vuelve a normalizar a lista de pares
            if indices_paso and isinstance(indices_paso, tuple) and len(indices_paso) == 2:
                indices_paso = [indices_paso]
            elif indices_paso and isinstance(indices_paso, list) and indices_paso and isinstance(indices_paso[0], (float, int)):
                indices_paso = list(zip(indices_paso[::2], indices_paso[1::2]))

            details = resultado_paso.get('details', [])

            fig = go.Figure()
            matriz_ciclos = []
            fases_usadas = {fase: False for fase in colores_fases}
            pico_label_usado = False
            valle_label_usado = False

            # recorremos TODOS los detalles y buscamos (t_ini,t_fin) de dos fuentes
            usados = 0
            for j in range(len(details)):
                info = details[j]
                if not info.get('is_valid', False):
                    continue

                # 1) ideal: viene de indices_paso
                if j < len(indices_paso) and isinstance(indices_paso[j], (list, tuple)) and len(indices_paso[j]) == 2:
                    inicio, fin = indices_paso[j]
                # 2) fallback: usar lo que trae el detalle (si está)
                elif 'start_index' in info and 'end_index' in info:
                    inicio, fin = info['start_index'], info['end_index']
                else:
                    continue

                datos_ciclo = df_sensores.loc[inicio:fin].copy()
                if len(datos_ciclo) < 5:
                    continue

                # vGRF total / BW
                suma_total = datos_ciclo.sum(axis=1) / float(peso_kg)

                # 0–100% del apoyo
                x_original = np.linspace(0, 100, len(suma_total))
                x_interp   = np.linspace(0, 100, 100)
                f_interp   = interp1d(x_original, suma_total.values, kind='linear')
                ciclo_interp = f_interp(x_interp)
                matriz_ciclos.append(ciclo_interp)

                # Eventos (o fallback)
                peaks   = info.get('peaks', [])
                valleys = info.get('valleys', [])
                if len(peaks) >= 2 and len(valleys) >= 1:
                    p1, p2 = peaks[0], peaks[1]
                    v      = valleys[0]
                else:
                    p1 = datos_ciclo.index[int(round(0.30*(len(datos_ciclo)-1)))]
                    v  = datos_ciclo.index[int(round(0.50*(len(datos_ciclo)-1)))]
                    p2 = datos_ciclo.index[int(round(0.70*(len(datos_ciclo)-1)))]

                segmentos = {
                    'loading_response': (inicio, p1),
                    'midstance':        (p1, v),
                    'terminal_stance':  (v, p2),
                    'pre_swing':        (p2, fin)
                }

                for fase, (t0, t1) in segmentos.items():
                    i0 = idx_nearest(datos_ciclo.index, t0)
                    i1 = idx_nearest(datos_ciclo.index, t1)
                    if i1 < i0:
                        i0, i1 = i1, i0
                    x_fase = x_original[i0:i1+1]
                    y_fase = suma_total.iloc[i0:i1+1].values
                    fig.add_trace(go.Scatter(
                        x=x_fase, y=y_fase,
                        mode='lines',
                        line=dict(color=colores_fases[fase], width=2, dash='dash'),
                        opacity= 0.7,
                        name=fase.replace("_", " ").capitalize() if not fases_usadas[fase] else None,
                        showlegend=not fases_usadas[fase]
                    ))
                    fases_usadas[fase] = True

                # marcadores si hay eventos reales
                if len(peaks) >= 2 and len(valleys) >= 1:
                    for p in [p1, p2]:
                        ip = idx_nearest(datos_ciclo.index, p)
                        fig.add_trace(go.Scatter(
                            x=[x_original[ip]],
                            y=[suma_total.iloc[ip]],
                            mode='markers',
                            marker=dict(symbol='x', size=8, color='#1f77b4'),
                            name='Pico' if not pico_label_usado else None,
                            showlegend=not pico_label_usado
                        ))
                        pico_label_usado = True
                    iv = idx_nearest(datos_ciclo.index, v)
                    fig.add_trace(go.Scatter(
                        x=[x_original[iv]],
                        y=[suma_total.iloc[iv]],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color='#ff7f0e'),
                        name='Valle' if not valle_label_usado else None,
                        showlegend=not valle_label_usado
                    ))
                    valle_label_usado = True

                usados += 1

            # Promedio ±1SD de lo que sí se usó
            if matriz_ciclos:
                matriz = np.vstack(matriz_ciclos)
                promedio = matriz.mean(axis=0)
                std      = matriz.std(axis=0)
                fig.add_trace(go.Scatter(x=x_interp, y=promedio, mode='lines',
                                        name='Promedio', line=dict(color='black', width=3)))
                fig.add_trace(go.Scatter(x=x_interp, y=promedio + std, mode='lines',
                                        name='+1 SD', line=dict(color='#7f7f7f', width=1, dash='dot'), opacity= 0.5, showlegend=False))
                fig.add_trace(go.Scatter(x=x_interp, y=promedio - std, mode='lines',
                                        name='-1 SD', line=dict(color='#7f7f7f', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(127,127,127,0.20)', opacity=0.5, showlegend=False))

            fig.update_layout(
                title=f"{sufijo} - Ciclo de apoyo normalizado",
                xaxis_title="% ciclo de apoyo",
                yaxis_title="vGRF (%BW)",
                template="plotly_white"
            )
            fig.update_xaxes(range=[0, 100])  # siempre 0–100%

            figuras.append(fig)

        if not figuras:
            figuras = [go.Figure()]

        return figuras
    
    
    def combinar_fases_en_uno(fig_fases_izq: go.Figure,
                          fig_fases_der: go.Figure,
                          subtitulo_izq: str = "",
                          subtitulo_der: str = "") -> go.Figure:
        """
        Une las fases Izq/Der en un solo Figure con 2 subplots,
        título general, subtítulos por subplot y leyenda ÚNICA en el medio.
        """
        fig = make_subplots(
            rows=1, cols=2,
            # el spacing formal lo dejamos chico; vamos a ajustar domains a mano
            horizontal_spacing=0.08,
            subplot_titles=(
                f"Pie Izquierdo<br><sup>{subtitulo_izq}</sup>",
                f"Pie Derecho<br><sup>{subtitulo_der}</sup>"
            ),
        )

        # Pegar trazas: IZQ con leyenda visible
        for tr in fig_fases_izq.data:
            fig.add_trace(tr, row=1, col=1)

        # Pegar trazas: DER sin duplicar leyenda
        for tr in fig_fases_der.data:
            tr.showlegend = False
            fig.add_trace(tr, row=1, col=2)

        # ===== Dominios manuales para abrir un "canal" central de leyenda =====
        # Izquierda: 0.00 → 0.46 | Canal: 0.46 → 0.54 | Derecha: 0.54 → 1.00
        fig.update_xaxes(domain=[0.00, 0.46], row=1, col=1)
        fig.update_xaxes(domain=[0.54, 1.00], row=1, col=2)

        # Ejes + grilla
        grid = '#d9d9d9'
        axis = '#bdbdbd'
        for c in (1, 2):
            fig.update_xaxes(
                title="% Ciclo de apoyo",
                showgrid=True, gridcolor=grid, gridwidth=1,
                zeroline=False, linecolor=axis, ticks='outside',
                ticklen=4, tickcolor=axis,
                range=[0, 100],
                row=1, col=c
            )
            fig.update_yaxes(
                title="vGRF (% BW)",
                showgrid=True, gridcolor=grid, gridwidth=1,
                zeroline=False, linecolor=axis, ticks='outside',
                ticklen=4, tickcolor=axis,
                row=1, col=c
            )

        # Título general + leyenda centrada EN EL CANAL
        fig.update_layout(
            title=dict(text="<b>vGRF normalizado — Detección de eventos y subfases</b>"),
            template="plotly_white",
            autosize=True, height=None, width=None,
            margin=dict(t=120, r=40, b=60, l=60),   # ↑ más margen superior para la leyenda
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=1.08, yanchor="bottom",           # ← arriba, fuera del área de datos
                bgcolor="rgba(255,255,255,0.9)",
                itemsizing="constant",
                font=dict(size=12)
            )
        )

        return fig


    def plot_cop_pie_izq_der(copx_izq_list, copy_izq_list, copx_der_list, copy_der_list, pasada_idx=0):


        # --- Coordenadas base de sensores (mm) ---
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

        # --- Preparar reflejo/offsets (mm) para separar pies ---
        coord_df_izq = coord_df.copy()
        coord_df_izq["X"] = -coord_df_izq["X"]
        offset_izq = abs(coord_df_izq["X"].min()) + 10
        offset_der = abs(coord_df["X"].min()) + 10
        coord_df_izq["X"] += offset_izq

        coord_df_der = coord_df.copy()
        coord_df_der["X"] += offset_der

        # --- Parámetros de dibujo ---
        margen = 20  # mm extra alrededor
        sensor_colores = ['blue', 'yellow', 'grey', 'pink', 'green', 'orange', 'purple', 'red']

        # Diámetro real en mm para los círculos de sensores
        diam_mm = 14.63
        r = diam_mm / 2.0

        # --- Fig con subplots (Izq / Der) ---
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Pie Izquierdo", "Pie Derecho"])
        legend_once = False  # para no duplicar "COP promedio" en leyenda

        for col, (coord_this, copx_list, copy_list, pie_nombre) in enumerate([
            (coord_df_izq, copx_izq_list, copy_izq_list, "Izquierdo"),
            (coord_df_der, copx_der_list, copy_der_list, "Derecho"),
        ], start=1):

            # Rango de ejes por pie
            x_min = coord_this['X'].min() - margen
            x_max = coord_this['X'].max() + margen
            y_min = coord_this['Y'].min() - margen
            y_max = coord_this['Y'].max() + margen

            # Ejes anclados 1:1 para que el círculo en mm no se deforme
            fig.update_yaxes(scaleanchor=('x' if col == 1 else 'x2'), scaleratio=1, row=1, col=col)

            # --- Sensores como círculos en mm (shapes) + etiqueta centrada ---
            # Elegimos refs correctos por subplot
            xref = 'x' if col == 1 else 'x2'
            yref = 'y' if col == 1 else 'y2'

            for i, (sensor, fila) in enumerate(coord_this.iterrows()):
                cx, cy = float(fila["X"]), float(fila["Y"])
                col_hex = sensor_colores[i % len(sensor_colores)]

                # Círculo (en datos mm)
                fig.add_shape(
                    type="circle",
                    xref=xref, yref=yref,
                    x0=cx - r, x1=cx + r,
                    y0=cy - r, y1=cy + r,
                    line=dict(color=col_hex, width=1),
                    fillcolor=col_hex,
                    opacity=0.5,
                    layer="below",  # debajo de la trayectoria del COP
                    row=1, col=col
                )

                # Etiqueta centrada
                fig.add_annotation(
                    x=cx, y=cy, xref=xref, yref=yref,
                    text=sensor, showarrow=False,
                    font=dict(size=10, color="black"),
                    row=1, col=col
                )

            # --- Trayectorias de COP por ciclo (líneas finas) ---
            for i, (x, y) in enumerate(zip(copx_list, copy_list)):
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line=dict(dash='dot', width=1,
                            color=f"rgba({(i*50)%255},{(i*70)%255},{(i*90)%255},0.6)"),
                    name='COP ciclo',
                    showlegend=False
                ), row=1, col=col)

            # --- COP promedio (solo una entrada en la leyenda) ---
            if len(copx_list) > 0:
                x_mean = pd.concat(copx_list, axis=1).mean(axis=1)
                y_mean = pd.concat(copy_list, axis=1).mean(axis=1)
                fig.add_trace(go.Scatter(
                    x=x_mean, y=y_mean,
                    mode='lines',
                    line=dict(color='black', width=3),
                    name='COP promedio',
                    legendgroup='cop-promedio',
                    showlegend=(col == 1) 
                ), row=1, col=col)

            # Ejes y rangos
            fig.update_xaxes(range=[x_min, x_max], title="X (mm)", row=1, col=col)
            fig.update_yaxes(range=[y_min, y_max], title="Y (mm)", row=1, col=col)

        # --- Layout general: que se adapte al contenedor (la card manda) ---
        fig.update_layout(
            title=dict( text="<b>Trayectoria del Centro de Presión (COP)</b>"),
            autosize=True, height=None, width=None,
            margin=dict(t=170, r=40, b=60, l=60),   # ← techo para título + leyenda
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=1.08, yanchor="top",
                xref="paper", yref="paper",
                bgcolor="rgba(255,255,255,0.90)",
                itemsizing="constant",
                font=dict(size=12)
            )
        )




        return fig


    def fig_aporte_por_fases(pie, pasada_df, resultados, coord_df=None):
        """
        Figura 1x4 con el aporte porcentual por sensor (%) para cada fase del apoyo.
        - pie: "Izquierdo" o "Derecho"
        - pasada_df: DataFrame indexado en tiempo con columnas de sensores (Izquierda_S1..S8 o Derecha_S1..S8)
        - resultados: lista/dict con ciclos válidos (usa start_index, end_index, peaks, valleys o fallback)
        - coord_df: DataFrame index S1..S8 con X,Y (mm). Si es None, usa coords por defecto.
        """
        

        FASES = [
            ("loading_response", "<i>loading response</i>"),
            ("midstance",        "<i>midstance</i>"),
            ("terminal_stance",  "<i>terminal stance</i>"),
            ("pre_swing",        "<i>pre-swing</i>"),
        ]

        if coord_df is None:
            coord_df = pd.DataFrame([
                ['S1',  6.4, 182.0],
                ['S2',  0.0, 149.3],
                ['S3', 28.7, 154.0],
                ['S4', 56.8, 149.5],
                ['S5', 54.4,  93.6],
                ['S6',  6.4,  16.0],
                ['S7', 23.4,   0.1],
                ['S8', 41.3,  14.6],
            ], columns=["Sensor", "X", "Y"]).set_index("Sensor")
        coords = coord_df.copy()

        # Reflejo/offset para pie izquierdo o derecho
        if pie.lower().startswith("iz"):
            coords["X"] = -coords["X"]
            coords["X"] += abs(coords["X"].min()) + 10
        else:
            coords["X"] += abs(coords["X"].min()) + 10

        def _extraer_details(resultados):
            # Normaliza a una lista de "details"
            if isinstance(resultados, dict) and "details" in resultados:
                return resultados.get("details", [])
            if isinstance(resultados, list):
                det = []
                for it in resultados:
                    if isinstance(it, dict) and "details" in it:
                        det.extend(it.get("details", []))
                return det
            return []

        def _aportes_promedio_por_fase(df, resultados):
            out = {f: pd.Series(0.0, index=df.columns, dtype=float) for f, _ in FASES}
            cnt = {f: 0 for f, _ in FASES}

            details = _extraer_details(resultados)
            usados = 0

            if details:
                for info in details:
                    if not info.get("is_valid", False):
                        continue
                    t_ini = info.get("start_index", None)
                    t_fin = info.get("end_index", None)
                    peaks = info.get("peaks", []) or []
                    valleys = info.get("valleys", []) or []
                    if t_ini is None or t_fin is None:
                        continue

                    # Eventos: si no hay suficientes, fallback a 30–50–70%
                    if len(peaks) >= 2 and len(valleys) >= 1:
                        p1, p2 = float(peaks[0]), float(peaks[1])
                        v = float(valleys[0])
                    else:
                        dur = float(t_fin) - float(t_ini)
                        if dur <= 0:
                            continue
                        p1 = float(t_ini) + 0.30 * dur
                        v  = float(t_ini) + 0.50 * dur
                        p2 = float(t_ini) + 0.70 * dur

                    segmentos = {
                        'loading_response': (float(t_ini), p1),
                        'midstance':        (p1, v),
                        'terminal_stance':  (v, p2),
                        'pre_swing':        (p2, float(t_fin))
                    }

                    for f in segmentos:
                        a, b = segmentos[f]
                        seg = df.loc[(df.index >= a) & (df.index <= b)]
                        if seg.empty:
                            continue
                        suma = seg.sum(axis=0)
                        total = float(suma.sum())
                        if total <= 0:
                            continue
                        out[f] = out[f].add((suma / total) * 100.0, fill_value=0.0)
                        cnt[f] += 1
                        usados += 1

            if usados == 0 and not df.empty:
                # fallback: toda la pasada
                suma = df.sum(axis=0)
                total = float(suma.sum())
                base = (suma / total) * 100.0 if total > 0 else pd.Series(0.0, index=df.columns)
                for f, _ in FASES:
                    out[f] = base.copy()
                    cnt[f] = 1

            # Promedio final y renombrado a S1..S8
            for f, _ in FASES:
                if cnt[f] > 0:
                    out[f] /= cnt[f]
                out[f].index = [c.split("_")[-1] for c in out[f].index]
                out[f] = out[f].reindex([f"S{i}" for i in range(1, 9)])
            return out

        # Tamaño de marcador en función del %
        def _size_px(p):
            p = max(0.0, float(p))
            return 8.0 + 4.0 * np.sqrt(p)

        # Herramientas para el borde “más oscuro” que el relleno
        def _darker_rgb(rgb_str, factor=0.75):
            # rgb_str: "rgb(r,g,b)" -> "rgb(r',g',b')"
            try:
                nums = rgb_str.strip()[4:-1].split(',')
                r, g, b = [int(float(v)) for v in nums]
                r = max(0, min(255, int(r * factor)))
                g = max(0, min(255, int(g * factor)))
                b = max(0, min(255, int(b * factor)))
                return f"rgb({r},{g},{b})"
            except Exception:
                return "rgba(0,0,0,0.6)"

        # Rangos del lienzo
        x_min, x_max = coords["X"].min() - 15, coords["X"].max() + 15
        y_min, y_max = coords["Y"].min() - 15, coords["Y"].max() + 15

        aportes = _aportes_promedio_por_fase(pasada_df, resultados)

        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=[t for _, t in FASES],
            horizontal_spacing=0.06
        )

        # Colorbar compartido (relleno)
        CMIN, CMAX = 0.0, 40.0
        fig.update_layout(coloraxis=dict(
            colorscale="Turbo",
            cmin=CMIN, cmax=CMAX,
            colorbar=dict(
                title="<b>Aporte (%)</b>",
                orientation="h",
                thickness=10,
                x=0.5, xanchor="center",
                y=-0.12, yanchor="top",
                len=0.30
            )
        ))

        # helper para xref/yref de anotaciones por subplot
        def _ax_ref(j):
            return "" if j == 1 else str(j)

        for j, (f, _title_html) in enumerate(FASES, start=1):
            s = aportes.get(f, pd.Series(0.0, index=[f"S{i}" for i in range(1, 9)]))
            valores = [float(s.get(f"S{i}", 0.0)) for i in range(1, 9)]
            sizes = [_size_px(v) for v in valores]

            # Borde por punto: muestreamos la escala Turbo y la oscurecemos
            # (line.color acepta array por punto)
            norm = [np.clip((v - CMIN) / (CMAX - CMIN), 0, 1) if np.isfinite(v) else 0 for v in valores]
            base_cols = [sample_colorscale("Turbo", [t])[0] for t in norm]
            line_cols = [_darker_rgb(c, 0.70) for c in base_cols]

            # 1) PUNTOS (relleno por coloraxis + borde oscuro por punto)
            fig.add_trace(
                go.Scatter(
                    x=coords.loc[[f"S{i}" for i in range(1, 9)], "X"],
                    y=coords.loc[[f"S{i}" for i in range(1, 9)], "Y"],
                    mode="markers",  # sin 'text': lo hacemos con anotaciones para tener caja blanca y borde negro
                    marker=dict(
                        size=sizes, sizemode="diameter",
                        line=dict(color=line_cols, width=1.5),
                        color=valores, coloraxis="coloraxis",
                        opacity=0.95
                    ),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1, col=j
            )

            # 2) NÚMEROS como anotaciones: fondo BLANCO + borde NEGRO
            xref = f"x{_ax_ref(j)}"
            yref = f"y{_ax_ref(j)}"
            xs = coords.loc[[f"S{i}" for i in range(1, 9)], "X"].values
            ys = coords.loc[[f"S{i}" for i in range(1, 9)], "Y"].values
            for (xk, yk, vk) in zip(xs, ys, valores):
                fig.add_annotation(
                    x=xk, y=yk,
                    xref=xref, yref=yref,
                    text=f"<b>{vk:.1f}%<b>",
                    showarrow=False,
                    xanchor="center", yanchor="middle",
                    font=dict(size=11, color="black")
                )

            # ✔ Grid ON, ejes y ticks OFF
            fig.update_xaxes(
                range=[x_min, x_max],
                showgrid=True, gridcolor="rgba(0,0,0,0.1)",
                showticklabels=False, ticks='',
                showline=False, zeroline=False, mirror=False,
                title_text=None,
                row=1, col=j
            )
            fig.update_yaxes(
                range=[y_min, y_max],
                showgrid=True, gridcolor="rgba(0,0,0,0.1)",
                showticklabels=False, ticks='',
                showline=False, zeroline=False, mirror=False,
                title_text=None,
                row=1, col=j
            )

        fig.update_layout(
            title=dict(text=f"<b>Aporte porcentual por sensor – Pie {pie}</b>", pad=dict(t=0, b=24)),
            template="plotly_white",
            margin=dict(t=80, l=20, r=20, b=120),
            autosize=True, height=None, width=None,
            showlegend=False
        )
        return fig


    

    # === Desempaquetar datos relevantes con protección de estructura ===
    filt_izq = data['filt_izq'][0] if isinstance(data['filt_izq'], list) else data['filt_izq']
    filt_der = data['filt_der'][0] if isinstance(data['filt_der'], list) else data['filt_der']
    results_izq = data['resultados_izq'][0] if isinstance(data['resultados_izq'], list) else data['resultados_izq']
    results_der = data['resultados_der'][0] if isinstance(data['resultados_der'], list) else data['resultados_der']
    indices_apoyo_izq = data['indices_apoyo_izq'][0] if isinstance(data['indices_apoyo_izq'], list) else data['indices_apoyo_izq']
    indices_apoyo_der = data['indices_apoyo_der'][0] if isinstance(data['indices_apoyo_der'], list) else data['indices_apoyo_der']
    copx_izq = data['copx_izq'][0] if isinstance(data['copx_izq'], list) else data['copx_izq']
    copy_izq = data['copy_izq'][0] if isinstance(data['copy_izq'], list) else data['copy_izq']
    copx_der = data['copx_der'][0] if isinstance(data['copx_der'], list) else data['copx_der']
    copy_der = data['copy_der'][0] if isinstance(data['copy_der'], list) else data['copy_der']
    sum_izq = data['sum_izq']
    sum_der = data['sum_der']
    min_indices_izq = data['min_indices_izq']
    min_indices_der = data['min_indices_der']
    
    idx_min_der = (min_indices_der[0]
               if isinstance(min_indices_der, list) and len(min_indices_der) and isinstance(min_indices_der[0], list)
               else min_indices_der)
    idx_min_izq = (min_indices_izq[0]
               if isinstance(min_indices_izq, list) and len(min_indices_izq) and isinstance(min_indices_izq[0], list)
               else min_indices_izq)
    



    # 1. Gráfico de ciclos válidos/inválidos
    fig_ciclos = plot_ciclos_validos_invalidos(
        sum_izq,
        sum_der,
        results_izq,
        results_der,
        idx_min_izq,   # ya reempaquetados
        idx_min_der,   # ya reempaquetados
    ) if (has_valid_der or has_valid_izq) else go.Figure()


    # 2. Gráficos por fases (izquierdo y derecho)
    fig_fases_izq = plot_vgrf_por_fases(
        [filt_izq],
        [results_izq],
        [indices_apoyo_izq[0]],
        peso_kg=BW,
        lado="I"
    )[0] if has_valid_izq else go.Figure()

    fig_fases_der = plot_vgrf_por_fases(
        [filt_der],
        [results_der],
        [indices_apoyo_der[0]],
        peso_kg=BW,
        lado="R"
    )[0] if has_valid_der else go.Figure()
    

    fig_fases = combinar_fases_en_uno(fig_fases_izq, fig_fases_der)


    # 3. Gráfico de COP (Centro de Presión)
    fig_cop = plot_cop_pie_izq_der(
        copx_izq,
        copy_izq,
        copx_der,
        copy_der,
        pasada_idx=0
    ) if (has_valid_der or has_valid_izq) else go.Figure()

    # 4. Gráficos de aporte por fase (izquierdo y derecho)
    fig_aporte_izq = fig_aporte_por_fases("Izquierdo", data["filt_izq"], data["resultados_izq"]) if has_valid_izq else go.Figure()
    fig_aporte_der = fig_aporte_por_fases("Derecho",   data["filt_der"], data["resultados_der"]) if has_valid_der else go.Figure()

    fig_cop.update_layout(margin=dict(t=60, r=30, b=80, l=60))
    fig_aporte_izq.update_layout(margin=dict(t=50, r=30, b=40, l=60))
    fig_aporte_der.update_layout(margin=dict(t=50, r=30, b=40, l=60))


    return (
        fig_ciclos,
        fig_fases,
        fig_cop,           
        fig_aporte_izq,
        fig_aporte_der
    )


