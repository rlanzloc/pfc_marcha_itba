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

def generar_figuras_cinetico(data):
    print("🧪 Entrando a generar_figuras_cinetico()")
    BW = 52  # peso corporal fijo para normalización
    
    # Mostrar datos de diagnóstico
    print("📊 Diagnóstico de datos recibidos:")
    print(f"Suma derecha: {data['sum_der']}")
    print(f"Suma izquierda: {data['sum_izq']}")
    
    # Asegurar que los datos sean listas aunque haya un solo elemento
    def ensure_list(x):
        return x if isinstance(x, list) else [x]
    
    # Verificar si hay datos válidos
    has_valid_der = data['resultados_der'][0]['valid'] if 'resultados_der' in data else False
    has_valid_izq = data['resultados_izq'][0]['valid'] if 'resultados_izq' in data else False
    
    if not has_valid_der and not has_valid_izq:
        print("⚠️ No hay ciclos válidos para graficar")
        fig_empty = go.Figure()
        fig_empty.update_layout(title="No se detectaron ciclos válidos")
        return [fig_empty] * 6

    def plot_ciclos_validos_invalidos(sum_izq, sum_der, res_izq, res_der, idx_min_izq, idx_min_der):
        print("🧪 Entrando a plot_ciclos_validos_invalidos()")
        fig = go.Figure()

        # 🛡️ Validación robusta para idx_min_der
        if idx_min_der and isinstance(idx_min_der, list):
            if all(isinstance(x, (int, float)) for x in idx_min_der):
                if len(idx_min_der) % 2 != 0:
                    print(f"❌ idx_min_der tiene cantidad impar de índices: {len(idx_min_der)}. Se saltea.")
                    return fig

        for ciclo in res_der['details']:
            j = ciclo['cycle_num']
            if j+1 >= len(idx_min_der): continue
            i0 = idx_min_der[j]
            i1 = idx_min_der[j+1]
            tramo = sum_der.loc[i0:i1, 0] / BW
            color = 'green' if ciclo['is_valid'] else 'red'
            fig.add_trace(go.Scatter(
                x=tramo.index, y=tramo.values,
                mode='lines',
                line=dict(color=color, width=2),
                name='Derecho',
                opacity=1.0 if ciclo['is_valid'] else 0.5,
                showlegend=False
            ))

        # 🛡️ Validación robusta para idx_min_izq
        if idx_min_izq and isinstance(idx_min_izq, list):
            if all(isinstance(x, (int, float)) for x in idx_min_izq):
                if len(idx_min_izq) % 2 != 0:
                    print(f"❌ idx_min_izq tiene cantidad impar de índices: {len(idx_min_izq)}. Se saltea.")
                    return fig

        for ciclo in res_izq['details']:
            j = ciclo['cycle_num']
            if j+1 >= len(idx_min_izq): continue
            i0 = idx_min_izq[j]
            i1 = idx_min_izq[j+1]
            tramo = sum_izq.loc[i0:i1, 0] / BW
            color = 'green' if ciclo['is_valid'] else 'red'
            fig.add_trace(go.Scatter(
                x=tramo.index, y=tramo.values,
                mode='lines',
                line=dict(color=color, width=2),
                name='Izquierdo',
                opacity=0.6,
                showlegend=False
            ))

        fig.update_layout(
            title="Ciclos Válidos / Inválidos",
            xaxis_title="Tiempo (s)",
            yaxis_title="vGRF (%BW)",
            template="plotly_white"
        )
        return fig





    def plot_vgrf_por_fases(df_sensores_list, resultados, indices_ciclos,
                                    peso_kg=52, lado="R"):
        print("🧪 Entrando a plot_vgrf_por_fases()")
        """
        Genera una figura interactiva en Plotly con vGRF normalizado y fases de apoyo.
        """

        sufijo = "Derecha" if lado.upper() == "R" else "Izquierda"
        colores_fases = {
            'loading_response': '#9467bd',
            'midstance': '#d62728',
            'terminal_stance': '#e377c2',
            'pre_swing': '#2ca02c'
        }

        figuras = []

        print("🧩 Len df_sensores_list:", len(df_sensores_list))
        print("🧩 Len resultados:", len(resultados))
        print("🧩 Len indices_ciclos:", len(indices_ciclos))

        for pasada_idx, (df_sensores, resultado_paso, indices_paso) in enumerate(zip(df_sensores_list, resultados, indices_ciclos)):
            print(f"🔍 resultado_paso keys: {list(resultado_paso.keys())}")
            print(f"🔍 resultado_paso completo:\n{resultado_paso}")

            fig = go.Figure()
            matriz_ciclos = []

            fases_usadas = {fase: False for fase in colores_fases.keys()}
            pico_label_usado = False
            valle_label_usado = False

            print(f"📊 Pasada {pasada_idx+1} — ciclos detectados: {len(resultado_paso['details'])}")
            for i, ciclo in enumerate(resultado_paso['details']):
                estado = "✅ válido" if ciclo["is_valid"] else "❌ inválido"
                print(f"   • Ciclo {i}: {estado}, picos: {len(ciclo.get('peaks', []))}, valles: {len(ciclo.get('valleys', []))}")

            # ✅ Validación robusta de formato de índices
            if indices_paso and isinstance(indices_paso, list):
                if all(isinstance(x, (int, float)) for x in indices_paso):
                    if len(indices_paso) % 2 == 0:
                        print("⚠️ indices_paso vino como lista de números, reempaquetando en tuplas...")
                        indices_paso = list(zip(indices_paso[::2], indices_paso[1::2]))
                    else:
                        print(f"❌ Cantidad impar de índices. Revisión necesaria: {indices_paso}")
                        continue
                elif all(isinstance(x, (list, tuple)) and len(x) == 2 for x in indices_paso):
                    pass  # ya está bien
                else:
                    print(f"❌ indices_paso tiene formato no esperado: {indices_paso}")
                    continue

            for ciclo_info, indices_ciclo in zip(resultado_paso['details'], indices_paso):
                if not isinstance(indices_ciclo, (list, tuple)) or len(indices_ciclo) != 2:
                    print(f"⚠️ Ciclo inválido o mal formado: {indices_ciclo}")
                    continue

                print(f"🔎 tipo de indices_ciclo: {type(indices_ciclo)}, valor: {indices_ciclo}")
                if not ciclo_info['is_valid']:
                    continue

                inicio, fin = indices_ciclo[0], indices_ciclo[1]
                datos_ciclo = df_sensores.loc[inicio:fin].copy()
                suma_total = datos_ciclo.sum(axis=1).to_frame().loc[:, 0] / peso_kg
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
                    print(f"⛏️ buscando get_loc({start}) en datos_ciclo.index: {datos_ciclo.index}")
                    idx_start = datos_ciclo.index.get_loc(start)
                    idx_end = datos_ciclo.index.get_loc(end)
                    x_fase = x_original[idx_start:idx_end+1]
                    y_fase = suma_total.loc[idx_start:idx_end+1].values
                    fig.add_trace(go.Scatter(
                        x=x_fase, y=y_fase,
                        mode='lines',
                        line=dict(color=colores_fases[fase], width=2, dash='dash'),
                        name=fase.replace("_", " ").capitalize() if not fases_usadas[fase] else None,
                        showlegend=not fases_usadas[fase]
                    ))
                    fases_usadas[fase] = True

                for p in [p1, p2]:
                    idx_p = datos_ciclo.index.get_loc(p)
                    fig.add_trace(go.Scatter(
                        x=[x_original[idx_p]],
                        y=[suma_total.iloc[idx_p]],
                        mode='markers',
                        marker=dict(symbol='x', color='#1f77b4', size=8),
                        name='Pico' if not pico_label_usado else None,
                        showlegend=not pico_label_usado
                    ))
                    pico_label_usado = True

                idx_v = datos_ciclo.index.get_loc(v)
                fig.add_trace(go.Scatter(
                    x=[x_original[idx_v]],
                    y=[suma_total.iloc[idx_v]],
                    mode='markers',
                    marker=dict(symbol='circle', color='#ff7f0e', size=8),
                    name='Valle' if not valle_label_usado else None,
                    showlegend=not valle_label_usado
                ))
                valle_label_usado = True


            if matriz_ciclos:
                matriz = np.vstack(matriz_ciclos)
                promedio = matriz.mean(axis=0)
                std = matriz.std(axis=0)
                fig.add_trace(go.Scatter(
                    x=x_interp, y=promedio,
                    mode='lines', name='Promedio',
                    line=dict(color='black', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=x_interp, y=promedio + std,
                    mode='lines', name='+1 SD',
                    line=dict(color='gray', dash='dot'),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=x_interp, y=promedio - std,
                    mode='lines', name='-1 SD',
                    line=dict(color='gray', dash='dot'),
                    fill='tonexty',
                    showlegend=False
                ))

            fig.update_layout(
                title=f"vGRF Normalizado - Pie {sufijo} - Pasada {pasada_idx+1}",
                xaxis_title="% del ciclo de apoyo",
                yaxis_title="% BW",
                template="plotly_white"
            )
            figuras.append(fig)

        if not figuras:
            print("⚠️ No se generaron figuras en plot_vgrf_por_fases(). Posibles causas:")
            print("   ↪️ No hay ciclos válidos o no se detectaron picos/valle en los datos.")
            figuras = [go.Figure()]  # Evita el IndexError

        return figuras


    def plot_cop_pie_izq_der(copx_izq_list, copy_izq_list, copx_der_list, copy_der_list, pasada_idx=0):
        print("🧪 Entrando a plot_cop_pie_izq_der()")
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

        # Preparar coordenadas reflejadas y desplazadas
        coord_df_izq = coord_df.copy()
        coord_df_izq["X"] = -coord_df_izq["X"]
        offset_izq = abs(coord_df_izq["X"].min()) + 10
        offset_der = abs(coord_df["X"].min()) + 10
        coord_df_izq["X"] += offset_izq
        coord_df_der = coord_df.copy()
        coord_df_der["X"] += offset_der

        # Parámetros comunes
        margen = 20
        sensor_colores = ['blue', 'yellow', 'grey', 'pink', 'green', 'orange', 'purple', 'red']

        # Crear figura con subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Pie Izquierdo", "Pie Derecho"])

        for col, (coord_df, copx_list, copy_list, pie_nombre) in enumerate([
            (coord_df_izq, copx_izq_list, copy_izq_list, "Izquierdo"),
            (coord_df_der, copx_der_list, copy_der_list, "Derecho")
        ], start=1):

            x_min = coord_df['X'].min() - margen
            x_max = coord_df['X'].max() + margen
            y_min = coord_df['Y'].min() - margen
            y_max = coord_df['Y'].max() + margen

            # Sensores
            for i, (sensor, fila) in enumerate(coord_df.iterrows()):
                fig.add_trace(go.Scatter(
                    x=[fila["X"]], y=[fila["Y"]],
                    mode="markers+text",
                    marker=dict(size=14, color=sensor_colores[i], opacity=0.5),
                    text=[sensor], textposition="middle center",
                    showlegend=False
                ), row=1, col=col)

            # Trayectorias COP
            for i, (x, y) in enumerate(zip(copx_list, copy_list)):
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line=dict(dash='dot', width=1, color=f"rgba({(i*50)%255},{(i*70)%255},{(i*90)%255},0.6)"),
                    name='COP ciclo', showlegend=False
                ), row=1, col=col)

            # COP promedio
            if copx_list:
                x_mean = pd.concat(copx_list, axis=1).mean(axis=1)
                y_mean = pd.concat(copy_list, axis=1).mean(axis=1)
                fig.add_trace(go.Scatter(
                    x=x_mean, y=y_mean,
                    mode='lines',
                    line=dict(color='black', width=3),
                    name='COP promedio'
                ), row=1, col=col)

            fig.update_xaxes(range=[x_min, x_max], title="X (mm)", row=1, col=col)
            fig.update_yaxes(range=[y_min, y_max], title="Y (mm)", row=1, col=col)

        fig.update_layout(
            title=f"Trayectoria del Centro de Presión - Pasada {pasada_idx + 1}",
            height=600,
            width=1200,
            template="plotly_white",
            showlegend=True
        )

        return fig
    
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
    

    def plot_aporte_2D(pasada_df, resultados, coord_sensores, pie="Derecho"):
        fases = ["loading_response", "midstance", "terminal_stance", "pre_swing"]
        acumulador = {fase: [] for fase in fases}

        for ciclo in resultados:
            if not ciclo["is_valid"]:
                continue

            inicio = ciclo["start_index"]
            fin = ciclo["end_index"]
            peaks = ciclo["peaks"]
            valleys = ciclo["valleys"]

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

            for fase in fases:
                i0, i1 = segmentos[fase]
                tramo = pasada_df.loc[i0:i1].mean(axis=0)

                if not isinstance(tramo, pd.Series):
                    tramo = pd.Series(tramo, index=pasada_df.columns)

                acumulador[fase].append(tramo)

        # Calcular promedio por fase
        promedio_fases = {}
        for fase, valores in acumulador.items():
            if not valores:
                promedio_fases[fase] = pd.Series([0] * 8, index=pasada_df.columns)
            else:
                try:
                    promedio_fases[fase] = pd.concat(valores, axis=1).mean(axis=1)
                except Exception as e:
                    print(f"❌ Error al concatenar fase {fase}: {e}")
                    promedio_fases[fase] = pd.Series([0] * 8, index=pasada_df.columns)

        # Crear figura Plotly
        fig = go.Figure()
        fase_labels = ["LR", "MS", "TS", "PS"]
        colormap = plt.get_cmap('jet')

        for i, fase in enumerate(fases):
            valores = promedio_fases[fase]
            max_valor = valores.max() if valores.max() != 0 else 1
            norm_valores = valores / max_valor

            for j, sensor in enumerate(pasada_df.columns):
                nombre_base = sensor.split("_")[-1]  # extrae S1, S2, ...
                if nombre_base not in coord_sensores.index:
                    print(f"⚠️ {nombre_base} no está en coord_sensores. Saltando.")
                    continue
                x, y = coord_sensores.loc[nombre_base]

                if pie.lower().startswith("izq"):
                    x = -x
                offset = i * 1.2
                fig.add_trace(go.Scatter(
                    x=[x + offset], y=[y],
                    mode="markers+text",
                    marker=dict(
                        size=50,
                        color=norm_valores[j],
                        colorscale='Jet',
                        cmin=0, cmax=1,
                        line=dict(color='black', width=1)
                    ),
                    text=[f"{valores[j]:.0f}%"],
                    textposition="middle center",
                    showlegend=False
                ))

            fig.add_annotation(
                x=offset, y=1.1,
                text=fase_labels[i],
                showarrow=False,
                font=dict(size=14, color="black")
            )

        fig.update_layout(
            title=f"Aporte porcentual por sensor – Pie {pie}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
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


    # 1. Gráfico de ciclos válidos/inválidos
    fig_ciclos = plot_ciclos_validos_invalidos(
        sum_izq,
        sum_der,
        results_izq,
        results_der,
        min_indices_izq,
        min_indices_der
    ) if (has_valid_der or has_valid_izq) else go.Figure()

    # 2. Gráficos por fases (izquierdo y derecho)
    fig_fases_izq = plot_vgrf_por_fases(
        [filt_izq],
        [results_izq],
        [indices_apoyo_izq],
        peso_kg=BW,
        lado="I"
    )[0] if has_valid_izq else go.Figure()

    fig_fases_der = plot_vgrf_por_fases(
        [filt_der],
        [results_der],
        [indices_apoyo_der],
        peso_kg=BW,
        lado="R"
    )[0] if has_valid_der else go.Figure()
    

    print("🧪 fig_ciclos tiene", len(fig_ciclos.data), "trazas")
    print("🧪 fig_fases_der tiene", len(fig_fases_der.data), "trazas")

    # 3. Gráfico de COP (Centro de Presión)
    fig_cop = plot_cop_pie_izq_der(
        copx_izq,
        copy_izq,
        copx_der,
        copy_der,
        pasada_idx=0
    ) if (has_valid_der or has_valid_izq) else go.Figure()

    # 4. Gráficos de aporte por fase (izquierdo y derecho)
    fig_aporte_izq = plot_aporte_2D(
        filt_izq,
        results_izq['details'],
        coord_sensores,
        pie="Izquierdo"
    ) if has_valid_izq else go.Figure()

    fig_aporte_der = plot_aporte_2D(
        filt_der,
        results_der['details'],
        coord_sensores,
        pie="Derecho"
    ) if has_valid_der else go.Figure()

    return (
        fig_ciclos,
        fig_fases_izq,  
        fig_fases_der,
        fig_cop,           
        fig_aporte_izq,
        fig_aporte_der
    )


