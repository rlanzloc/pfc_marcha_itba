import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import pandas as pd
import plotly.graph_objects as go
from analisis_marcha import procesar_archivo_c3d  # Importa la función de análisis
import tempfile
import os
import numpy as np

dash.register_page(__name__, path="/analysis", name="Análisis de Marcha")

# Rangos de los ejes Y para cada articulación
RANGOS_Y = {
    "tobillo": {"Z": (-30, 30), "Y": (-30, 30), "X": (-30, 30)},
    "rodilla": {"Z": (-20, 70), "Y": (-30, 30), "X": (-30, 30)},
    "cadera": {"Z": (-20, 60), "Y": (-30, 30), "X": (-30, 30)},
}

layout = dbc.Container([
    dcc.Store(id='stored-data'),
    dbc.Row([
        dbc.Col(html.H1("Análisis de Marcha", className="text-center mb-4"))
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-c3d',
                children=html.Div([
                    'Arrastra o selecciona un archivo C3D'
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False  # Permitir solo un archivo
            ),
            html.Div(id='output-c3d-upload'),
        ])
    ]),
    dbc.Row(id='controls-row', style={'display': 'none'}, children=[
        dbc.Col([
            dbc.Label("Selecciona las articulaciones:", className="mb-2",  style={"fontSize": "24px", "fontWeight": "bold"}),  # Título
            dbc.Checklist(
                id='articulacion-checklist',
                options=[
                    {'label': ' Tobillo', 'value': 'tobillo'},  # Espacio antes del texto
                    {'label': ' Rodilla', 'value': 'rodilla'},
                    {'label': ' Cadera', 'value': 'cadera'}
                ],
                value=['tobillo'],  # Valor por defecto
                inline=True,  # Muestra los elementos en línea
                style={'font-size': '22px', 'margin-right': '20px'}  # Tamaño de fuente y margen
            )
        ], width=12),  # Ocupa todo el ancho
        dbc.Col([
            dcc.Dropdown(
                id='lado-dropdown',
                options=[
                    {'label': 'Derecha', 'value': 'Derecha'},
                    {'label': 'Izquierda', 'value': 'Izquierda'},
                    {'label': 'Ambos', 'value': 'Ambos'}
                ],
                value='Derecha',
                clearable=False
            )
        ], width=6)
    ]),
    dbc.Row(id='graphs-row'),  # Fila para los gráficos
    dbc.Row(id='parametros-espaciotemporales-row')  # Nueva fila para los parámetros espaciotemporales
])


@callback(
    Output('parametros-espaciotemporales-row', 'children'),
    Input('stored-data', 'data')
)
def update_parametros_espaciotemporales(stored_data):
    if stored_data is None or 'parametros_espaciotemporales' not in stored_data:
        return []
    
    parametros = stored_data['parametros_espaciotemporales']
    
    # Determinar el rango máximo para la escala común
    max_duracion = max(parametros['duracion_ciclo_derecho'], parametros['duracion_ciclo_izquierdo'], parametros['tiempo_paso'])
    max_longitud = max(parametros['longitud_ciclo_derecho'], parametros['longitud_ciclo_izquierdo'], parametros['longitud_paso'])
    
    # Función para crear gráficos de barras con escala consistente
    def crear_grafico_barras(valores, titulo, colores, max_valor):
        fig = go.Figure()
        
        # Para pasos (una sola barra)
        if len(valores) == 1:
            fig.add_trace(go.Bar(
                x=['Paso'],
                y=valores,
                marker_color=['#9E9E9E'],  # Gris
                text=[f"{valores[0]:.2f}"],
                textposition='auto'
            ))
        # Para ciclos (dos barras)
        else:
            fig.add_trace(go.Bar(
                x=['Derecho', 'Izquierdo'],
                y=valores,
                marker_color=colores,
                text=[f"{v:.2f}" for v in valores],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=titulo,
            margin=dict(l=20, r=20, t=40, b=20),
            height=150,
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=False, 
                showticklabels=False,
                range=[0, max_valor * 1.1]  # 10% más para espacio
            )
        )
        return dcc.Graph(figure=fig, config={'staticPlot': True}, style={'height': '150px'})
    
    # Gráficos para ciclos (dos barras)
    duracion_fig = crear_grafico_barras(
        [parametros['duracion_ciclo_derecho'], parametros['duracion_ciclo_izquierdo']],
        "Duración (s)",
        ['#FF5252', '#4285F4'],  # Rojo y azul
        max_duracion
    )
    
    longitud_fig = crear_grafico_barras(
        [parametros['longitud_ciclo_derecho'], parametros['longitud_ciclo_izquierdo']],
        "Longitud (m)",
        ['#FF5252', '#4285F4'],  # Rojo y azul
        max_longitud
    )
    
    # Gráficos para pasos (una barra gris)
    paso_duracion_fig = crear_grafico_barras(
        [parametros['tiempo_paso']],
        "Duración paso (s)",
        None,
        max_duracion
    )
    
    paso_longitud_fig = crear_grafico_barras(
        [parametros['longitud_paso']],
        "Longitud paso (m)",
        None,
        max_longitud
    )
    
    return dbc.Row([
        # Tarjeta para ciclos
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("Ciclos de Marcha", className="text-center mb-0")),
            dbc.CardBody([
                html.H5(f"Número: {parametros['num_ciclos_derecho']}D / {parametros['num_ciclos_izquierdo']}I", 
                        className="text-center mb-3"),
                duracion_fig,
                longitud_fig
            ])
        ], className="h-100"), width=4),
        
        # Tarjeta para pasos
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("Pasos", className="text-center mb-0")),
            dbc.CardBody([
                html.H5(f"Número: {parametros['num_pasos']}", className="text-center mb-3"),
                paso_duracion_fig,
                paso_longitud_fig
            ])
        ], className="h-100"), width=4),
        
        # Tarjeta para velocidad y cadencia
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("Otros Parámetros", className="text-center mb-0")),
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.P("Velocidad:", className="mb-1"),
                        html.P(f"{parametros['velocidad']:.2f} km/h", className="h4 text-primary")
                    ], className="text-center mb-3"),
                    html.Div([
                        html.P("Cadencia:", className="mb-1"),
                        html.P(f"{parametros['cadencia']:.2f} pasos/min", className="h4 text-primary")
                    ], className="text-center")
                ])
            ])
        ], className="h-100"), width=4)
    ], className="mb-4")


def final_plot_plotly(curva_derecha=None, curva_izquierda=None, posibilidad="Derecha", rango_z=(-20, 60), rango_y=(-30, 30), rango_x=(-30, 30), articulacion=""):
    """
    Grafica las curvas de una articulación según la posibilidad especificada usando Plotly.
    """
    # Verificar la posibilidad y asignar colores
    if posibilidad == "Derecha":
        colors = ["red"]
        labels = ["Derecha"]
        curvas = [curva_derecha]  # Solo curva derecha
    elif posibilidad == "Izquierda":
        colors = ["blue"]
        labels = ["Izquierda"]
        curvas = [curva_izquierda]  # Solo curva izquierda
    elif posibilidad == "Ambos":
        colors = ["red", "blue"]
        labels = ["Derecha", "Izquierda"]
        curvas = [curva_derecha, curva_izquierda]  # Ambas curvas
    else:
        raise ValueError("La posibilidad debe ser 'Derecha', 'Izquierda' o 'Ambos'.")
    light_color = {
        'red': 'rgba(255, 0, 0, 0.2)',  # Rojo claro (transparente)
        'blue': 'rgba(0, 0, 255, 0.2)',  # Azul claro (transparente)
    }

    # Crear los gráficos para cada eje
    fig_z = go.Figure()
    fig_y = go.Figure()
    fig_x = go.Figure()

    # Procesar y graficar según la posibilidad
    for curva, color, label in zip(curvas, colors, labels):
        if curva is not None:
            # Convertir las columnas Z, Y, X en arrays de NumPy
            Z_curves = np.array(curva["Z"].tolist())
            Y_curves = np.array(curva["Y"].tolist())
            X_curves = np.array(curva["X"].tolist())

            # Calcular promedios y desviaciones estándar
            average_Z = np.mean(Z_curves, axis=0)
            std_Z = np.std(Z_curves, axis=0)

            average_Y = np.mean(Y_curves, axis=0)
            std_Y = np.std(Y_curves, axis=0)

            average_X = np.mean(X_curves, axis=0)
            std_X = np.std(X_curves, axis=0)

            # Crear el tiempo normalizado (0% a 100%)
            fixed_time = np.linspace(0, 100, num=len(average_Z))

            # Graficar para Z
            fig_z.add_trace(go.Scatter(
                x=fixed_time,
                y=average_Z,
                line=dict(color=color),
                name=f"{label}: Promedio"
            ))
            fig_z.add_trace(go.Scatter(
                x=fixed_time,
                y=average_Z + std_Z,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig_z.add_trace(go.Scatter(
                x=fixed_time,
                y=average_Z - std_Z,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=light_color[color],
                name=f"{label}: Desviación estándar"
            ))

            # Graficar para Y
            fig_y.add_trace(go.Scatter(
                x=fixed_time,
                y=average_Y,
                line=dict(color=color),
                name=f"{label}: Promedio"
            ))
            fig_y.add_trace(go.Scatter(
                x=fixed_time,
                y=average_Y + std_Y,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig_y.add_trace(go.Scatter(
                x=fixed_time,
                y=average_Y - std_Y,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=light_color[color],
                name=f"{label}: Desviación estándar"
            ))

            # Graficar para X
            fig_x.add_trace(go.Scatter(
                x=fixed_time,
                y=average_X,
                line=dict(color=color),
                name=f"{label}: Promedio"
            ))
            fig_x.add_trace(go.Scatter(
                x=fixed_time,
                y=average_X + std_X,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig_x.add_trace(go.Scatter(
                x=fixed_time,
                y=average_X - std_X,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=light_color[color],
                name=f"{label}: Desviación estándar"
            ))

    # Configuración de los gráficos
    fig_z.update_layout(
        title=f"Articulación de {articulacion.capitalize()} - Ángulos de Z",
        xaxis_title="Porcentaje del Ciclo de Marcha (%)",
        yaxis_title="Ángulo en Z (°)",
        yaxis_range=rango_z,
        legend=dict(
            orientation="h",  # Leyenda horizontal
            yanchor="top",  # Anclada en la parte inferior
            y=-0.3,  # Posición debajo del gráfico
            xanchor="center",  # Centrada horizontalmente
            x=0.5
        )
    )
    fig_y.update_layout(
        title=f"Articulación de {articulacion.capitalize()} - Ángulos de Y",
        xaxis_title="Porcentaje del Ciclo de Marcha (%)",
        yaxis_title="Ángulo en Y (°)",
        yaxis_range=rango_y,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    fig_x.update_layout(
        title=f"Articulación de {articulacion.capitalize()} - Ángulos de X",
        xaxis_title="Porcentaje del Ciclo de Marcha (%)",
        yaxis_title="Ángulo en X (°)",
        yaxis_range=rango_x,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )

    return fig_z, fig_y, fig_x

@callback(
    [Output('output-c3d-upload', 'children'),
     Output('stored-data', 'data'),
     Output('controls-row', 'style')],  # Mostrar/ocultar controles
    Input('upload-c3d', 'contents'),
    State('upload-c3d', 'filename')
)

def update_output(contents, filename):
    if contents is not None:
        results = {
            'tobillo_derecho': [],
            'tobillo_izquierdo': [],
            'rodilla_derecha': [],
            'rodilla_izquierda': [],
            'cadera_derecha': [],
            'cadera_izquierda': [],
            'parametros_espaciotemporales': {}  # Nuevo campo para los parámetros espaciotemporales
        }

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Crear un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as temp_file:
                temp_file.write(decoded)
                temp_filepath = temp_file.name  # Obtener la ruta

            # Procesar el archivo C3D con la ruta
            curvas_tobillo_derecho, curvas_tobillo_izquierdo, curvas_rodilla_derecha, curvas_rodilla_izquierda, curvas_cadera_derecha, curvas_cadera_izquierda, parametros_espaciotemporales = procesar_archivo_c3d(temp_filepath)

            # Eliminar el archivo temporal
            os.remove(temp_filepath)

            # Agregar los resultados al diccionario
            results['tobillo_derecho'].extend(curvas_tobillo_derecho.to_dict('records'))
            results['tobillo_izquierdo'].extend(curvas_tobillo_izquierdo.to_dict('records'))
            results['rodilla_derecha'].extend(curvas_rodilla_derecha.to_dict('records'))
            results['rodilla_izquierda'].extend(curvas_rodilla_izquierda.to_dict('records'))
            results['cadera_derecha'].extend(curvas_cadera_derecha.to_dict('records'))
            results['cadera_izquierda'].extend(curvas_cadera_izquierda.to_dict('records'))
            results['parametros_espaciotemporales'] = parametros_espaciotemporales

        except Exception as e:
            return html.Div(f"Error al procesar el archivo {filename}: {str(e)}"), None, {'display': 'none'}

        return html.Div(f"Archivo {filename} cargado correctamente."), results, {'display': 'block'}
    return html.Div(), None, {'display': 'none'}

@callback(
    Output('graphs-row', 'children'),
    [Input('articulacion-checklist', 'value'),
     Input('lado-dropdown', 'value'),
     Input('stored-data', 'data')]
)


def update_graphs(articulaciones, lado, stored_data):
    if stored_data is None:
        return []

    # Convertir los datos de vuelta a DataFrames
    curvas_tobillo_derecho = pd.DataFrame(stored_data['tobillo_derecho'])
    curvas_tobillo_izquierdo = pd.DataFrame(stored_data['tobillo_izquierdo'])
    curvas_rodilla_derecha = pd.DataFrame(stored_data['rodilla_derecha'])
    curvas_rodilla_izquierda = pd.DataFrame(stored_data['rodilla_izquierda'])
    curvas_cadera_derecha = pd.DataFrame(stored_data['cadera_derecha'])
    curvas_cadera_izquierda = pd.DataFrame(stored_data['cadera_izquierda'])

    graphs = []
    for articulacion in articulaciones:
        if articulacion == "tobillo":
            curva_derecha = curvas_tobillo_derecho
            curva_izquierda = curvas_tobillo_izquierdo
            rango_z, rango_y, rango_x = RANGOS_Y["tobillo"].values()
        elif articulacion == "rodilla":
            curva_derecha = curvas_rodilla_derecha
            curva_izquierda = curvas_rodilla_izquierda
            rango_z, rango_y, rango_x = RANGOS_Y["rodilla"].values()
        elif articulacion == "cadera":
            curva_derecha = curvas_cadera_derecha
            curva_izquierda = curvas_cadera_izquierda
            rango_z, rango_y, rango_x = RANGOS_Y["cadera"].values()

        # Generar los gráficos usando Plotly
        fig_z, fig_y, fig_x = final_plot_plotly(curva_derecha, curva_izquierda, posibilidad=lado, rango_z=rango_z, rango_y=rango_y, rango_x=rango_x, articulacion=articulacion)

        # Agregar los gráficos a la fila
        graphs.append(dbc.Col(dcc.Graph(figure=fig_z), width=4))
        graphs.append(dbc.Col(dcc.Graph(figure=fig_y), width=4))
        graphs.append(dbc.Col(dcc.Graph(figure=fig_x), width=4))

    return graphs