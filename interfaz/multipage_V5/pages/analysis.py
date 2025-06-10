import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analisis_marcha import procesar_archivo_c3d  # Importa la función de análisis
import tempfile
import os
import numpy as np


dash.register_page(__name__, path="/analysis", name="Análisis Cinemático", icon="file-earmark-text")
#dash.register_page(__name__, path="/analysis", name="Análisis de Marcha")

# Rangos de los ejes Y para cada articulación
RANGOS_Y = {
    "tobillo": {"Z": (-30, 30), "Y": (-30, 30), "X": (-30, 30)},
    "rodilla": {"Z": (-20, 70), "Y": (-30, 30), "X": (-30, 30)},
    "cadera": {"Z": (-20, 60), "Y": (-30, 30), "X": (-30, 30)},
}

layout = dbc.Container([
    dcc.Store(id='stored-data'),
    dcc.Store(id='session-stored-data', storage_type='session'), 
    dbc.Row([
        dbc.Col(html.H1("Análisis Cinemático", className="text-center mb-4", style={
            'color': '#2c3e50',
            'fontWeight': '600',
            'marginTop': '10px'
        }))
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-c3d',
                children=html.Div([
                    html.I(className="bi bi-cloud-arrow-up me-2"),
                    "Arrastra o selecciona un archivo C3D"
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0',
                    'cursor': 'pointer',
                    'backgroundColor': 'rgba(66, 139, 202, 0.1)',
                    'borderColor': '#428bca',
                    'color': '#428bca',
                    'transition': 'all 0.3s'
                },
                multiple=False
            ),
            html.Div(id='output-c3d-upload', className="text-center mb-3", style={
                'color': '#28a745',
                'fontWeight': '500'
            }),
        ], width=12)
    ], className="mb-4"),
    # Contenedor para las tabs mejorado
    html.Div(id='tabs-container', style={'display': 'none'})
], fluid=True, style={'padding': '20px'})

# Callback para mostrar/ocultar las tabs según si hay datos
@callback(
    Output('tabs-container', 'children'),
    Output('tabs-container', 'style'),
    Input('session-stored-data', 'data') 
)

def show_hide_tabs(stored_data):
    if stored_data is None:
        return None, {'display': 'none'}
    
    tabs = dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(
            label='Ángulos Articulares',
            value='tab-1',
            style={
                'borderTop': '1px solid #d6d6d6',
                'borderLeft': '1px solid #d6d6d6',
                'borderRight': '1px solid #d6d6d6',
                'borderBottom': 'none',
                'padding': '12px',
                'fontSize': '16px',
                'fontWeight': 'bold',
                'backgroundColor': 'rgba(240, 240, 240, 0.5)',
                'color': '#555'
            },
            selected_style={
                'borderTop': '2px solid #428bca',
                'borderLeft': '1px solid #d6d6d6',
                'borderRight': '1px solid #d6d6d6',
                'borderBottom': 'none',
                'padding': '12px',
                'fontSize': '16px',
                'fontWeight': 'bold',
                'backgroundColor': 'white',
                'color': '#2c3e50'
            },
           children=[
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            dcc.RadioItems(
                                id='lado-dropdown',
                                options=[
                                    {'label': ' Derecho', 'value': 'Derecha'},
                                    {'label': ' Izquierdo', 'value': 'Izquierda'},
                                    {'label': ' Ambos', 'value': 'Ambos'}
                                ],
                                value='Ambos',
                                inline=True,
                                inputStyle={'margin-right': '5px'},
                                labelStyle={
                                    'display': 'inline-block',
                                    'margin-right': '15px',
                                    'cursor': 'pointer'
                                },
                                style={
                                    'fontSize': '16px',
                                    'padding': '10px 0'
                                }
                            ),
                            style={'margin-bottom': '20px'}
                        )
                    ], width=12)
                ], className="mb-4"),
                dbc.Row(id='graphs-container')
            ]
        ),
        dcc.Tab(
            label='Parámetros Espaciotemporales',
            value='tab-2',
            style={
                'borderTop': '1px solid #d6d6d6',
                'borderLeft': '1px solid #d6d6d6',
                'borderRight': '1px solid #d6d6d6',
                'borderBottom': 'none',
                'padding': '12px',
                'fontSize': '16px',
                'fontWeight': 'bold',
                'backgroundColor': 'rgba(240, 240, 240, 0.5)',
                'color': '#555'
            },
            selected_style={
                'borderTop': '2px solid #428bca',
                'borderLeft': '1px solid #d6d6d6',
                'borderRight': '1px solid #d6d6d6',
                'borderBottom': 'none',
                'padding': '12px',
                'fontSize': '16px',
                'fontWeight': 'bold',
                'backgroundColor': 'white',
                'color': '#2c3e50'
            },
            children=[
                html.Div(id='parametros-container')
            ]
        )
    ], style={
        'borderBottom': '1px solid #d6d6d6',
        'marginBottom': '20px'
    })
    
    return tabs, {'display': 'block'}


@callback(
    Output('parametros-container', 'children'),
     Input('session-stored-data', 'data')  # Cambiado a usar session-stored-data
)


def update_parametros_espaciotemporales(stored_data):
    if stored_data is None or 'parametros_espaciotemporales' not in stored_data:
        return []
    
    parametros = stored_data['parametros_espaciotemporales']


    def crear_grafico_barras_apilado(valores, titulo, tipo, max_valor=None):
        fig = go.Figure()
        
        # Colores definidos
        color_derecho = '#FF5252'  # Rojo
        color_derecho_claro = '#FF9999'  # Rojo claro
        color_izquierdo = '#4285F4'  # Azul
        color_izquierdo_claro = '#9999FF'  # Azul claro
        
        if tipo == "porcentaje":
            # Valores para porcentajes
            balanceo_derecho = valores[0]
            apoyo_derecho = 100 - valores[0]
            balanceo_izquierdo = valores[1]
            apoyo_izquierdo = 100 - valores[1]
            unidades = "%"
        else:
            # Valores para tiempos
            balanceo_derecho = valores[0]
            apoyo_derecho = max_valor - valores[0]
            balanceo_izquierdo = valores[1]
            apoyo_izquierdo = max_valor - valores[1]
            unidades = "s"
        
        # Barras para el lado DERECHO
        fig.add_trace(go.Bar(
            x=['Derecho'],
            y=[balanceo_derecho],
            name='Balanceo Derecho',
            marker_color=color_derecho,
            text=[f"{balanceo_derecho:.1f}{unidades}"],
            textposition='inside',
            textfont=dict(color='white', size=12),
            width=0.4
        ))
        fig.add_trace(go.Bar(
            x=['Derecho'],
            y=[apoyo_derecho],
            name='Apoyo Derecho',
            marker_color=color_derecho_claro,
            text=[f"{apoyo_derecho:.1f}{unidades}"],
            textposition='inside',
            textfont=dict(color='white', size=12),
            width=0.4
        ))
        
        # Barras para el lado IZQUIERDO
        fig.add_trace(go.Bar(
            x=['Izquierdo'],
            y=[balanceo_izquierdo],
            name='Balanceo Izquierdo',
            marker_color=color_izquierdo,
            text=[f"{balanceo_izquierdo:.1f}{unidades}"],
            textposition='inside',
            textfont=dict(color='white', size=12),
            width=0.4
        ))
        fig.add_trace(go.Bar(
            x=['Izquierdo'],
            y=[apoyo_izquierdo],
            name='Apoyo Izquierdo',
            marker_color=color_izquierdo_claro,
            text=[f"{apoyo_izquierdo:.1f}{unidades}"],
            textposition='inside',
            textfont=dict(color='white', size=12),
            width=0.4
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>{titulo}</b>",
                font=dict(size=14, family="Arial"),
                xanchor='center',
                x=0.5
            ),
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=80, b=40),
            height=350,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            xaxis=dict(
                showline=True,
                linecolor='rgba(0,0,0,0.2)',
                showgrid=False
            ),
            yaxis=dict(
                showline=True,
                linecolor='rgba(0,0,0,0.2)',
                showgrid=False,
                title="Valor"
            )
        )
        
         # Mapeo de títulos a imágenes
        IMAGENES_TOOLTIP_APILADAS = {
            "Distribución del Ciclo": "/assets/images/distribucion_ciclo.png",
            "Tiempos del Ciclo": "/assets/images/tiempos_ciclo.png"
        }
        
        imagen_path = IMAGENES_TOOLTIP_APILADAS.get(titulo, "")
        
        graph = dcc.Graph(
            figure=fig,
            style={'margin': '20px 0'},
            config={'displayModeBar': False}
        )
        
        if imagen_path:
            return html.Div([
                dbc.Tooltip(
                    html.Img(
                        src=imagen_path,
                        style={
                            'width': '300px',
                            'borderRadius': '5px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
                        }
                    ),
                    target=f"tooltip-{titulo.replace(' ', '-').lower()}",
                    placement="top",
                    style={
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'maxWidth': 'none'
                    }
                ),
                html.Div(graph, id=f"tooltip-{titulo.replace(' ', '-').lower()}")
            ])
        return graph
    

    
    # Función para crear gráficos de barras con el estilo consistente
    def crear_grafico_barras(valores, titulo, colores, max_valor=None, unidades="", apilado=False):
        fig = go.Figure()
        
        if max_valor is None:
            max_valor = max(valores) * 1.2 if valores else 1
        
        if len(valores) == 2:
            if apilado:
                # Definir colores apilados basados en el título
                colores_apilados = ['#FF5252', '#FF9999'] if 'balanceo' in titulo.lower() else ['#4285F4', '#9999FF']
                
                # Gráfico apilado (para balanceo/apoyo)
                fig.add_trace(go.Bar(
                    x=['Derecho', 'Izquierdo'],
                    y=[valores[0], valores[1]],
                    name='Balanceo' if 'balanceo' in titulo.lower() else 'Apoyo',
                    marker_color=colores_apilados[0],
                    marker_line=dict(width=1, color='rgba(0,0,0,0.3)'),
                    width=0.5,
                    text=[f"{valores[0]:.1f}{unidades}", f"{valores[1]:.1f}{unidades}"],
                    textposition='inside',
                    textfont=dict(size=12, color='white')
                ))
                fig.add_trace(go.Bar(
                    x=['Derecho', 'Izquierdo'],
                    y=[100-valores[0], 100-valores[1]] if '%' in unidades else [max_duracion-valores[0], max_duracion-valores[1]],
                    name='Apoyo' if 'balanceo' in titulo.lower() else 'Balanceo',
                    marker_color=colores_apilados[1],
                    marker_line=dict(width=1, color='rgba(0,0,0,0.3)'),
                    width=0.5,
                    textposition='inside',
                    textfont=dict(size=12, color='white')
                ))
            else:
                # Gráfico normal de barras
                fig.add_trace(go.Bar(
                    x=['Derecho', 'Izquierdo'],
                    y=valores,
                    marker_color=colores,
                    marker_line=dict(width=1, color='rgba(0,0,0,0.3)'),
                    width=0.5,
                    text=[f"{v:.2f}{unidades}" for v in valores],
                    textposition='auto',
                    textfont=dict(size=12, color='white')
                ))
            # Mapeo de títulos a rutas de imágenes
        IMAGENES_TOOLTIP = {
        "Duración del Ciclo": "/assets/images/duracion_ciclo.png",
        "Longitud del Ciclo": "/assets/images/longitud_ciclo.png",
        "Duración del Paso": "/assets/images/duracion_paso.png",
        # Agrega más mapeos según necesites
        }
    
         # Obtener la ruta de la imagen para este título
        imagen_path = IMAGENES_TOOLTIP.get(titulo.split("(")[0].strip(), "")
        
        
        fig.update_layout(
            title=dict(
                text=f"<b>{titulo}</b>",
                font=dict(size=14, family="Arial"),
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial"),
            showlegend=apilado,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ) if apilado else None,
            xaxis=dict(
                showline=True,
                linecolor='rgba(0,0,0,0.2)',
                showgrid=False
            ),
            yaxis=dict(
                showline=True,
                linecolor='rgba(0,0,0,0.2)',
                showgrid=False,
                range=[0, max_valor * 1.1] if not apilado else None,
                showticklabels=True,
                title="Valor"
            ),
            barmode='stack' if apilado else 'group'
        )

        graph = dcc.Graph(
            figure=fig,
            style={'margin': '20px 0'},
            config={'displayModeBar': False}
        )
        
        # Si hay imagen para este gráfico, agregar el tooltip
        if imagen_path:
            return html.Div([
                dbc.Tooltip(
                    html.Img(
                        src=imagen_path,
                        style={
                            'width': '300px',
                            'borderRadius': '5px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
                        }
                    ),
                    target=f"tooltip-{titulo.replace(' ', '-').lower()}",
                    placement="top",
                    style={
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'maxWidth': 'none'
                    }
                ),
                html.Div(graph, id=f"tooltip-{titulo.replace(' ', '-').lower()}")
            ])
        return graph
    
    # Colores consistentes
    colores_lados = ['#FF5252', '#4285F4']  # Rojo y azul
    
    # Rangos máximos
    max_duracion = max(
        parametros['duracion_ciclo_derecho'], parametros['duracion_ciclo_izquierdo'],
        parametros['tiempo_paso_derecho'], parametros['tiempo_paso_izquierdo']
    ) * 1.2
    
    max_longitud = max(
        parametros['longitud_ciclo_derecho'], parametros['longitud_ciclo_izquierdo'],
        parametros['longitud_paso_derecho'], parametros['longitud_paso_izquierdo']
    ) * 1.2
    
    max_ancho = max(
        parametros['ancho_paso_derecho'], parametros['ancho_paso_izquierdo']
    ) * 1.2
    
    # Primera fila: Números de ciclos y pasos
    fila1 = dbc.Row([
        dbc.Col(html.Div([
            html.H5("Ciclos Derecha", style={
                'textAlign': 'center',
                'color': '#FF5252',
                'marginBottom': '10px'
            }),
            html.H2(f"{parametros['num_ciclos_derecho']}", style={
                'textAlign': 'center',
                'color': '#FF5252',
                'fontSize': '42px',
                'fontWeight': '600'
            })
        ], style={
            'padding': '20px',
            'borderRadius': '8px',
            'backgroundColor': 'rgba(255, 82, 82, 0.1)'
        }), width=4),
        
        dbc.Col(html.Div([
            html.H5("Ciclos Izquierda", style={
                'textAlign': 'center',
                'color': '#4285F4',
                'marginBottom': '10px'
            }),
            html.H2(f"{parametros['num_ciclos_izquierdo']}", style={
                'textAlign': 'center',
                'color': '#4285F4',
                'fontSize': '42px',
                'fontWeight': '600'
            })
        ], style={
            'padding': '20px',
            'borderRadius': '8px',
            'backgroundColor': 'rgba(66, 133, 244, 0.1)'
        }), width=4),
        
        dbc.Col(html.Div([
            html.H5("Pasos Totales", style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '10px'
            }),
            html.H2(f"{parametros['num_pasos']}", style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'fontSize': '42px',
                'fontWeight': '600'
            })
        ], style={
            'padding': '20px',
            'borderRadius': '8px',
            'backgroundColor': 'rgba(44, 62, 80, 0.1)'
        }), width=4)
    ], className="mb-4")
    
    # Segunda fila: Duración y longitud del ciclo + espacio
    fila2 = dbc.Row([
        dbc.Col(crear_grafico_barras(
            [parametros['duracion_ciclo_derecho'], parametros['duracion_ciclo_izquierdo']],
            "Duración del Ciclo (s)",
            colores_lados,
            max_duracion
        ), width=4),
        
        dbc.Col(crear_grafico_barras(
            [parametros['longitud_ciclo_derecho'], parametros['longitud_ciclo_izquierdo']],
            "Longitud del Ciclo (m)",
            colores_lados,
            max_longitud
        ), width=4),
        
        dbc.Col(html.Div([
            html.H5("Velocidad", style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '5px'
            }),
            html.H3(f"{parametros['velocidad']:.2f} m/s", style={
                'textAlign': 'center',
                'color': '#428bca',
                'fontSize': '32px',
                'fontWeight': '600',
                'marginBottom': '20px'
            }),
            html.H5("Cadencia", style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '5px'
            }),
            html.H3(f"{parametros['cadencia']:.2f} pasos/min", style={
                'textAlign': 'center',
                'color': '#428bca',
                'fontSize': '32px',
                'fontWeight': '600'
            })
        ], style={
            'padding': '30px',
            'borderRadius': '8px',
            'backgroundColor': 'rgba(255,255,255,0.7)',
            'border': '1px solid rgba(0,0,0,0.1)',
            'height': '100%',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center'
        }), width=4)
    ], className="mb-4")
    
    # Tercera fila: Parámetros del paso
    fila3 = dbc.Row([
        dbc.Col(crear_grafico_barras(
            [parametros['tiempo_paso_derecho'], parametros['tiempo_paso_izquierdo']],
            "Duración del Paso (s)",
            colores_lados,
            max_duracion
        ), width=4),
        
        dbc.Col(crear_grafico_barras(
            [parametros['longitud_paso_derecho'], parametros['longitud_paso_izquierdo']],
            "Longitud del Paso (m)",
            colores_lados,
            max_longitud
        ), width=4),
        
        dbc.Col(crear_grafico_barras(
            [parametros['ancho_paso_derecho'], parametros['ancho_paso_izquierdo']],
            "Ancho del Paso (m)",
            colores_lados,
            max_ancho
        ), width=4)
    ], className="mb-4")
    

    
    # Quinta fila: Porcentajes apilados
    fila5 = dbc.Row([
        # Gráfico de porcentajes apilados
        dbc.Col(crear_grafico_barras_apilado(
            [parametros['balanceo_derecho'], parametros['balanceo_izquierdo']],
            "Distribución del Ciclo (%)",
            tipo="porcentaje"
        ), width=6),
        
        # Gráfico de tiempos apilados
        dbc.Col(crear_grafico_barras_apilado(
            [parametros['tiempo_balanceo_derecho'], parametros['tiempo_balanceo_izquierdo']],
            "Tiempos del Ciclo (s)",
            tipo="tiempo",
            max_valor=max_duracion
        ), width=6)
    ])
    

    return html.Div([
        html.H3("Parámetros Espaciotemporales", style={
            'textAlign': 'center',
            'color': '#2c3e50',
            'margin': '20px 0 30px 0',
            'fontFamily': 'Arial',
            'fontWeight': '600'
        }),
        
        fila1,
        fila2,
        fila3,
        fila5
    ], style={'padding': '0 20px'})

def final_plot_plotly(curva_derecha=None, curva_izquierda=None, posibilidad="Derecha",
                      rango_z=(-20, 60), rango_y=(-30, 30), rango_x=(-30, 30), articulacion=""):
    
    # Colores y etiquetas
    colors = {
        'Derecha': '#FF5252',  # Rojo
        'Izquierda': '#4285F4',  # Azul
    }
    
    labels = {
        'Derecha': 'Derecho',
        'Izquierda': 'Izquierdo'
    }

    light_colors = {
        '#FF5252': 'rgba(255, 82, 82, 0.2)',
        '#4285F4': 'rgba(66, 133, 244, 0.2)'
    }

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"<b>Ángulos de Z</b>",
            f"<b>Ángulos de Y</b>",
            f"<b>Ángulos de X</b>"
        ),
        shared_yaxes=False,
        horizontal_spacing=0.08
    )

    # Procesar cada lado según la selección
    sides_to_plot = []
    if posibilidad == 'Derecha' and curva_derecha is not None:
        sides_to_plot.append(('Derecha', curva_derecha))
    elif posibilidad == 'Izquierda' and curva_izquierda is not None:
        sides_to_plot.append(('Izquierda', curva_izquierda))
    elif posibilidad == 'Ambos':
        if curva_derecha is not None:
            sides_to_plot.append(('Derecha', curva_derecha))
        if curva_izquierda is not None:
            sides_to_plot.append(('Izquierda', curva_izquierda))

    for side, curva in sides_to_plot:
        color = colors[side]
        label = labels[side]
        
        Z_curves = np.array(curva["Z"].tolist())
        Y_curves = np.array(curva["Y"].tolist())
        X_curves = np.array(curva["X"].tolist())

        average_Z = np.mean(Z_curves, axis=0)
        std_Z = np.std(Z_curves, axis=0)

        average_Y = np.mean(Y_curves, axis=0)
        std_Y = np.std(Y_curves, axis=0)

        average_X = np.mean(X_curves, axis=0)
        std_X = np.std(X_curves, axis=0)

        fixed_time = np.linspace(0, 100, num=len(average_Z))

        # Función para agregar trazos
        def add_traces(row, col, average, std, axis):
            # Solo mostrar leyenda en el primer gráfico (Z)
            show_legend = col == 1
            
            # Línea de promedio
            fig.add_trace(go.Scatter(
                x=fixed_time,
                y=average,
                line=dict(color=color, width=2),
                name=f"{label}: Promedio",
                legendgroup=label,
                showlegend=show_legend,
                mode='lines'
            ), row=row, col=col)
            
            # Área de desviación estándar
            fig.add_trace(go.Scatter(
                x=fixed_time,
                y=average + std,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                legendgroup=label
            ), row=row, col=col)
            
            fig.add_trace(go.Scatter(
                x=fixed_time,
                y=average - std,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=light_colors[color],
                name=f"{label}: Desviación",
                legendgroup=label,
                showlegend=show_legend,
            ), row=row, col=col)

        # Agregar trazos para cada eje
        add_traces(1, 1, average_Z, std_Z, "Z")
        add_traces(1, 2, average_Y, std_Y, "Y")
        add_traces(1, 3, average_X, std_X, "X")

    fig.update_layout(
        title_text=f"<b>Articulación de {articulacion.capitalize()}</b> - Ángulos por eje",
        height=400,
        width=1200,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=1.15,  # Posición vertical ajustada
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.7)',
            itemsizing='constant',
            traceorder='normal'  # Orden normal de las leyendas
        ),
        margin=dict(l=40, r=40, t=100, b=40)
    )

    # Configuración de ejes
    for col, rango in zip([1, 2, 3], [rango_z, rango_y, rango_x]):
        fig.update_yaxes(
            title_text="Ángulo (°)",
            range=rango,
            row=1, col=col,
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='rgba(0,0,0,0.2)'
        )
        fig.update_xaxes(
            title_text="% Ciclo de Marcha",
            row=1, col=col,
            gridcolor='rgba(0,0,0,0.1)'
        )

    return fig



@callback(
    [Output('output-c3d-upload', 'children'),
     Output('stored-data', 'data'),
     Output('session-stored-data', 'data')],  # Nueva salida
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
            'parametros_espaciotemporales': {}
        }

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as temp_file:
                temp_file.write(decoded)
                temp_filepath = temp_file.name

            curvas_tobillo_derecho, curvas_tobillo_izquierdo, curvas_rodilla_derecha, curvas_rodilla_izquierda, curvas_cadera_derecha, curvas_cadera_izquierda, parametros_espaciotemporales = procesar_archivo_c3d(temp_filepath)

            os.remove(temp_filepath)

            results['tobillo_derecho'].extend(curvas_tobillo_derecho.to_dict('records'))
            results['tobillo_izquierdo'].extend(curvas_tobillo_izquierdo.to_dict('records'))
            results['rodilla_derecha'].extend(curvas_rodilla_derecha.to_dict('records'))
            results['rodilla_izquierda'].extend(curvas_rodilla_izquierda.to_dict('records'))
            results['cadera_derecha'].extend(curvas_cadera_derecha.to_dict('records'))
            results['cadera_izquierda'].extend(curvas_cadera_izquierda.to_dict('records'))
            results['parametros_espaciotemporales'] = parametros_espaciotemporales
            
            alerta = dbc.Alert(
                f"Archivo {filename} cargado correctamente",
                color="success",
                dismissable=True,
                style={
                    'margin': '10px 0',
                    'borderLeft': '4px solid #28a745'
                }
            )
            return alerta, results, results  # Devuelve los datos a ambos almacenamientos

        except Exception as e:
            alerta = dbc.Alert(
                f"Error al procesar el archivo {filename}: {str(e)}",
                color="danger",
                dismissable=True,
                style={
                    'margin': '10px 0',
                    'borderLeft': '4px solid #dc3545'
                }
            )
            return alerta, None, None
    
    return html.Div(), dash.no_update, dash.no_update  # No actualices los almacenamientos si no hay contenido





@callback(
    Output('graphs-container', 'children'),
    [Input('lado-dropdown', 'value'),
     Input('session-stored-data', 'data')]  
)
def update_graphs(lado, stored_data):
    if stored_data is None:
        return []
    
    # Mapear los valores de los radio buttons a los nombres usados en los datos
    lado_map = {
        'Derecha': 'derecho',
        'Izquierda': 'izquierdo',
        'Ambos': 'Ambos'
    }
    lado_seleccionado = lado_map.get(lado, 'Ambos')

    # Convertir los datos a DataFrames
    def crear_df(lado):
        key = f'tobillo_{lado}'
        return pd.DataFrame(stored_data[key]) if stored_data.get(key) else None
    
    curvas_tobillo_derecho = crear_df('derecho')
    curvas_tobillo_izquierdo = crear_df('izquierdo')
    curvas_rodilla_derecha = pd.DataFrame(stored_data['rodilla_derecha']) if stored_data.get('rodilla_derecha') else None
    curvas_rodilla_izquierda = pd.DataFrame(stored_data['rodilla_izquierda']) if stored_data.get('rodilla_izquierda') else None
    curvas_cadera_derecha = pd.DataFrame(stored_data['cadera_derecha']) if stored_data.get('cadera_derecha') else None
    curvas_cadera_izquierda = pd.DataFrame(stored_data['cadera_izquierda']) if stored_data.get('cadera_izquierda') else None

    graphs = []

    articulaciones = [
        ('tobillo', curvas_tobillo_derecho, curvas_tobillo_izquierdo, RANGOS_Y["tobillo"]),
        ('rodilla', curvas_rodilla_derecha, curvas_rodilla_izquierda, RANGOS_Y["rodilla"]),
        ('cadera', curvas_cadera_derecha, curvas_cadera_izquierda, RANGOS_Y["cadera"])
    ]

    for articulacion, curva_derecha, curva_izquierda, rangos in articulaciones:
        rango_z, rango_y, rango_x = rangos.values()
        
        # Determinar qué curvas mostrar según la selección
        if lado_seleccionado == 'derecho' and curva_derecha is not None:
            curvas_a_mostrar = (curva_derecha, None)
        elif lado_seleccionado == 'izquierdo' and curva_izquierda is not None:
            curvas_a_mostrar = (None, curva_izquierda)
        elif lado_seleccionado == 'Ambos':
            curvas_a_mostrar = (curva_derecha, curva_izquierda)
        else:
            continue  # No hay datos para mostrar

        fig = final_plot_plotly(
            curvas_a_mostrar[0],  # Curva derecha
            curvas_a_mostrar[1],  # Curva izquierda
            posibilidad=lado,
            rango_z=rango_z,
            rango_y=rango_y,
            rango_x=rango_x,
            articulacion=articulacion
        )
        
        graphs.append(dbc.Col(dcc.Graph(figure=fig), width=12))

    return graphs if graphs else dbc.Alert("No hay datos disponibles para la selección actual", color="warning")