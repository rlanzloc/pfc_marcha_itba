import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', icon="house")

layout = html.Div(
    [
        # TÍTULO
        dbc.Row(
            dbc.Col(
                dcc.Markdown(
                    "## 🚶 **Bienvenido al sistema de análisis de marcha** 🚶",
                    className="mb-4",
                    style={
                        'color': '#2c3e50',
                        'fontWeight': '600',
                        'textAlign': 'center',
                        'marginTop': '0px',
                        'marginBottom': '0px'
                    }
                )
            )
        ),

        # FILA DE TEXTOS EN CARDS LADO A LADO
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Markdown(
                                '''
Este sistema fue desarrollado para integrar información **cinemática y cinética** de la marcha humana, combinando datos de **captura de movimiento 3D** y **plantillas instrumentadas**.

La interfaz permite:
- 📁 Cargar archivos `.c3d` y `.csv` de pruebas experimentales.
- 🔌 Leer datos en tiempo real desde plantillas instrumentadas vía **Bluetooth**.
- 📊 Visualizar curvas articulares, fuerzas plantares, fases de apoyo y centro de presión.
                                ''',
                                style={
                                    'fontSize': '15px',
                                    'color': '#2c3e50',
                                    'lineHeight': '1.3'
                                }
                            )
                        ),
                        className="mk-tight",
                        style={'height': '100%'}
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Markdown(
                                '''
El análisis se divide en dos componentes complementarios:

🔹 **Análisis cinemático**
- Ángulos articulares (cadera, rodilla, tobillo)
- Variables espaciotemporales

🔹 **Análisis cinético**
- Fuerza vertical de reacción del suelo (**vGRF**)
- Distribución de cargas en el pie
- Trayectoria del **centro de presión (CoP)**
                                ''',
                                style={
                                    'fontSize': '15px',
                                    'color': '#2c3e50',
                                    'lineHeight': '1.3'
                                }
                            )
                        ),
                        className="mk-tight",
                        style={'height': '100%'}
                    ),
                    width=6
                )
            ],
            className="gy-1",
            align="stretch",
            style={'marginBottom':'8px'}  
        ),

        # FILA DE GIFs
        dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src='assets/animacion.gif',
                        style={
                            'width': '100%',
                            'maxWidth': '100%',
                            'height': 'auto',
                            'maxHeight': '200px', 
                            'borderRadius': '10px'
                        }
                    ),
                    width=6
                ),
                dbc.Col(
                    html.Img(
                        src='assets/plantillas.gif',
                        style={
                            'width': '100%',
                            'maxWidth': '100%',
                            'height': 'auto',
                            'maxHeight': '200px',  
                            'borderRadius': '10px'
                        }
                    ),
                    width=6
                )
            ]
        )
    ],
    className="gy-1",
    style={
        "padding": "10px",
        "paddingTop": "0px",
        'marginTop':'6px',
        "backgroundColor": "#f8f9fa"
    }
)


