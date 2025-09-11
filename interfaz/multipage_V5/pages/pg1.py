import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State

dash.register_page(__name__, path='/', name='Home', icon="house")

layout = html.Div(
    [

        # TÍTULO + BOTÓN
        dbc.Row(
            [
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
                    ),
                    width="auto"
                ),

                dbc.Col(
                    [
                        dbc.Button(
                            "Ver protocolo",
                            id="btn-ver-protocolo",
                            color="primary",
                            size="sm",
                            style={
                                "float": "right",
                                "marginTop": "4px"
                            }
                        ),

                        # Modal con PDF incrustado
                        dbc.Modal(
                            [
                                dbc.ModalHeader(
                                    dbc.ModalTitle("Protocolo de Captura de Datos para el Análisis de la Marcha"),
                                    close_button=True, 
                                ),
                                dbc.ModalBody(
                                    html.Iframe(
                                        id="iframe-protocolo",
                                        src="/assets/protocolo.pdf",
                                        style={
                                            "width": "100%",
                                            "height": "80vh",
                                            "border": "none"
                                        }
                                    )
                                ),
                            ],
                            id="modal-protocolo",
                            is_open=False,
                            size="xl",       # ancho grande (cambié "l" -> "xl")
                            centered=True,   # centrado vertical
                            backdrop=True,   # fondo opaco
                            scrollable=True,
                        ),
                    ],
                    width="auto",
                    style={"textAlign": "center", "marginBottom": "12px"}
                ),
            ],
            justify="between",   # separa título y botón
            align="center",
            style={"marginBottom": "10px"}
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
            style={'marginBottom': '8px'}
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
        'marginTop': '6px',
        "backgroundColor": "#f8f9fa"
    }
)




@callback(
    Output("modal-protocolo", "is_open"),
    Input("btn-ver-protocolo", "n_clicks"),
    State("modal-protocolo", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(n_open, is_open):
    if n_open:
        return not is_open  # alterna al hacer clic en "Ver protocolo"
    return is_open