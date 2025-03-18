import dash
from dash import dcc, html, callback, Output, Input

import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home') # '/' is home page

layout = html.Div(
    [
        dbc.Row([dcc.Markdown(' # Informacion sobre el proyecto y la pag web')]), 
        dbc.Row([dbc.Col(
                [
                    html.Img(src='assets/description.jpg')
                ], width=8
            )
            ]),
        dbc.Row([dcc.Markdown(''' Esta es la informacion que tiene que saber sobre el proyecto y la pag web. 
                              
        En un laboratorio de análisis de la marcha, se lleva a cabo un minucioso estudio destinado a comprender y evaluar los patrones de movimiento humano
        durante la locomoción. Este proceso implica la utilización de avanzadas tecnologías y equipos especializados para obtener datos precisos y detallados
        sobre la forma en que las personas caminan o corren.

        Uno de los componentes clave en este laboratorio es el sistema de captura de movimiento, que utiliza cámaras tridimensionales para rastrear los
        movimientos del cuerpo en tiempo real. Los sujetos, equipados con marcadores reflectantes estratégicamente ubicados en sus articulaciones y extremidades,
        son observados mientras caminan sobre una superficie específica. Este sistema proporciona información invaluable sobre la cinemática de la marcha,
        permitiendo a los investigadores analizar aspectos como la longitud de zancada, la velocidad de paso y los ángulos articulares. ''')])
    ]
)

