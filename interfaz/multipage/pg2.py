import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc

import subprocess

dash.register_page(__name__, name='Tip Analysis')




layout = html.Div([
    html.Button('Ejecutar script de Python', id='button'),
    html.Div(id='output')
])

@callback(
    Output('output', 'children'),
    [Input('button', 'n_clicks')]
)
def run_python_script(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        try:
            # Reemplaza 'ruta/al/tu_script.py' con la ruta de tu script de Python
            subprocess.Popen(['python', 'assets\leer_y_guardar_6.py'])
            return 'El script de Python se ha ejecutado correctamente.'
        except Exception as e:
            return f'Error al ejecutar el script de Python: {str(e)}'
    else:
        return ''

