import dash
from dash import dcc, html, callback, Output, Input, State, ctx, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import base64, io, os, serial, threading, datetime
import plotly.graph_objects as go
from pathlib import Path
import json
from serial.tools import list_ports
from dash.exceptions import PreventUpdate
from io import StringIO
import plotly.express as px  

dash.register_page(__name__, path="/pg4", name='Análisis Cinético', icon="file-earmark-text")
from analisis_plantillas import procesar_archivo_plantillas

# -------- Inicialización de puertos -------------
def safe_serial(port):
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        ser.flushInput()
        ser.flushOutput()
        print(f"✅ Conectado a {port}")
        return ser
    except serial.SerialException as e:
        print(f"❌ Error al conectar {port}: {e}")
        return None

arduino_der = safe_serial("COM11")
arduino_izq = safe_serial("COM3")
puerto_disponible = arduino_der is not None and arduino_izq is not None

# -------- Variables y directorio -----------------
ruta_actual = Path(__file__).resolve().parent
directorio_guardado = ruta_actual / "assets" / "datos_sensores"
directorio_guardado.mkdir(parents=True, exist_ok=True)

reading = False
df_der = pd.DataFrame(columns=['Hora', 'Tiempo'] + [f'Derecha_S{i+1}' for i in range(8)])
df_izq = pd.DataFrame(columns=['Hora', 'Tiempo'] + [f'Izquierda_S{i+1}' for i in range(8)])
lock = threading.Lock()

def leer_datos(arduino, dataframe, pie):
    global reading
    while True:
        if reading and arduino and arduino.in_waiting > 0:
            try:
                linea = arduino.readline().decode('utf-8').strip()
                datos = linea.split(',')
                if len(datos) == 9:
                    hora_actual = datetime.datetime.now().strftime('%H:%M:%S.%f')
                    tiempo_arduino = datos[0]
                    valores = [int(val) for val in datos[1:]]
                    if pie == "Izquierda":
                        valores = valores[::-1]
                    fila = {'Hora': hora_actual, 'Tiempo': tiempo_arduino}
                    for i in range(8):
                        fila[f"{pie}_S{i+1}"] = valores[i]
                    with lock:
                        dataframe.loc[len(dataframe)] = fila
            except Exception as e:
                print(f"❌ Error leyendo {pie}: {e}")

if arduino_der:
    threading.Thread(target=leer_datos, args=(arduino_der, df_der, "Derecha"), daemon=True).start()
if arduino_izq:
    threading.Thread(target=leer_datos, args=(arduino_izq, df_izq, "Izquierda"), daemon=True).start()

layout = html.Div([
    dcc.Store(id='session-stored-cinetico', storage_type='session'),
    dbc.Col(html.H1("Análisis Cinético", className="text-center mb-4", style={
        'color': '#2c3e50',
        'fontWeight': '600',
        'marginTop': '10px'
    })),

    # Upload visible siempre arriba
    dcc.Upload(
        id='upload-csv',
        children=html.Div(['Arrastra o selecciona archivos para pie Izquierdo y Derecho']),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'marginTop': '10px',  # bajado de 30px a 10px
            'marginBottom': '10px',
            'cursor': 'pointer',
            'backgroundColor': 'rgba(66, 139, 202, 0.1)',
            'borderColor': '#428bca',
            'color': '#428bca',
            'transition': 'all 0.3s'
        },
        multiple=True
    ),

    html.Div(id='csv-validation'),

    html.Hr(),

    # Pestañas
    dcc.Tabs(id='tabs', value='tab-sensores', children=[
        dcc.Tab(label='Lectura en Vivo de Sensores', value='tab-sensores',
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
            }),
        dcc.Tab(label='Importar CSV', value='tab-csv',
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
            }),
    ]),

    html.Div(id='tabs-content'),

    # Los gráficos se muestran solo si hay datos válidos
    html.Div([
        dcc.Graph(id='grafico-izquierda', style={'display': 'none'}),
        dcc.Graph(id='grafico-derecha', style={'display': 'none'})
    ]),

    dcc.Store(id='data-izq-store'),
    dcc.Store(id='data-der-store')
])

# ---------- Renderizado dinámico de pestañas ----------
@callback(
    Output('port-status-dynamic', 'children'),
    Output('start-button', 'disabled'),
    Output('stop-button', 'disabled'),
    Input('refresh-button', 'n_clicks'),
)
def actualizar_estado_puerto(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate
    
    def verificar_puerto(port):
        try:
            ser = serial.Serial(port, 115200, timeout=0.5)  # Timeout corto solo para verificación
            ser.close()
            return True
        except:
            return False
    
    # Verificar ambos puertos
    der_disponible = verificar_puerto("COM11")
    izq_disponible = verificar_puerto("COM3")
    
    if der_disponible and izq_disponible:
        return " Disponible ✅", False, False
    elif der_disponible or izq_disponible:
        # Opcional: mensaje cuando solo uno está disponible
        disponible = "COM11" if der_disponible else "COM3"
        return f" Parcial ({disponible}) ⚠️", True, True
    else:
        return " No disponibles ❌", True, True


def render_tab(tab):
    if tab == 'tab-sensores':
        return html.Div([
            html.Div([
                html.Span("Estado de puertos: ", id='port-status-label'),
                html.Span(id='port-status-dynamic', children="Actualizar estado..."),
                html.Button("🔄", id='refresh-button', n_clicks=0, title="Actualizar", style={
                    'border': 'none',
                    'background': 'none',
                    'cursor': 'pointer',
                    'fontSize': '20px',
                    'marginLeft': '8px'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),  # menos margen aquí

            html.Div([  # contenedor flex para poner botones en la misma línea
                html.Button('Iniciar Lectura', id='start-button', n_clicks=0, style={
                    'backgroundColor': '#428bca', 'color': 'white', 'flex': '1'
                }),
                html.Button('Finalizar Lectura', id='stop-button', n_clicks=0, style={
                    'backgroundColor': '#428bca', 'color': 'white', 'flex': '1'
                }),
            ], style={
                'display': 'flex',
                'gap': '12px',  # espacio entre botones
                'width': '100%',
                'maxWidth': '400px',
                'marginBottom': '10px',
            }),

        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center',
            'maxWidth': '400px',
            'margin': 'auto',
            'gap': '15px'
        })


# ---------- Procesamiento del CSV ----------
@callback(
    Output('csv-validation', 'children'),
    Output('session-stored-cinetico', 'data'),
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename'),
    State('session-stored-cinetico', 'data'),
    prevent_initial_call=True
)
def handle_csv_upload(contents, filename, stored_data):
    if stored_data is None:
        stored_data = {'izq': None, 'der': None}
    expected_cols_izq = ['Hora', 'Tiempo'] + [f'Izquierda_S{i+1}' for i in range(8)]
    expected_cols_der = ['Hora', 'Tiempo'] + [f'Derecha_S{i+1}' for i in range(8)]
    alerts = []

    def parse_contents(contents, filename, expected_columns, label):
        nonlocal alerts
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                alerts.append(dbc.Alert(
                    f"{filename} - Faltan columnas para {label}: {', '.join(missing_cols)}",
                    color="danger",
                    dismissable=True,
                    style={
                        'margin': '10px 0',
                        'borderLeft': '4px solid #dc3545',
                        'width': '100%',
                        'boxSizing': 'border-box',
                        'textAlign': 'center'
                    }
                ))
                return None
            else:
                alerts.append(dbc.Alert(
                    f"{filename} subido correctamente como {label}",
                    color="success",
                    dismissable=True,
                    style={
                        'margin': '10px 0',
                        'borderLeft': '4px solid #28a745',
                        'width': '100%',
                        'boxSizing': 'border-box',
                        'textAlign': 'center'
                    }
                ))
                return df
        except Exception as e:
            alerts.append(dbc.Alert(
                f"Error al procesar {label}: {str(e)}",
                color="danger",
                dismissable=True,
                style={
                    'margin': '10px 0',
                    'borderLeft': '4px solid #dc3545',
                    'width': '100%',
                    'boxSizing': 'border-box',
                    'textAlign': 'center'
                }
            ))
            return None

    if contents and filename:
        # Puede subir varios archivos, pero dash los pasa como lista si multiple=True
        if isinstance(contents, list):
            for cont, fname in zip(contents, filename):
                fname_lower = fname.lower()
                if 'izq' in fname.lower():
                    df_izq_csv = parse_contents(cont, fname, expected_cols_izq, "Pie Izquierdo")
                    if df_izq_csv is not None:
                        stored_data['izq'] = df_izq_csv.to_json(date_format='iso', orient='split')
                elif 'der' in fname.lower():
                    df_der_csv = parse_contents(cont, fname, expected_cols_der, "Pie Derecho")
                    if df_der_csv is not None:
                        stored_data['der'] = df_der_csv.to_json(date_format='iso', orient='split')
                else:
                    alerts.append(dbc.Alert(
                        f"No se reconoce si el archivo {fname} es de pie izquierdo o derecho. Debe incluir 'izq' o 'der' en el nombre.",
                        color="warning",
                        dismissable=True,
                        style={
                            'margin': '10px 0',
                            'borderLeft': '4px solid #ffc107',
                            'width': '100%',
                            'boxSizing': 'border-box',
                            'textAlign': 'center'
                        }
                    ))
        else:
            # Caso de un solo archivo
            fname_lower = filename.lower()
            if 'izq' in fname_lower or 'izquierda' in fname_lower:
                df_izq_csv = parse_contents(contents, filename, expected_cols_izq, "Pie Izquierdo")
                if df_izq_csv is not None:
                        stored_data['izq'] = df_izq_csv.to_json(date_format='iso', orient='split')
            elif 'der' in fname_lower or 'derecha' in fname_lower:
                df_der_csv = parse_contents(contents, filename, expected_cols_der, "Pie Derecho")
                if df_der_csv is not None:
                    stored_data['der'] = df_der_csv.to_json(date_format='iso', orient='split')
            else:
                alerts.append(dbc.Alert(
                    f"No se reconoce si el archivo {filename} es de pie izquierdo o derecho. Debe incluir 'izq' o 'der' en el nombre.",
                    color="warning",
                    dismissable=True,
                    style={
                        'margin': '10px 0',
                        'borderLeft': '4px solid #ffc107',
                        'width': '100%',
                        'boxSizing': 'border-box',
                        'textAlign': 'center'
                    }
                ))
    else:
        return dash.no_update, dash.no_update, dash.no_update


    return alerts, stored_data

# ---------- Mostrar gráficos con los datos cargados ----------
@callback(
    Output('grafico-izquierda', 'figure'),
    Output('grafico-izquierda', 'style'),
    Output('grafico-derecha', 'figure'),
    Output('grafico-derecha', 'style'),
    Input('session-stored-cinetico', 'data'),
    Input('tabs', 'value'),
)
def actualizar_graficos(stored_data, tab_value):
    fig_izq = go.Figure()
    fig_der = go.Figure()
    style_hide = {'display': 'none'}
    style_show = {'display': 'block', 'width': '95%', 'height': '400px', 'margin': 'auto'}
    
    style_der = style_hide
    style_izq = style_hide

    # Si la pestaña activa es lectura en vivo
    if tab_value == 'tab-sensores':
        # Usar datos globales de lectura en vivo df_izq y df_der
        global df_izq, df_der
        with lock:
            if not df_izq.empty:
                df_live_izq = df_izq.copy()
            else:
                df_live_izq = None
            if not df_der.empty:
                df_live_der = df_der.copy()
            else:
                df_live_der = None

        if df_live_izq is not None:
            df_live_izq.set_index('Tiempo', inplace=True)
            for i in range(8):
                sensor_col = f'Izquierda_S{i+1}'
                fig_izq.add_trace(go.Scatter(
                    x=df_live_izq.index,
                    y=df_live_izq[sensor_col],
                    mode='lines',
                    name=sensor_col
                ))
            fig_izq.update_layout(
                title='Lectura en Vivo - Pie Izquierdo',
                xaxis_title='Tiempo',
                yaxis_title='Valores crudos',
                template='plotly_white'
            )
            style_izq = style_show
        else:
            style_izq = style_hide

        if df_live_der is not None:
            df_live_der.set_index('Tiempo', inplace=True)
            for i in range(8):
                sensor_col = f'Derecha_S{i+1}'
                fig_der.add_trace(go.Scatter(
                    x=df_live_der.index,
                    y=df_live_der[sensor_col],
                    mode='lines',
                    name=sensor_col
                ))
            fig_der.update_layout(
                title='Lectura en Vivo - Pie Derecho',
                xaxis_title='Tiempo',
                yaxis_title='Valores crudos',
                template='plotly_white'
            )
            style_der = style_show
        else:
            style_der = style_hide

    # Si la pestaña activa es CSV
    elif tab_value == 'tab-csv' and stored_data:
        # Procesar datos si están disponibles
        if stored_data.get('izq') or stored_data.get('der'):
            # Cargar datos crudos
            df_izq_csv = pd.read_json(StringIO(stored_data['izq']), orient='split') if stored_data.get('izq') else None
            df_der_csv = pd.read_json(StringIO(stored_data['der']), orient='split') if stored_data.get('der') else None
            
            # Aplicar función de procesamiento
            try:
                filt_der, filt_izq = procesar_archivo_plantillas(
                    df_der_csv if df_der_csv is not None else pd.DataFrame(),
                    df_izq_csv if df_izq_csv is not None else pd.DataFrame()
                )
                
                # Asumimos que cada lista contiene al menos un DataFrame
                df_izq_procesado = filt_izq[0] if filt_izq else df_izq_csv
                df_der_procesado = filt_der[0] if filt_der else df_der_csv
                
            except Exception as e:
                print(f"Error en procesamiento: {e}")
                df_izq_procesado = df_izq_csv
                df_der_procesado = df_der_csv

            # --- Gráfico PIE IZQUIERDO ---
            if df_izq_procesado is not None:
                for i in range(8):
                    col_name = f'Izquierda_S{i+1}'
                    if col_name in df_izq_procesado.columns:
                        fig_izq.add_trace(go.Scatter(
                            x=df_izq_procesado.index,
                            y=df_izq_procesado[col_name],
                            mode='lines',
                            name=f'{col_name}',
                            line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)], width=1.5)
                        ))
                fig_izq.update_layout(
                    title='<b>Pie Izquierdo</b> - Datos Procesados',
                    xaxis_title='Tiempo (s)',
                    yaxis_title='Fuerza (kg)',
                    template='plotly_white'
                )
                style_izq = style_show

            # --- Gráfico PIE DERECHO ---
            if df_der_procesado is not None:
                for i in range(8):
                    col_name = f'Derecha_S{i+1}'
                    if col_name in df_der_procesado.columns:
                        fig_der.add_trace(go.Scatter(
                            x=df_der_procesado.index,
                            y=df_der_procesado[col_name],
                            mode='lines',
                            name=f'{col_name}',
                            line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)], width=1.5)  # Colores diferentes
                        ))
                fig_der.update_layout(
                    title='<b>Pie Derecho</b> - Datos Procesados',
                    xaxis_title='Tiempo (s)',
                    yaxis_title='Fuerza (kg)',
                    template='plotly_white'
                )
                style_der = style_show

        else:
            style_izq = style_hide
            style_der = style_hide

    return fig_izq, style_izq, fig_der, style_der


# ---------- Callback para mostrar contenido de pestañas ----------
@callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'),
    #prevent_initial_call=True
)
def render_content(tab):
    return render_tab(tab)






