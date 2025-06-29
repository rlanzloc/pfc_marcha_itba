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
import time

dash.register_page(__name__, path="/pg4", name='Análisis Cinético', icon="file-earmark-text")
from analisis_plantillas import procesar_archivo_plantillas

# -------- Variables y directorio -----------------
ruta_actual = Path(__file__).resolve().parent
directorio_guardado = ruta_actual / "assets" / "datos_sensores"
directorio_guardado.mkdir(parents=True, exist_ok=True)

reading = False
lectura_ya_iniciada = False
buffer_der = []
buffer_izq = []
df_der = pd.DataFrame()
df_izq = pd.DataFrame()
lock = threading.Lock()

def leer_datos(arduino, dataframe, pie):
    global reading
    print(f"🧵 Hilo iniciado para {pie}")
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
                        if pie == "Izquierda":
                            buffer_izq.append(fila)
                        else:
                            buffer_der.append(fila)
            except Exception as e:
                print(f"❌ Error leyendo {pie}: {e}")
        time.sleep(0.01)

# -------- Inicialización de puertos -------------
def safe_serial(port):
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        ser.flushInput()
        ser.flushOutput()
        print(f"✅ Conectado a {port}")
        return ser
    except serial.SerialException:
        return None

arduino_der = None
arduino_izq = None
hilos_lanzados = False  # para evitar múltiples hilos

def inicializar_puertos_y_hilos():
    global arduino_der, arduino_izq, hilos_lanzados

    # Solo intentar conectar si aún no está conectado
    if arduino_der is None:
        temp = safe_serial("COM6")
        if temp:
            arduino_der = temp

    if arduino_izq is None:
        temp = safe_serial("COM3")
        if temp:
            arduino_izq = temp

    # Iniciar hilos solo si ambos están conectados y aún no se iniciaron
    if arduino_der and arduino_izq and not hilos_lanzados:
        threading.Thread(target=leer_datos, args=(arduino_der, df_der, "Derecha"), daemon=True).start()
        threading.Thread(target=leer_datos, args=(arduino_izq, df_izq, "Izquierda"), daemon=True).start()

        hilos_lanzados = True


layout = html.Div([
    dcc.Store(id='session-stored-cinetico', storage_type='session'),
    
    dbc.Col(html.H1("Análisis Cinético", className="text-center mb-4", style={
        'color': '#2c3e50',
        'fontWeight': '600',
        'marginTop': '10px'
    })),

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


def render_tab(tab):
    if tab == 'tab-sensores':
        return html.Div([
            html.Div([
                html.Span("Estado de puertos:", id='port-status-label', style={'marginRight': '5px'}),
                html.Span(id='port-status-dynamic', children="Actualizando..."),
                html.Button("🔄", id='refresh-button', n_clicks=0, title="Actualizar", style={
                    'border': 'none',
                    'background': 'none',
                    'cursor': 'pointer',
                    'fontSize': '20px',
                    'marginLeft': '8px'
                }),
                html.Div(style={'flex': '1'}),  # separador flexible
                html.Button('Iniciar Lectura', id='start-button', n_clicks=0, disabled=True, style={
                    'backgroundColor': '#428bca',
                    'color': 'white',
                    'marginLeft': '10px'
                }),
                html.Button('Finalizar Lectura', id='stop-button', n_clicks=0, disabled=True, style={
                    'backgroundColor': '#428bca',
                    'color': 'white',
                    'marginLeft': '10px'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'flexWrap': 'wrap',
                'gap': '10px',
                'marginBottom': '10px',
                'width': '100%'
            }),

            html.Div(id='reading-status', style={'width': '100%'})
        ])
    elif tab == 'tab-csv':
        return html.Div([
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
                    'marginTop': '10px',
                    'marginBottom': '10px',
                    'cursor': 'pointer',
                    'backgroundColor': 'rgba(66, 139, 202, 0.1)',
                    'borderColor': '#428bca',
                    'color': '#428bca',
                    'transition': 'all 0.3s'
                },
                multiple=True
            ),
            html.Div(id='csv-validation')
        ])


# ---------- Callback para mostrar contenido de pestañas ----------
@callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'),
    #prevent_initial_call=True
)
def render_content(tab):
    return render_tab(tab)

# ---------- Renderizado dinámico de pestañas ----------    
@callback(
    Output('reading-status', 'children'),
    Output('session-stored-cinetico', 'data', allow_duplicate=True),
    Output('start-button', 'disabled'),
    Output('stop-button', 'disabled'),
    Output('port-status-dynamic', 'children'),
    Output('start-button', 'style'),
    Output('stop-button', 'style'),
    Input('start-button', 'n_clicks'),
    Input('stop-button', 'n_clicks'),
    Input('refresh-button', 'n_clicks'),
    prevent_initial_call=True
)
def controlar_puertos_y_lectura(n_start, n_stop, n_refresh):
    global puerto_disponible, lectura_ya_iniciada, reading
    global buffer_izq, buffer_der, df_izq, df_der 
    
    inicializar_puertos_y_hilos()

    triggered_id = ctx.triggered_id


    def verificar_objeto_serial(ser):
        return ser is not None and ser.is_open

    # 🔁 Verificamos los objetos serial ya abiertos
    der_disponible = verificar_objeto_serial(arduino_der)
    izq_disponible = verificar_objeto_serial(arduino_izq)

    puerto_disponible = der_disponible and izq_disponible

    port_status = ""
    start_disabled = True
    stop_disabled = True
    lectura_msg = dash.no_update
    session_data = dash.no_update

    if puerto_disponible:
        port_status = " Disponible ✅"
        if lectura_ya_iniciada:
            start_disabled = True
            stop_disabled = False
        else:
            start_disabled = False
            stop_disabled = True
    elif der_disponible or izq_disponible:
        port_status = " Parcialmente disponible ⚠️"
        start_disabled = True
        stop_disabled = True
    else:
        port_status = " No disponibles ❌"
        start_disabled = True
        stop_disabled = True

    # 🎬 Iniciar lectura
    if triggered_id == 'start-button':
        if not puerto_disponible or lectura_ya_iniciada:
            print("❌ No se puede iniciar: puerto_disponible =", puerto_disponible, "ya_iniciada =", lectura_ya_iniciada)
            raise PreventUpdate
        
        lectura_ya_iniciada = True
        reading = True
        
        with lock:
            buffer_izq.clear()
            buffer_der.clear()
                
        # 🔄 Limpiar buffer del puerto serial
        if arduino_der:
            arduino_der.reset_input_buffer()
        if arduino_izq:
            arduino_izq.reset_input_buffer()
        
        lectura_msg = dbc.Alert(
            "✅ Lectura en curso...",
            color="success",
            dismissable=True,   
            is_open=True,       
            duration=25000,      
            style={'textAlign': 'center', 'marginTop': '10px'}
        )

        start_disabled = True
        stop_disabled = False

    # ⏹️ Finalizar lectura
    elif triggered_id == 'stop-button':
        if not lectura_ya_iniciada:
            raise PreventUpdate

        lectura_ya_iniciada = False
        reading = False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with lock:

            columnas_izq = ['Hora', 'Tiempo'] + [f'Izquierda_S{i+1}' for i in range(8)]
            columnas_der = ['Hora', 'Tiempo'] + [f'Derecha_S{i+1}' for i in range(8)]

            df_izq = pd.DataFrame(buffer_izq)[columnas_izq] if buffer_izq else pd.DataFrame(columns=columnas_izq)
            df_der = pd.DataFrame(buffer_der)[columnas_der] if buffer_der else pd.DataFrame(columns=columnas_der)

            if not df_izq.empty:
                df_izq.to_csv(directorio_guardado / f"izquierda_{timestamp}.csv", index=False)
                print("✅ Archivo pie izquierdo guardado.")
            else:
                print("⚠️ No se guardó archivo izquierdo (vacío).")

            if not df_der.empty:
                df_der.to_csv(directorio_guardado / f"derecha_{timestamp}.csv", index=False)
                print("✅ Archivo pie derecho guardado.")
            else:
                print("⚠️ No se guardó archivo derecho (vacío).")

            session_data = {
                "izq": df_izq.to_json(date_format='iso', orient='split'),
                "der": df_der.to_json(date_format='iso', orient='split')
            }

        alert_text = [
            html.Span("⏹️ Lectura finalizada y datos guardados en:"),
            html.Code(str(directorio_guardado.resolve())),
            html.Br(),
        ]


        lectura_msg = dbc.Alert(
            alert_text,
            color="info",
            dismissable=True,
            is_open=True,
            duration=6000,
            style={'textAlign': 'center', 'marginTop': '10px'})

        start_disabled = False
        stop_disabled = True

    elif triggered_id == 'refresh-button':
        inicializar_puertos_y_hilos()  # reconecta si es necesario
        lectura_msg = dash.no_update
        session_data = dash.no_update
    
    # Estilos dinámicos según estado
    style_base = {
        'backgroundColor': '#428bca',
        'color': 'white',
        'marginLeft': '10px',
        'transition': 'all 0.3s ease'
    }
    style_disabled = style_base | {'opacity': '0.4', 'cursor': 'not-allowed'}
    style_enabled = style_base | {'opacity': '1', 'cursor': 'pointer'}

    start_style = style_disabled if start_disabled else style_enabled
    stop_style = style_disabled if stop_disabled else style_enabled

    return lectura_msg, session_data, start_disabled, stop_disabled, port_status, start_style, stop_style


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

    def detect_delimiter_from_string(content_string, sample_size=5):
        delimiters = [',', ';', '\t']
        lines = content_string.splitlines()[:sample_size]

        for delimiter in delimiters:
            counts = [line.count(delimiter) for line in lines]
            if all(count == counts[0] for count in counts) and counts[0] > 0:
                return delimiter
        return ','  # Por defecto

    def parse_contents(contents, filename, expected_columns, label):
        nonlocal alerts
        try:
            content_type, content_string = contents.split(',')
            decoded_string = base64.b64decode(content_string).decode('utf-8-sig')

            # Detectar delimitador
            delimiter = detect_delimiter_from_string(decoded_string)

            # Leer CSV con el delimitador detectado
            df = pd.read_csv(io.StringIO(decoded_string), sep=delimiter)

            # Validar columnas
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
        
    data_prev_izq = stored_data.get('izq')
    data_prev_der = stored_data.get('der')

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
        return dash.no_update, dash.no_update

    # Si uno de los dos no fue actualizado, conservar el valor anterior
    if stored_data['izq'] is None:
        stored_data['izq'] = data_prev_izq
    if stored_data['der'] is None:
        stored_data['der'] = data_prev_der

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
        global df_izq, df_der, reading
        df_live_izq = None
        df_live_der = None

        # Mostrar solo si NO está leyendo
        if not reading:
            with lock:
                if not df_izq.empty:
                    df_live_izq = df_izq.copy()
                if not df_der.empty:
                    df_live_der = df_der.copy()

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









