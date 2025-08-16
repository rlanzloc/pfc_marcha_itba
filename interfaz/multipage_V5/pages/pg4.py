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
import traceback
from dash import no_update

dash.register_page(__name__, path="/pg4", name='Análisis Cinético', icon="file-earmark-text")
from analisis_plantillas import procesar_archivo_plantillas
from generar_figuras_cinetico import generar_figuras_cinetico

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


# ── Encabezado compacto: inputs (izquierda) + título ───────────────────────────
header_compacto = html.Div([
    html.Div([
        dbc.InputGroup([
            dbc.Input(id="input-nombre", placeholder="Nombre", type="text",
                      size="sm", style={"flex": "0 0 100px"}),   # ⬅️ clave
            dbc.Input(id="input-edad", placeholder="Edad", type="number", min=0, step=1,
                      size="sm", style={"flex": "0 0 70px", "textAlign": "center"}),
            dbc.Input(id="input-peso", placeholder="Peso", type="number", min=0, step=0.1,
                      size="sm", style={"flex": "0 0 70px", "textAlign": "center"}),
            dbc.Button("Guardar", id="btn-save-patient", size="sm", color="primary",
                       style={"flex": "0 0 auto"})
        ], size="sm", className="me-2"),

        html.Small(id="patient-saved-msg", children="", style={"color": "#6c757d", "marginTop": "4px", "display": "block"}),
    ], style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start", "gap": "4px"}),

    html.H1("Análisis Cinético", style={"margin": "6px 0 0 0", "fontWeight": "700", "color": "#2c3e50",
                                        "lineHeight": "1.1", "textAlign": "center", "width": "100%"}),
], style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginBottom": "8px"})





layout = html.Div([
    dcc.Store(id='patient-info', storage_type='session'),
    dcc.Store(id='session-stored-cinetico', storage_type='session'),
    header_compacto,
    
    # Pestañas
    dcc.Tabs(id='tabs-cinetico', value='tab-csv', children=[
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

    ]),

    html.Div(id='tabs-content'),

])


def render_tab(tab):

    # ---- Contenido por solapa ----
    if tab == 'tab-sensores':
        # Cabecera de controles + card a la derecha
        header_row = html.Div([
            html.Div([
                html.Div([
                    html.Span("Estado de puertos:", id='port-status-label', style={'marginRight': '5px'}),
                    html.Span(id='port-status-dynamic', children="Actualizando..."),
                    html.Button("🔄", id='refresh-button', n_clicks=0, title="Actualizar", style={
                        'border': 'none','background': 'none','cursor': 'pointer',
                        'fontSize': '20px','marginLeft': '8px'
                    }),
                    html.Div(style={'flex': '1'}),
                    html.Button('Iniciar Lectura', id='start-button', n_clicks=0, disabled=True, style={
                        'backgroundColor': '#428bca','color': 'white','marginLeft': '10px'
                    }),
                    html.Button('Finalizar Lectura', id='stop-button', n_clicks=0, disabled=True, style={
                        'backgroundColor': '#428bca','color': 'white','marginLeft': '10px'
                    })
                ], style={
                    'display': 'flex','alignItems': 'center','flexWrap': 'wrap',
                    'gap': '10px','marginBottom': '10px','width': '100%'
                }),
                html.Div(id='reading-status', style={'width': '100%'})
            ], style={'flex': '1 1 auto', 'minWidth': 0})

        ], style={
            'display': 'flex','alignItems': 'flex-start',
            'justifyContent': 'space-between','gap': '16px','flexWrap': 'wrap',
            'marginBottom': '8px'
        })

        # Gráficos (si tu solapa de lectura los muestra)
        graficos_row = html.Div([
            # ← si luego querés mostrar gráficos también en esta solapa, ponelos acá
        ])

        return html.Div([header_row, graficos_row])

    elif tab == 'tab-csv':
        header_row = html.Div([
            html.Div([
                dcc.Upload(
                    id='upload-csv',
                    children=html.Div(['Arrastra o selecciona archivos para pie Izquierdo y Derecho']),
                    style={
                        'width': '100%','height': '60px','lineHeight': '60px',
                        'borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px',
                        'textAlign': 'center','marginTop': '10px','marginBottom': '10px',
                        'cursor': 'pointer','backgroundColor': 'rgba(66, 139, 202, 0.1)',
                        'borderColor': '#428bca','color': '#428bca','transition': 'all 0.3s'
                    },
                    multiple=True
                ),
                html.Div(id='csv-validation'),
            ], style={'flex': '1 1 auto', 'minWidth': 0})


        ], style={
            'display': 'flex','alignItems': 'flex-start',
            'justifyContent': 'space-between','gap': '16px','flexWrap': 'wrap',
            'marginBottom': '8px'
        })

        graficos_row = html.Div([
            # 🎯 GRÁFICOS CINÉTICOS
            html.Div([
                html.Div([ dcc.Graph(id='fig-ciclos', style={'height':'320px','width':'100%'}, config={'responsive': True}) ], className='graph-card full'),
                html.Div([ dcc.Graph(id='fig-fases',  style={'height':'480px','width':'100%'}, config={'responsive': True}) ], className='graph-card full'),
                html.Div([ dcc.Graph(id='fig-cop',    style={'height':'100%','width':'100%'}, config={'responsive': True}) ], className='graph-card full cop-card', style={'height': '600px', 'overflow': 'hidden', 'minWidth': 0}),
                html.Div([ dcc.Graph(id='fig-aporte-izq', style={'height':'360px','width':'100%'}, config={'responsive': True}) ], className='graph-card full'),
                html.Div([ dcc.Graph(id='fig-aporte-der', style={'height':'360px','width':'100%'}, config={'responsive': True}) ], className='graph-card full'),
            ], className='graph-grid')
        ])

        return html.Div([header_row, graficos_row])

    # Fallback
    return html.Div()


@callback(
    Output('patient-info', 'data'),
    Output('patient-saved-msg', 'children'),
    Output('input-nombre', 'value'),
    Output('input-edad', 'value'),
    Output('input-peso', 'value'),
    Output('store-patient-info', 'data'),
    Input('btn-save-patient', 'n_clicks'),
    State('input-nombre', 'value'),
    State('input-edad', 'value'),
    State('input-peso', 'value'),
    prevent_initial_call=True
)
def save_patient(n, nombre, edad, peso):
    data = {
        'nombre': (nombre or '').strip(),
        'edad': int(edad) if edad not in (None, "") else None,
        'peso': float(peso) if peso not in (None, "") else None
    }
    msg = f"Datos: {data['nombre'] or '—'} • {data['edad'] or '—'} años • {data['peso'] or '—'} kg" \
          if any(v not in (None, '') for v in data.values()) else "Sin datos"
    # limpiamos inputs y copiamos al store global
    return data, msg, "", None, None, data


# ---------- Callback para mostrar contenido de pestañas ----------
@callback(
    Output('tabs-content', 'children'),
    Input('tabs-cinetico', 'value'),
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
               
            else:
                print("⚠️ No se guardó archivo izquierdo (vacío).")

            if not df_der.empty:
                df_der.to_csv(directorio_guardado / f"derecha_{timestamp}.csv", index=False)
               
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
    Output('fig-ciclos', 'figure'), Output('fig-ciclos', 'style'),
    Output('fig-fases', 'figure'),  Output('fig-fases', 'style'),
    Output('fig-cop', 'figure'),    Output('fig-cop', 'style'),
    Output('fig-aporte-izq', 'figure'), Output('fig-aporte-izq', 'style'),
    Output('fig-aporte-der', 'figure'), Output('fig-aporte-der', 'style'),
    Output('store-figs-cinetico', 'data'), 
    Input('session-stored-cinetico', 'data'),
    Input('tabs-cinetico', 'value'),
    Input('patient-info', 'data'),
    State('session-stored-cinetico', 'data')
)
def actualizar_graficos_cineticos(trigger, tab_value, patient_data, stored_data):

    style_hide = {'display': 'none'}
    style_show = {'display': 'block', 'width': '100%', 'height': '480px', 'margin': 'auto'}
    style_cop  = {'display': 'block', 'width': '100%', 'height': '100%'}  # el Graph llena la card

    # 🔧 fallback de EXACTAMENTE 10 items (5 pares fig+style)
    fallback = [go.Figure(), style_hide] * 5 + [no_update]

    if tab_value != 'tab-csv' or not stored_data:
        return [go.Figure(), {'display': 'none'}] * 5 + [no_update]
    try:
        # Verificar que los datos tienen contenido válido
        if not stored_data.get('izq') or not stored_data.get('der'):
            raise ValueError("Datos incompletos")
        
        raw_izq = stored_data.get('izq')
        raw_der = stored_data.get('der')

        if not raw_izq or not raw_der:
            return fallback

        df_izq = pd.read_json(StringIO(raw_izq), orient='split')
        df_der = pd.read_json(StringIO(raw_der), orient='split')

        if df_izq.empty or df_der.empty:
            return fallback
    

        from analisis_plantillas import procesar_archivo_plantillas
        from generar_figuras_cinetico import generar_figuras_cinetico


        data_proc = procesar_archivo_plantillas(df_der, df_izq)
        
        peso_kg = None
        if isinstance(patient_data, dict):
            peso_kg = patient_data.get('peso', None)

        # Hacelo disponible para las funciones de downstream
        data_proc['paciente'] = {
            'nombre': (patient_data or {}).get('nombre'),
            'edad':   (patient_data or {}).get('edad'),
            'peso_kg': peso_kg
        }

        # Desempaquetar listas de 1 elemento (solo si corresponde)
        evitar_desempaquetar = {
            'resultados_der', 'resultados_izq',
            'copx_der', 'copy_der', 'copx_izq', 'copy_izq',
            'min_indices_der', 'min_indices_izq',
            'min_tiempos_der', 'min_tiempos_izq',
            'ciclos_apoyo_der', 'ciclos_apoyo_izq',
            'ciclos_completos_der', 'ciclos_completos_izq',
            'indices_apoyo_der', 'indices_apoyo_izq'
        }

        for key in data_proc:
            valor = data_proc[key]

            if isinstance(valor, list) and len(valor) == 1 and key not in evitar_desempaquetar:
                if isinstance(valor[0], (pd.DataFrame, list, dict)):
                    data_proc[key] = valor[0]
       
                    

        fig_ciclos, fig_fases, fig_cop, fig_aporte_izq, fig_aporte_der = generar_figuras_cinetico(data_proc)
 
        
        figs_payload = {
            "Ciclos vGRF":          {"title": fig_ciclos.layout.title.text if fig_ciclos.layout.title else "Ciclos vGRF", "json": fig_ciclos.to_json()},
            "Fases vGRF":           {"title": fig_fases.layout.title.text  if fig_fases.layout.title  else "Fases vGRF",  "json": fig_fases.to_json()},
            "Trayectoria COP":      {"title": fig_cop.layout.title.text    if fig_cop.layout.title    else "Trayectoria COP", "json": fig_cop.to_json()},
            "Aporte Izquierdo (%)": {"title": fig_aporte_izq.layout.title.text if fig_aporte_izq.layout.title else "Aporte Izquierdo (%)", "json": fig_aporte_izq.to_json()},
            "Aporte Derecho (%)":   {"title": fig_aporte_der.layout.title.text if fig_aporte_der.layout.title else "Aporte Derecho (%)", "json": fig_aporte_der.to_json()},
        }

        return [
            fig_ciclos,     style_show,
            fig_fases,      style_show,
            fig_cop,        style_cop,   
            fig_aporte_izq, style_show,
            fig_aporte_der, style_show, 
            figs_payload,
        ]

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        fig_error = go.Figure()
        fig_error.update_layout(title=f"Error: {str(e)}")
        return [fig_error, {'display': 'block'}] * 5 + [no_update]


@callback(
    Output('store-figs-cinetico-2', 'data'),
    Input('fig-ciclos', 'figure'),
    Input('fig-fases', 'figure'),
    Input('fig-cop', 'figure'),
    Input('fig-aporte-izq', 'figure'),
    Input('fig-aporte-der', 'figure'),
    prevent_initial_call=True
)
def harvest_cinetico(fig_ciclos, fig_fases, fig_cop, fig_ap_izq, fig_ap_der):
    figs = [
        ("Ciclos detectados (vGRF por paso)", fig_ciclos),
        ("vGRF normalizado – Detección de eventos y subfases", fig_fases),
        ("Trayectoria del Centro de Presión (COP)", fig_cop),
        ("Aporte porcentual por sensor – Pie Izquierdo", fig_ap_izq),
        ("Aporte porcentual por sensor – Pie Derecho", fig_ap_der),
    ]

    payload = {}
    for default_title, f in figs:
        if not f:
            continue
        # extraer título si viene en layout
        title = default_title
        t = f.get("layout", {}).get("title")
        if isinstance(t, dict):
            title = (t.get("text") or default_title).strip()
        elif isinstance(t, str) and t.strip():
            title = t.strip()
        payload[title] = {"title": title, "json": json.dumps(f)}

    return payload or no_update












