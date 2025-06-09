<<<<<<< HEAD
import dash
from dash import dcc, html, callback, Output, Input, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import base64, io, os, serial, threading, datetime
import plotly.graph_objects as go
from pathlib import Path


dash.register_page(__name__, name='Arduino read')

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
puerto_disponible = arduino_der is not None or arduino_izq is not None

# -------- Variables y directorio -----------------
# Ruta del archivo actual (por ejemplo, app.py)
ruta_actual = Path(__file__).resolve().parent

# Ruta a la carpeta 'assets'
directorio_guardado = ruta_actual / "assets" / "datos_sensores"

# Crear la carpeta si no existe
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

# Hilos
if arduino_der:
    threading.Thread(target=leer_datos, args=(arduino_der, df_der, "Derecha"), daemon=True).start()
if arduino_izq:
    threading.Thread(target=leer_datos, args=(arduino_izq, df_izq, "Izquierda"), daemon=True).start()

# ------------- Layout ----------------
layout = html.Div([
    html.H2("Lectura de Sensores o CSVs"),
    dcc.Tabs(id='tabs', value='tab-sensores', children=[
        dcc.Tab(label='Lectura en Vivo (Sensores)', value='tab-sensores'),
        dcc.Tab(label='Importar CSV', value='tab-csv')
    ]),
    html.Div(id='tabs-content'),
    dcc.Store(id='data-izq-store'),
    dcc.Store(id='data-der-store')
])

# ------------- Renderizado de pestañas -------------
@callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-sensores':
        return html.Div([
            html.Div("Estado de puertos: " + ("Disponible ✅" if puerto_disponible else "No disponible ❌")),
            html.Button('Iniciar Lectura', id='start-button', n_clicks=0, disabled=not puerto_disponible),
            html.Button('Finalizar Lectura', id='stop-button', n_clicks=0, disabled=not puerto_disponible),
            html.Div(id='output-state', children='Esperando acción...')
        ])
    elif tab == 'tab-csv':
        return html.Div([
            dcc.Upload(id='upload-csv', children=html.Div(['Arrastra o selecciona archivos izquierdo y derecho']),
                       style={'width': '45%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                              'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
                              'margin': '10px'}, multiple=True),
            html.Div(id='csv-validation'),
            html.Hr(),
            dcc.Graph(id='grafico-izquierda'),
            dcc.Graph(id='grafico-derecha'),
        ])

# -------- Callbacks de control de sensores --------
@callback(Output('output-state', 'children'),
          [Input('start-button', 'n_clicks'), Input('stop-button', 'n_clicks')])
def control_lectura(n_start, n_stop):
    global reading, df_der, df_izq
    if not puerto_disponible:
        return 'Puertos no disponibles.'
    if ctx.triggered_id == 'start-button':
        reading = True
        return '📡 Lectura iniciada...'
    elif ctx.triggered_id == 'stop-button':
        reading = False
        with lock:
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
            file_der = os.path.join(directorio_guardado, f'datos_derecha_{timestamp}.csv')
            file_izq = os.path.join(directorio_guardado, f'datos_izquierda_{timestamp}.csv')
            df_der.to_csv(file_der, index=False)
            df_izq.to_csv(file_izq, index=False)
        return f'🛑 Lectura finalizada. Archivos guardados:\n{file_der}\n{file_izq}'
    return 'Esperando acción...'

@callback(Output('start-button', 'disabled'),
          Input('start-button', 'n_clicks'),
          Input('stop-button', 'n_clicks'))
def toggle_start(n1, n2):
    return not puerto_disponible or reading

@callback(Output('stop-button', 'disabled'),
          Input('start-button', 'n_clicks'),
          Input('stop-button', 'n_clicks'))
def toggle_stop(n1, n2):
    return not puerto_disponible or not reading

# ------- Callback de lectura de CSVs --------
@callback(
    Output('csv-validation', 'children'),
    Output('data-izq-store', 'data'),
    Output('data-der-store', 'data'),
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename')
)
def handle_csv_upload(contents, filename):
    # Definir las columnas esperadas
    expected_cols_izq = ['Hora', 'Tiempo'] + [f'Izquierda_S{i+1}' for i in range(8)]
    expected_cols_der = ['Hora', 'Tiempo'] + [f'Derecha_S{i+1}' for i in range(8)]
    errores = []
    mensajes = []
    df_izq_csv = None
    df_der_csv = None

    def parse_contents(contents, filename, expected_columns, label):
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Verificar columnas
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                errores.append(f"❌ {filename} no tiene las siguientes columnas necesarias para {label}: {', '.join(missing_cols)}.")
                return None
            else:
                mensajes.append(f"✅ {filename} subido correctamente.")
                return df  # <-- ESTA LÍNEA FALTABA
        except Exception as e:
            errores.append(f"❌ Error al procesar {label}: {str(e)}")
            return None


    # Revisar los archivos subidos y procesarlos
    if contents:
        for i, content in enumerate(contents):
            if "izquierda" in filename[i].lower():
                df_izq_csv = parse_contents(content, filename[i], expected_cols_izq, "Izquierda")
            elif "derecha" in filename[i].lower():
                df_der_csv = parse_contents(content, filename[i], expected_cols_der, "Derecha")
    
    # Devolver los resultados
    return (
        html.Div([html.P(error) for error in errores] + [html.P(msg) for msg in mensajes]),
        df_izq_csv.to_dict('records') if df_izq_csv is not None else None,
        df_der_csv.to_dict('records') if df_der_csv is not None else None
    )
    
@callback(
    Output('grafico-izquierda', 'figure'),
    Output('grafico-derecha', 'figure'),
    Input('data-izq-store', 'data'),
    Input('data-der-store', 'data')
)
def graficar_sensores(data_izq, data_der):
    

    fig_izq = go.Figure()
    fig_der = go.Figure()

    if data_izq:
        df_izq = pd.DataFrame(data_izq)
        df_izq.set_index('Tiempo', inplace=True)
        for i in range(8):
            fig_izq.add_trace(go.Scatter(
                x=df_izq.index,
                y=df_izq[f'Izquierda_S{i+1}'],
                mode='lines',
                name=f'Izquierda S{i+1}'
            ))
        fig_izq.update_layout(title='Sensores Pie Izquierdo', xaxis_title='Tiempo', yaxis_title='Valor FSR')

    if data_der:
        df_der = pd.DataFrame(data_der)
        df_der.set_index('Tiempo', inplace=True)
        for i in range(8):
            fig_der.add_trace(go.Scatter(
                x=df_der.index,
                y=df_der[f'Derecha_S{i+1}'],
                mode='lines',
                name=f'Derecha S{i+1}'
            ))
        fig_der.update_layout(title='Sensores Pie Derecho', xaxis_title='Tiempo', yaxis_title='Valor FSR')

    return fig_izq, fig_der
=======
import dash
from dash import dcc, html, callback, Output, Input, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import base64, io, os, serial, threading, datetime
import plotly.graph_objects as go

dash.register_page(__name__, name='Arduino read')

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
puerto_disponible = arduino_der is not None or arduino_izq is not None

# -------- Variables y directorio -----------------
directorio_guardado = "C:/Users/Rashel Lanz Lo Curto/Desktop/datos_sensores"
os.makedirs(directorio_guardado, exist_ok=True)

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

# Hilos
if arduino_der:
    threading.Thread(target=leer_datos, args=(arduino_der, df_der, "Derecha"), daemon=True).start()
if arduino_izq:
    threading.Thread(target=leer_datos, args=(arduino_izq, df_izq, "Izquierda"), daemon=True).start()

# ------------- Layout ----------------
layout = html.Div([
    html.H2("Lectura de Sensores o CSVs"),
    dcc.Tabs(id='tabs', value='tab-sensores', children=[
        dcc.Tab(label='Lectura en Vivo (Sensores)', value='tab-sensores'),
        dcc.Tab(label='Importar CSV', value='tab-csv')
    ]),
    html.Div(id='tabs-content'),
    dcc.Store(id='data-izq-store'),
    dcc.Store(id='data-der-store')
])

# ------------- Renderizado de pestañas -------------
@callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-sensores':
        return html.Div([
            html.Div("Estado de puertos: " + ("Disponible ✅" if puerto_disponible else "No disponible ❌")),
            html.Button('Iniciar Lectura', id='start-button', n_clicks=0, disabled=not puerto_disponible),
            html.Button('Finalizar Lectura', id='stop-button', n_clicks=0, disabled=not puerto_disponible),
            html.Div(id='output-state', children='Esperando acción...')
        ])
    elif tab == 'tab-csv':
        return html.Div([
            dcc.Upload(id='upload-csv', children=html.Div(['Arrastra o selecciona archivos izquierdo y derecho']),
                       style={'width': '45%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                              'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
                              'margin': '10px'}, multiple=True),
            html.Div(id='csv-validation'),
            html.Hr(),
            dcc.Graph(id='grafico-izquierda'),
            dcc.Graph(id='grafico-derecha'),
        ])

# -------- Callbacks de control de sensores --------
@callback(Output('output-state', 'children'),
          [Input('start-button', 'n_clicks'), Input('stop-button', 'n_clicks')])
def control_lectura(n_start, n_stop):
    global reading, df_der, df_izq
    if not puerto_disponible:
        return 'Puertos no disponibles.'
    if ctx.triggered_id == 'start-button':
        reading = True
        return '📡 Lectura iniciada...'
    elif ctx.triggered_id == 'stop-button':
        reading = False
        with lock:
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
            file_der = os.path.join(directorio_guardado, f'datos_derecha_{timestamp}.csv')
            file_izq = os.path.join(directorio_guardado, f'datos_izquierda_{timestamp}.csv')
            df_der.to_csv(file_der, index=False)
            df_izq.to_csv(file_izq, index=False)
        return f'🛑 Lectura finalizada. Archivos guardados:\n{file_der}\n{file_izq}'
    return 'Esperando acción...'

@callback(Output('start-button', 'disabled'),
          Input('start-button', 'n_clicks'),
          Input('stop-button', 'n_clicks'))
def toggle_start(n1, n2):
    return not puerto_disponible or reading

@callback(Output('stop-button', 'disabled'),
          Input('start-button', 'n_clicks'),
          Input('stop-button', 'n_clicks'))
def toggle_stop(n1, n2):
    return not puerto_disponible or not reading

# ------- Callback de lectura de CSVs --------
@callback(
    Output('csv-validation', 'children'),
    Output('data-izq-store', 'data'),
    Output('data-der-store', 'data'),
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename')
)
def handle_csv_upload(contents, filename):
    # Definir las columnas esperadas
    expected_cols_izq = ['Hora', 'Tiempo'] + [f'Izquierda_S{i+1}' for i in range(8)]
    expected_cols_der = ['Hora', 'Tiempo'] + [f'Derecha_S{i+1}' for i in range(8)]
    errores = []
    mensajes = []
    df_izq_csv = None
    df_der_csv = None

    def parse_contents(contents, filename, expected_columns, label):
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Verificar columnas
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                errores.append(f"❌ {filename} no tiene las siguientes columnas necesarias para {label}: {', '.join(missing_cols)}.")
                return None
            else:
                mensajes.append(f"✅ {filename} subido correctamente.")
                return df  # <-- ESTA LÍNEA FALTABA
        except Exception as e:
            errores.append(f"❌ Error al procesar {label}: {str(e)}")
            return None


    # Revisar los archivos subidos y procesarlos
    if contents:
        for i, content in enumerate(contents):
            if "izquierda" in filename[i].lower():
                df_izq_csv = parse_contents(content, filename[i], expected_cols_izq, "Izquierda")
            elif "derecha" in filename[i].lower():
                df_der_csv = parse_contents(content, filename[i], expected_cols_der, "Derecha")
    
    # Devolver los resultados
    return (
        html.Div([html.P(error) for error in errores] + [html.P(msg) for msg in mensajes]),
        df_izq_csv.to_dict('records') if df_izq_csv is not None else None,
        df_der_csv.to_dict('records') if df_der_csv is not None else None
    )
    
@callback(
    Output('grafico-izquierda', 'figure'),
    Output('grafico-derecha', 'figure'),
    Input('data-izq-store', 'data'),
    Input('data-der-store', 'data')
)
def graficar_sensores(data_izq, data_der):
    

    fig_izq = go.Figure()
    fig_der = go.Figure()

    if data_izq:
        df_izq = pd.DataFrame(data_izq)
        df_izq.set_index('Tiempo', inplace=True)
        for i in range(8):
            fig_izq.add_trace(go.Scatter(
                x=df_izq.index,
                y=df_izq[f'Izquierda_S{i+1}'],
                mode='lines',
                name=f'Izquierda S{i+1}'
            ))
        fig_izq.update_layout(title='Sensores Pie Izquierdo', xaxis_title='Tiempo', yaxis_title='Valor FSR')

    if data_der:
        df_der = pd.DataFrame(data_der)
        df_der.set_index('Tiempo', inplace=True)
        for i in range(8):
            fig_der.add_trace(go.Scatter(
                x=df_der.index,
                y=df_der[f'Derecha_S{i+1}'],
                mode='lines',
                name=f'Derecha S{i+1}'
            ))
        fig_der.update_layout(title='Sensores Pie Derecho', xaxis_title='Tiempo', yaxis_title='Valor FSR')

    return fig_izq, fig_der
>>>>>>> aa826df3f144f694e80961dcb492492d9a19d734
