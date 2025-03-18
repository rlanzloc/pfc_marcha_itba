import base64
import datetime
import io
import dash                     # pip install dash
from dash import dcc, html, callback, Output, Input, State
from dash import dcc
from dash import html
import plotly.express as px     # pip install plotly==5.2.2
from dash import dash_table
from dash.dash_table.Format import Group
import dash_bootstrap_components as dbc 
import pandas as pd             # pip install pandas

dash.register_page(__name__, name='Reporte')


layout = dbc.Container([

    dbc.Row(
        dbc.Col([html.H1("Análisis de la Marcha",
                        className='text-center text-primary mb-4'), #centrado, en azul y con padding abajo
        ], width=12)
    ),

    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Upload(id='upload-data', 
                       children=html.Button('MOCAP')),
                html.Div(id='output-upload-data'), 
        ])], width={'size':5, 'offset':0}),
    ], justify='start'),


    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='x-axis', multi=False, value='Gait cycle',
                         options=[{'label':'Time', 'value':'Time'},
                                  {'label':'Gait cycle', 'value':'Gait cycle'}]),
        ], width={'size':5, 'offset':0, 'order':0}),
                #xs=12, sm=12, md=12, lg=5, xl=5,
        dbc.Col([
            dcc.Dropdown(id='y-axis', multi=False, value='KneeOut',
                         options=[{'label':'KneeOut', 'value':'KneeOut'},
                                  {'label':'AnkleOut', 'value':'AnkleOut'},
                                  {'label':'WaistFront', 'value':'WaistFront'}])
        ], width={'size':5, 'offset':0, 'order':1}),
        dbc.Col([    
            dcc.Graph(id='line-fig', figure={})
        ], width={'size':8, 'offset':1, 'order':2}),
                #xs=12, sm=12, md=12, lg=5, xl=5,)       
    ], justify='start'),  # Horizontal:start,center,end,between,around
    dcc.Store(id='stored-data', data=[], storage_type='session') # 'local' or 'session'     
], fluid=True)







#-----------------------FUNCTIONS-------------------------
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return dbc.Row([
        dbc.Col([
            html.Div([
                 html.H5(filename),
                 dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    page_size=15,
                    fill_width = True
                ),
                dcc.Store(id='stored-data', data=df.to_dict('records')),

                 html.Hr(),  # horizontal line
            ])
        ])
    ]) 



#------------------------CALLBACK--------------------------
@callback(Output('output-upload-data', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [parse_contents(list_of_contents, list_of_names, list_of_dates)]
        return children
    

@callback(Output('line-fig', 'figure'),
          Input('stored-data', 'data'),
          Input('x-axis', 'value'),
          Input('y-axis', 'value'),
          prevent_initial_call=True)
def create_graph1(data, xaxis, yaxis):
    dff = pd.DataFrame(data)
    fig1 = px.line(dff, x=xaxis, y=yaxis, range_x=[0,100])
    return fig1