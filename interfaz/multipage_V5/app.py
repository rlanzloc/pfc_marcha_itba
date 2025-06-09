import os
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, use_pages=True, 
               external_stylesheets=[dbc.themes.SPACELAB],
               suppress_callback_exceptions=True)



# Configuración para Bootstrap Icons
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


sidebar = dbc.Nav(
    [
        html.Div(
            [
                # Título de la aplicación - Sin línea inferior
                html.Div(
                    "Menu",
                    style={
                        "fontSize": "24px",
                        "fontWeight": "bold",
                        "color": "white",
                        "padding": "20px",
                        "textAlign": "center",
                        "backgroundColor": "#2c3e50"  # Eliminado borderBottom
                    }
                ),
                
                # Elementos del menú
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.NavLink(
                                [
                                    html.I(className=f"bi bi-{page.get('icon', 'circle')} me-2"),
                                    html.Span(page["name"])
                                ],
                                href=page["path"],
                                active="exact",
                                className="py-2 px-3 rounded",
                                style={
                                    "color": "#e0e0e0",
                                    "margin": "3px 10px",
                                    "transition": "all 0.3s",
                                    "borderLeft": "3px solid transparent"
                                }
                            )
                        )
                        for page in dash.page_registry.values()
                    ],
                    vertical=True,
                    pills=True,
                    className="px-2",
                    style={"backgroundColor": "#2c3e50"}
                )
            ]
        ),
        
        # Botón de toggle mejorado con mejor visibilidad
        html.Div(
            dbc.Button(
                html.I(className="bi bi-chevron-left"),
                id="sidebar-toggle",
                className="btn-sm",
                style={
                    "position": "absolute",
                    "right": "-15px",
                    "top": "20px",
                    "zIndex": "1100",
                    "width": "30px",
                    "height": "30px",
                    "borderRadius": "50%",
                    "backgroundColor": "#2c3e50",
                    "color": "white",
                    "border": "1px solid rgba(255,255,255,0.3)",  # Borde más visible
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",  # Sombra más pronunciada
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "padding": "0"
                },
                n_clicks=0
            )
        )
    ],
    vertical=True,
    style={
        "height": "100vh",
        "width": "250px",
        "position": "fixed",
        "zIndex": "1000",
        "boxShadow": "2px 0 10px rgba(0, 0, 0, 0.1)",
        "padding": "0",
        "borderRight": "1px solid rgba(255,255,255,0.1)",
        "backgroundColor": "#2c3e50"
    }
)

# Layout principal
app.layout = dbc.Container(
    [
        # Contenedor del sidebar
        html.Div(
            sidebar,
            id="sidebar-container",
            style={
                "transition": "all 0.3s",
                "marginLeft": "0"
            }
        ),
        
        # Contenedor del contenido principal
        html.Div(
            [
                dash.page_container
            ],
            id="content-container",
            style={
                "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                "marginLeft": "250px",
                "padding": "30px",
                "minHeight": "100vh",
                "backgroundColor": "#f8f9fa"
            }
        )
    ],
    fluid=True,
    style={"padding": "0", "overflowX": "hidden"}
)

# CSS personalizado mejorado
app.clientside_css = [{
    'selector': '.nav-pills .nav-link.active',
    'rule': '''
        background-color: rgba(66, 139, 202, 0.2) !important;
        color: white !important;
        font-weight: 500;
        border-left: 3px solid #428bca !important;
    '''
},
{
    'selector': '.nav-pills .nav-link:hover:not(.active)',
    'rule': '''
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    '''
}]

# Callback mejorado para el sidebar
@app.callback(
    [Output("sidebar-container", "style"),
     Output("content-container", "style"),
     Output("sidebar-toggle", "children"),
     Output("sidebar-toggle", "style")],
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar-container", "style"),
     State("content-container", "style"),
     State("sidebar-toggle", "style")]
)
def toggle_sidebar(n, sidebar_style, content_style, btn_style):
    if n is None:
        n = 0
    
    if n % 2 == 1:  # Sidebar cerrado
        sidebar_style.update({
            "marginLeft": "-250px",
            "boxShadow": "none"
        })
        content_style.update({
            "marginLeft": "0",
            "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
        })
        icon = html.I(className="bi bi-chevron-right")
        # Flecha más visible con fondo blanco y color azul oscuro
        # En el callback, dentro del if n % 2 == 1 (sidebar cerrado), cambia:
        btn_style.update({
            "color": "white",  # Cambia de "#2c3e50" a "white"
            "backgroundColor": "#2c3e50",  # Fondo oscuro para contraste
            "border": "1px solid rgba(255,255,255,0.3)",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.3)"
        })
    else:  # Sidebar abierto
        sidebar_style.update({
            "marginLeft": "0",
            "boxShadow": "2px 0 10px rgba(0, 0, 0, 0.1)"
        })
        content_style.update({
            "marginLeft": "250px",
            "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
        })
        icon = html.I(className="bi bi-chevron-left")
        btn_style.update({
            "color": "white",
            "backgroundColor": "#2c3e50",
            "border": "1px solid rgba(255,255,255,0.3)"
        })
    
    return sidebar_style, content_style, icon, btn_style

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto que Render asigna
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)


#if __name__ == "__main__":
#    port = int(os.environ.get("PORT", 5000))  # Usa el puerto que Render asigna
#    app.run(host="0.0.0.0", port=port, debug=True)

