<<<<<<< HEAD
import os
=======
>>>>>>> aa826df3f144f694e80961dcb492492d9a19d734
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

<<<<<<< HEAD
app = dash.Dash(__name__, suppress_callback_exceptions=True, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB])
=======
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions=True)

   

# Sidebar (menú lateral)
>>>>>>> aa826df3f144f694e80961dcb492492d9a19d734
sidebar = dbc.Nav(
    [
        dbc.NavLink(
            [
                html.Div(page["name"], className="ms-2"),
            ],
            href=page["path"],
            active="exact",
        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    className="bg-light",
    style={
        "height": "100vh",
        "padding": "20px 0",
        "transition": "all 0.3s",
        "width": "250px",
        "position": "fixed",
        "zIndex": "1000"
    }
)

# Layout principal
app.layout = dbc.Container(
    [
        # Botón de toggle fuera del sidebar para mejor accesibilidad
        dbc.Button(
            "☰",
            id="sidebar-toggle",
            className="position-fixed",
            style={
                "left": "10px",
                "top": "10px",
                "zIndex": "1100",
                "fontSize": "20px",
                "width": "40px",
                "height": "40px"
            },
            n_clicks=0
        ),
        
        # Contenedor del sidebar
        html.Div(
            sidebar,
            id="sidebar-container",
            style={
                "transition": "all 0.3s",
                "marginLeft": "0px"
            }
        ),
        
        # Contenedor del contenido principal
        html.Div(
            [
     
                
                # Contenido de la página
                dash.page_container
            ],
            id="content-container",
            style={
                "transition": "all 0.3s",
                "marginLeft": "250px",
                "padding": "20px"
            }
        )
    ],
    fluid=True,
    style={"padding": "0", "overflowX": "hidden"}
)

# Callback para mostrar/ocultar el sidebar
@app.callback(
    [Output("sidebar-container", "style"),
     Output("content-container", "style"),
     Output("sidebar-toggle", "children")],
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar-container", "style"),
     State("content-container", "style")]
)
def toggle_sidebar(n, sidebar_style, content_style):
    if n is None:
        n = 0
    
    if n % 2 == 1:  # Sidebar cerrado
        sidebar_style.update({"marginLeft": "-250px"})
        content_style.update({"marginLeft": "0"})
        icon = "»"  # Icono de menú
    else:  # Sidebar abierto
        sidebar_style.update({"marginLeft": "0"})
        content_style.update({"marginLeft": "250px"})
        icon = "«"  # Icono de cierre
    
    return sidebar_style, content_style, icon

if __name__ == "__main__":
<<<<<<< HEAD
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto que Render asigna
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
=======
    app.run(debug=True)

#if __name__ == "__main__":
#    port = int(os.environ.get("PORT", 5000))  # Usa el puerto que Render asigna
#    app.run(host="0.0.0.0", port=port, debug=True)
>>>>>>> aa826df3f144f694e80961dcb492492d9a19d734
