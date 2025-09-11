import os
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, use_pages=True, 
               external_stylesheets=[dbc.themes.SPACELAB],
               suppress_callback_exceptions=True)

from pages import pg4  # Análisis Cinético: usa tabs-cinetico, session-stored-cinetico, fig-*
from pages import analysis  # Análisis Cinemático
from pages import report  # Reporte
from pages import pg1 


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
                    "Menú",
                    style={
                        "fontSize": "24px",
                        "fontWeight": "bold",
                        "color": "white",
                        "padding": "20px",
                        "textAlign": "center",
                        "backgroundColor": "#2c3e50"
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
        
        # Botón de toggle
        html.Div(
            dbc.Button(
                html.I(className="bi bi-chevron-left"),
                id="sidebar-toggle",
                className="btn-sm",
                style={
                    "position": "fixed",
                    "left": "calc(250px - 12px)",
                    "top": "20px",
                    "zIndex": "1100",
                    "width": "28px",
                    "height": "28px",
                    "borderRadius": "50%",
                    "backgroundColor": "#2c3e50",
                    "color": "white",
                    "border": "1px solid rgba(255,255,255,0.3)",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
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
    id="sidebar",
    className="",
    style={
        "padding": "0",
        "borderRight": "1px solid rgba(255,255,255,0.1)",
        "backgroundColor": "#2c3e50",
        "width": "250px",             # ancho fijo del sidebar
        "position": "fixed",
        "top": "0",
        "left": "0",
        "height": "100vh",
        "zIndex": "1000",
        "boxShadow": "2px 0 10px rgba(0, 0, 0, 0.1)",
        "transition": "width 0.3s ease", 
        "overflow": "hidden",
    }
)



# Layout principal
app.layout = dbc.Container(
    [
        # Stores
        dcc.Store(id="__force_resize_token"),  # NEW: para forzar relayout Plotly
        dcc.Store(id='store-figs-cinetico', storage_type='session'),
        dcc.Store(id='store-figs-cinetico-2', storage_type='session'),
        dcc.Store(id='store-figs-cinematico', storage_type='session'),
        dcc.Store(id='store-patient-info', storage_type='session'),
        dcc.Store(id='session-stored-cinetico', storage_type='session'),
        dcc.Store(id="session-stored-data", storage_type="session"),

        

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
                # NEW: envolvemos el page_container para ancho max + bordes blancos
                html.Div(
                    dash.page_container,
                    className="container-max"  # ← definido en assets/00_layout.css
                )
            ],
            id="content-container",
            style={
                "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                "marginLeft": "250px",      # el callback alterna 250px ↔ 0
                "padding": "30px",
                "minHeight": "100vh",
                "backgroundColor": "#f8f9fa"
            }
        ),
        
    ],
    fluid=True,
    style={"padding": "0", "overflowX": "hidden"},

)


app.validation_layout = html.Div([
    
    app.layout, 
    pg4.layout,
    analysis.layout,
    report.layout,
    pg1.layout, 
    
    # Stores globales
    dcc.Store(id="__force_resize_token"),
    dcc.Store(id="store-figs-cinetico", storage_type="session"),
    dcc.Store(id="store-figs-cinetico-2", storage_type="session"),
    dcc.Store(id="store-figs-cinematico", storage_type="session"),
    dcc.Store(id="store-patient-info", storage_type="session"),
    dcc.Store(id="session-stored-cinetico", storage_type="session"),

    # ---- Cinético (/pg4) ----
    dcc.Tabs(id="tabs-cinetico", value="tab-csv"),
    html.Div(id="tabs-content"),
    dcc.Upload(id="upload-csv"),
    html.Div(id="csv-validation"),
    html.Span(id="port-status-label"),
    html.Span(id="port-status-dynamic"),
    html.Div(id="reading-status"),
    html.Button(id="start-button"),
    html.Button(id="stop-button"),
    html.Button(id="refresh-button"),

    # ---- Cinemático (/analysis) ----
    dcc.Upload(id="upload-c3d"),
    html.Div(id="output-c3d-upload"),
    html.Div(id="graphs-container"),
    html.Div(id="parametros-container"),
    dcc.Store(id="stored-data"),
    dcc.Store(id="session-stored-data", storage_type="session"),
    html.Div(id="tabs-container"),
    dcc.RadioItems(id="lado-dropdown"),

    # ---- Reporte (/reporte) ----
    dcc.Checklist(id="reporte-secciones"),
    html.Button(id="btn-gen-pdf"),
    html.Div(id="report-status"),
    dcc.Download(id="download-pdf"),
], style={"display": "none"})



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

# Callback para el sidebar (abre/cierra)
@app.callback(
    [
        Output("sidebar", "style"),            # cambiamos el ancho real del sidebar
        Output("content-container", "style"),  # y el margen del contenido
        Output("sidebar-toggle", "children"),
        Output("sidebar-toggle", "style"),     # ← movemos el botón aquí
    ],
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar", "style"),
    State("content-container", "style"),
    State("sidebar-toggle", "style"),
)
def toggle_sidebar(n, sidebar_style, content_style, btn_style):
    sidebar_style = dict(sidebar_style or {})
    content_style = dict(content_style or {})
    btn_style     = dict(btn_style or {})

    OPEN_W   = "250px"
    CLOSED_W = "15px"
    OVERLAP  = 10  # px que el botón sobresale del borde

    n = n or 0
    if n % 2 == 1:
        # CERRADO
        sidebar_style.update({"width": CLOSED_W})
        content_style.update({"marginLeft": CLOSED_W})
        icon = html.I(className="bi bi-chevron-right")
        btn_style.update({
            "position": "fixed",
            "left": f"calc({CLOSED_W} - {OVERLAP}px)",  # ← mueve el botón
            "top": "20px",
            "zIndex": "1100",
            "backgroundColor": "#2c3e50",
            "color": "white",
            "border": "1px solid rgba(255,255,255,0.3)",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.25)"
        })
    else:
        # ABIERTO
        sidebar_style.update({"width": OPEN_W})
        content_style.update({"marginLeft": OPEN_W})
        icon = html.I(className="bi bi-chevron-left")
        btn_style.update({
            "position": "fixed",
            "left": f"calc({OPEN_W} - {OVERLAP}px)",
            "top": "20px",
            "zIndex": "1100",
            "backgroundColor": "#2c3e50",
            "color": "white",
            "border": "1px solid rgba(255,255,255,0.3)",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.25)"
        })

    # asegurar props clave del sidebar por si se pierden
    sidebar_style.setdefault("position", "fixed")
    sidebar_style.setdefault("left", "0")
    sidebar_style.setdefault("top", "0")
    sidebar_style.setdefault("height", "100vh")
    sidebar_style.setdefault("zIndex", "1000")
    sidebar_style.setdefault("overflow", "hidden")
    sidebar_style.setdefault("transition", "width 0.3s ease")
    sidebar_style.setdefault("backgroundColor", "#2c3e50")
    sidebar_style.setdefault("boxShadow", "2px 0 10px rgba(0, 0, 0, 0.1)")
    sidebar_style.setdefault("borderRight", "1px solid rgba(255,255,255,0.1)")

    content_style.setdefault("transition", "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)")
    content_style.setdefault("padding", "30px")
    content_style.setdefault("minHeight", "100vh")
    content_style.setdefault("backgroundColor", "#f8f9fa")

    return sidebar_style, content_style, icon, btn_style


app.clientside_callback(
    """
    function(token) {
        if (token > 0) {
            setTimeout(function() {
                window.dispatchEvent(new Event('resize'));
            }, 150);
        }
        return token;
    }
    """,
    Output("__force_resize_token", "data", allow_duplicate=True),
    Input("__force_resize_token", "data"),
    prevent_initial_call=True
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)

