import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import json, io
import plotly.io as pio

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
import copy
import datetime




APP_BLUE = "#2c3e50"   # mismo azul que usás en los H1 de las páginas
ACCENT   = "#428bca"   # acento opcional (botones, bordes)


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="AppTitle",
        parent=styles["Title"],
        textColor=colors.HexColor(APP_BLUE),
        fontSize=22,
        leading=26
    ))
    styles.add(ParagraphStyle(
        name="AppH1",
        parent=styles["Heading1"],
        textColor=colors.HexColor(ACCENT),
        fontSize=16,
        leading=20,
        spaceBefore=8,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="AppText",
        parent=styles["Normal"],
        textColor=colors.HexColor(APP_BLUE),
        fontSize=11,
        leading=14
    ))
    styles.add(ParagraphStyle(
        name="SmallMuted",
        parent=styles["Italic"],
        textColor=colors.HexColor("#6c757d"),
        fontSize=9,
        leading=12
    ))
    styles.add(ParagraphStyle(
        name="MetricLabel",
        parent=styles["Normal"],
        textColor=colors.HexColor(APP_BLUE),
        fontSize=11,
        leading=14,
        spaceBefore=2,
        spaceAfter=0
    ))
    # Valor de métrica (línea completa con Der/Izq/Global)
    styles.add(ParagraphStyle(
        name="MetricValue",
        parent=styles["Normal"],
        textColor=colors.HexColor(APP_BLUE),
        fontSize=11,
        leading=14,
        leftIndent=10,
        spaceBefore=0,
        spaceAfter=4
    ))
    return styles


def _style_fig_for_pdf(fig):
    """Clona y ajusta la figura para que no se superpongan título y leyenda en PDF."""
    f = copy.deepcopy(fig)

    # 1) Colores/tema básicos coherentes con la app
    f.update_layout(
        font=dict(color=APP_BLUE, family="Arial", size=12),
        title=dict(
            font=dict(color=APP_BLUE, size=16),
            x=0.5, xanchor="center",
            pad=dict(t=6, b=8)   # separa de la leyenda
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        # 2) Márgenes generosos para imagen
        margin=dict(l=30, r=30, t=70, b=90),
        # 3) Leyenda abajo, horizontal, sin caja
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.15,   # debajo del área de trazado
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color=APP_BLUE, size=11),
            itemclick="toggleothers", itemdoubleclick="toggle"
        )
    )

    # 4) Evitar recortes de ejes/etiquetas
    f.update_xaxes(automargin=True, title_standoff=6, ticklabelposition="outside")
    f.update_yaxes(automargin=True, title_standoff=6, ticklabelposition="outside")

    # 5) Si hay subplots, bajar los títulos de subplots
    if getattr(f.layout, "annotations", None):
        for ann in f.layout.annotations:
            if getattr(ann, "xref", "") == "x domain" and getattr(ann, "yref", "") == "y domain":
                ann.y = 1.08  # un poco más arriba del trace para liberar espacio
                ann.font = dict(color=APP_BLUE, size=12)
    return f

def _fmt_num(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x) if x is not None else "—"

# Map opcional de unidades si tus claves coinciden
_METRIC_UNITS = {
    "cadencia": "pasos/min",
    "velocidad": "m/s",
    "long_paso": "m",
    "longitud_paso": "m",
    "long_zancada": "m",
    "tiempo_ciclo": "s",
    "stance": "%", "swing": "%", "doble_apoyo": "%"
}

# Map opcional de alias bonitos (titulo)
_METRIC_TITLES = {
    "cadencia": "Cadencia",
    "velocidad": "Velocidad",
    "long_paso": "Longitud de paso",
    "longitud_paso": "Longitud de paso",
    "long_zancada": "Longitud de zancada",
    "tiempo_ciclo": "Tiempo de ciclo",
    "stance": "Apoyo (stance)",
    "swing": "Balanceo (swing)",
    "doble_apoyo": "Doble apoyo"
}

def _render_esp_temp_cards(data_dict, styles):
    """
    Genera una lista de Paragraph/Spacer para mostrar variables espaciotemporales
    como en la solapa: sin tablas, con una línea por métrica.
    Estructuras soportadas:
      - {'cadencia': 108.2, 'velocidad': 1.15, 'long_paso': {'derecho': 0.65, 'izquierdo': 0.63, 'global': 0.64}, ...}
      - {'promedios': {...}, 'derecho': {...}, 'izquierdo': {...}, 'global': {...}}  -> se reconcilia
    """
    flows = []
    if not isinstance(data_dict, dict) or not data_dict:
        flows.append(Paragraph("No hay variables espaciotemporales disponibles.", styles["Italic"]))
        return flows

    # Si viene separada por lados/promedios, aplanamos a un esquema común:
    flattened = {}

    if any(k in data_dict for k in ("derecho", "izquierdo", "global", "promedios")):
        # Re-construimos por variable
        keys = set()
        for side_key in ("derecho", "izquierdo", "global", "promedios"):
            side = data_dict.get(side_key) or {}
            if isinstance(side, dict):
                keys.update(side.keys())
        for var in keys:
            v = {}
            der = (data_dict.get("derecho") or {}).get(var)
            izq = (data_dict.get("izquierdo") or {}).get(var)
            glo = (data_dict.get("global") or {}).get(var)
            pro = (data_dict.get("promedios") or {}).get(var)
            if der is not None: v["derecho"] = der
            if izq is not None: v["izquierdo"] = izq
            if glo is not None: v["global"] = glo
            elif pro is not None: v["global"] = pro
            flattened[var] = v if v else None
    else:
        flattened = data_dict  # ya viene aplanado por variable

    # Render línea por línea
    for raw_key, value in flattened.items():
        key = str(raw_key)
        title = _METRIC_TITLES.get(key, key.replace("_", " ").capitalize())
        unit = _METRIC_UNITS.get(key, "")

        flows.append(Paragraph(f"<b>{title}</b>", styles["MetricLabel"]))

        if isinstance(value, dict):
            der = _fmt_num(value.get("derecho"))
            izq = _fmt_num(value.get("izquierdo"))
            glo = _fmt_num(value.get("global") or value.get("promedio") or value.get("media"))
            # Línea de valores (chips simples con bullets)
            line_parts = []
            if der != "—":
                line_parts.append(f"Derecho: <b>{der}</b>{' ' + unit if unit else ''}")
            if izq != "—":
                line_parts.append(f"Izquierdo: <b>{izq}</b>{' ' + unit if unit else ''}")
            if glo != "—":
                line_parts.append(f"Global: <b>{glo}</b>{' ' + unit if unit else ''}")
            if not line_parts:
                line_parts.append("—")
            flows.append(Paragraph("  •  ".join(line_parts), styles["MetricValue"]))
        else:
            glo = _fmt_num(value)
            flows.append(Paragraph(f"Global: <b>{glo}</b>{' ' + unit if unit else ''}", styles["MetricValue"]))

    flows.append(Spacer(1, 8))
    return flows


dash.register_page(__name__, path="/reporte", name="Reporte", icon="file-earmark-pdf")

layout = html.Div(
    [
        dcc.Store(id="session-stored-data", storage_type="session"),
        dcc.Store(id="store-figs-esp-temp", storage_type="session"),
         
        html.H1("Reporte: Análisis de Marcha", style={"margin":"6px 0 12px 0","textAlign":"center","fontWeight":"700","color":"#2c3e50"}),
        html.Div(
            [
                dcc.Checklist(
                    id="reporte-secciones",
                    options=[
                        {"label":" Incluir análisis cinético", "value":"cinetico"},
                        {"label":" Incluir análisis cinemático", "value":"cinematico"},
                        {"label":" Incluir variables espaciotemporales", "value":"esp_temp"},
                        {"label":" Incluir encabezado con datos del usuario", "value":"paciente"},
                    ],
                    value=["cinetico","cinematico","esp_temp","paciente"],
                    inputStyle={"marginRight":"6px"},
                    labelStyle={"display":"inline-flex","alignItems":"center","margin":"6px 12px"},
                    style={
                        "display":"flex",
                        "justifyContent":"center",
                        "flexWrap":"wrap",
                        "gap":"8px",
                        "marginBottom":"10px"
                    }
                ),
                
                 html.Div(
                    [
                        html.Label("Nombre del archivo PDF", style={"marginRight":"8px"}),
                        dcc.Input(
                            id="pdf-filename",
                            type="text",
                            value="reporte_marcha.pdf",
                            placeholder="reporte_marcha.pdf",
                            style={"width":"260px"}
                        )
                    ],
                    style={"display":"flex","justifyContent":"center","alignItems":"center","gap":"8px","margin":"8px 0 12px"}
                ),
                
                html.Div(
                    html.Button(
                        "Generar PDF",
                        id="btn-gen-pdf",
                        style={
                            "backgroundColor":"#2c3e50","color":"white","border":"none",
                            "padding":"8px 14px","borderRadius":"6px","cursor":"pointer","fontWeight":"600"
                        }
                    ),
                    style={"textAlign":"center"}
                ),
                
                html.Div(id="report-status", style={"marginTop":"10px"})
            ],
            style={"maxWidth":"520px","margin":"0 auto"}
        ),

        # Descarga
        dcc.Download(id="download-pdf"),
    ],
    style={"padding":"10px"}
)

def _figs_from_store(store_data):
    """Convierte dict con JSONs de figuras a lista de (titulo, fig_obj)."""
    out = []
    if not store_data:
        return out
    for key, payload in store_data.items():
        # payload puede ser {"title": "...", "json": "..."} o directamente json
        if isinstance(payload, dict) and "json" in payload:
            title = payload.get("title") or key
            fig = pio.from_json(payload["json"])
        else:
            title = key
            fig = pio.from_json(payload)
        # Si el fig ya trae título, usalo
        title_from_layout = None
        try:
            title_from_layout = (fig.layout.title.text or "").strip()
        except Exception:
            pass
        final_title = title_from_layout or title
        out.append((final_title, fig))
    return out

@callback(
    Output("download-pdf", "data"),
    Output("report-status", "children"),
    Input("btn-gen-pdf", "n_clicks"),
    State("reporte-secciones", "value"),
    State("store-figs-cinetico-2", "data"),
    State("store-figs-cinematico", "data"),
    State("session-stored-data", "data"),
    State("store-figs-esp-temp", "data"),    
    State("store-patient-info", "data"),
    State("pdf-filename", "value"),
    prevent_initial_call=True
)
def generar_pdf(n_clicks, secciones, store_cinetico, store_cinematico,
                session_data, store_esp_temp, patient, pdf_name):
    if not n_clicks:
        return no_update, no_update

    incluir_cinetico   = "cinetico"   in (secciones or [])
    incluir_cinematico = "cinematico" in (secciones or [])

    figs_cinetico   = _figs_from_store(store_cinetico)   if incluir_cinetico   else []
    figs_cinematico = _figs_from_store(store_cinematico) if incluir_cinematico else []
    figs_esp_temp   = _figs_from_store(store_esp_temp)   if "esp_temp" in (secciones or []) else []

    filename = (pdf_name or "reporte_marcha.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    def _build_pdf(buffer):
        styles = _pdf_styles()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            # ↓ más ancho útil para que no se “aplasten”
            leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=28
        )
        Story = []

        def add_h1(text):
            Story.append(Paragraph(f"<b>{text}</b>", styles["AppH1"]))
            Story.append(Spacer(1, 6))

        def add_section(titulo, figs):
            if not figs:
                return
            Story.append(Spacer(1, 4))
            Story.append(Paragraph(f"<b>{titulo}</b>", styles["AppH1"]))
            Story.append(Spacer(1, 6))
            for title, fig in figs:
                # 1) estilo homogéneo para PDF
                fig_pdf = _style_fig_for_pdf(fig)
                # 2) export a PNG nítido
                img_bytes = pio.to_image(fig_pdf, format="png", scale=3)  # ↑ escala
                img = Image(io.BytesIO(img_bytes))
                # 3) ancho máximo del documento
                max_w = doc.width
                img.drawWidth  = max_w
                img.drawHeight = (img.imageHeight / float(img.imageWidth)) * max_w
                Story.append(img)
                Story.append(Spacer(1, 10))
            Story.append(PageBreak())

        # Portada
        Story.append(Paragraph("Reporte: Análisis de Marcha", styles["AppTitle"]))
        fecha_hoy = datetime.date.today().strftime("%d/%m/%Y")
        Story.append(Paragraph(f"Fecha: <b>{fecha_hoy}</b>", styles["AppText"]))
        
        if patient:
            nombre = (patient.get("nombre") or "—") if isinstance(patient, dict) else "—"
            edad   = (patient.get("edad")   or "—") if isinstance(patient, dict) else "—"
            peso   = (patient.get("peso")   or "—") if isinstance(patient, dict) else "—"
            Story.append(Paragraph(f"Usuario: <b>{nombre}</b>", styles["AppText"]))
            Story.append(Paragraph(f"Edad: <b>{edad}</b>  |  Peso: <b>{peso} kg</b>", styles["AppText"]))
        Story.append(Spacer(1, 12))

        # Secciones
        if incluir_cinetico:
            add_section("Análisis Cinético", figs_cinetico)
        if incluir_cinematico:
            add_section("Análisis Cinemático", figs_cinematico)
       
        if "esp_temp" in (secciones or []):                   
                add_section("Parámetros Espaciotemporales", figs_esp_temp)

        if not (incluir_cinetico or incluir_cinematico):
            Story.append(Paragraph("No se seleccionaron secciones para incluir.", styles["SmallMuted"]))

        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=28
        )

        def add_section(titulo, figs):
            if not figs:
                return
            Story.append(Spacer(1, 6))
            Story.append(Paragraph(f"<b>{titulo}</b>", styles["Heading1"]))
            Story.append(Spacer(1, 8))
            for title, fig in figs:
                fig_pdf = _style_fig_for_pdf(fig)              # ⬅️ normalizamos
                img_bytes = pio.to_image(fig_pdf, format="png", scale=3)  # ⬅️ más nítido
                img = Image(io.BytesIO(img_bytes))
                max_w = doc.width
                img.drawWidth  = max_w
                img.drawHeight = (img.imageHeight / float(img.imageWidth)) * max_w
                Story.append(img)
                Story.append(Spacer(1, 10))
            Story.append(PageBreak())
        doc.build(Story)

    return dcc.send_bytes(_build_pdf, filename), html.Div("Reporte generado ✅", style={"color": "#28a745", "textAlign":"center"})



