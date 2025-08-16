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

dash.register_page(__name__, path="/reporte", name="Reporte", icon="file-earmark-pdf")

layout = html.Div(
    [
        html.H1("Reporte: Análisis de Marcha", style={"margin":"6px 0 12px 0","textAlign":"center","fontWeight":"700","color":"#2c3e50"}),
        html.Div(
            [
                dcc.Checklist(
                    id="reporte-secciones",
                    options=[
                        {"label":" Incluir análisis cinético", "value":"cinetico"},
                        {"label":" Incluir análisis cinemático", "value":"cinematico"},
                        {"label":" Incluir encabezado con datos del paciente", "value":"paciente"},
                    ],
                    value=["cinetico","cinematico","paciente"],
                    inputStyle={"marginRight":"1px"},
                    labelStyle={"display":"block","margin":"1px 0", "color":"#00000097"}
                ),
                html.Button("Generar PDF", id="btn-gen-pdf", style={
                    "backgroundColor":"#2c3e50","color":"white","border":"none","padding":"8px 14px",
                    "borderRadius":"6px","cursor":"pointer","fontWeight":"600"
                }),
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
    State("store-patient-info", "data"),
    prevent_initial_call=True
)

def generar_pdf(n_clicks, secciones, store_kinetico, store_cinematico, patient):
    if not n_clicks:
        return no_update, no_update

    # Decidir qué secciones incluir según el checklist
    incluir_cinetico = "cinetico" in (secciones or [])
    incluir_cinematico = "cinematico" in (secciones or [])

    # Convertir los stores (dicts con JSON) a listas de (titulo, fig)
    figs_cinetico = _figs_from_store(store_kinetico) if incluir_cinetico else []
    figs_cinematico = _figs_from_store(store_cinematico) if incluir_cinematico else []

    filename = "reporte_marcha.pdf"

    def _build_pdf(buffer):
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
        )
        Story = []

        def add_h1(text):
            Story.append(Paragraph(f"<b>{text}</b>", styles["Title"]))
            Story.append(Spacer(1, 6))

        def add_section(titulo, figs):
            if not figs:
                return
            Story.append(Spacer(1, 6))
            Story.append(Paragraph(f"<b>{titulo}</b>", styles["Heading1"]))
            Story.append(Spacer(1, 8))
            for title, fig in figs:  # <- _figs_from_store devuelve (titulo, fig)
                # Render a imagen (requiere kaleido instalado)
                img_bytes = pio.to_image(fig, format="png", scale=2)
                img_buffer = io.BytesIO(img_bytes)
                img = Image(img_buffer)
                max_w = doc.width
                img.drawWidth = max_w
                img.drawHeight = (img.imageHeight / float(img.imageWidth)) * max_w
                Story.append(img)
                Story.append(Spacer(1, 8))
            Story.append(PageBreak())

        # Portada
        add_h1("Reporte de Marcha")
        if patient:
            nombre = (patient.get("nombre") or "—") if isinstance(patient, dict) else "—"
            edad = (patient.get("edad") or "—") if isinstance(patient, dict) else "—"
            peso = (patient.get("peso") or "—") if isinstance(patient, dict) else "—"
            Story.append(Paragraph(f"Usuario: <b>{nombre}</b>", styles["Normal"]))
            Story.append(Paragraph(f"Edad: <b>{edad}</b>  |  Peso: <b>{peso} kg</b>", styles["Normal"]))
        Story.append(Spacer(1, 12))

        # Secciones seleccionadas
        if incluir_cinetico:
            add_section("Análisis Cinético", figs_cinetico)
        if incluir_cinematico:
            add_section("Análisis Cinemático", figs_cinematico)

        # Si ninguna sección fue seleccionada, al menos dejar constancia
        if not (incluir_cinetico or incluir_cinematico):
            Story.append(Paragraph("No se seleccionaron secciones para incluir.", styles["Italic"]))

        doc.build(Story)

    return dcc.send_bytes(_build_pdf, filename), html.Div("<b>Reporte generado<b> ✅", style={"color": "#28a745"})


