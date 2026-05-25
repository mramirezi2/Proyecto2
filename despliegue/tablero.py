# ======================================================
# Tablero Proyecto 2 - Analítica de Datos
# Modelos predictivos Saber 11 - Cauca
# ======================================================

import dash
from dash import html, dcc, Input, Output
from pathlib import Path

# ======================================================
# Configuración inicial
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
app = dash.Dash(
    __name__,
    assets_folder=str(BASE_DIR / "assets"),
    suppress_callback_exceptions=True)
server = app.server

# ======================================================
# Estilos generales
# ======================================================

COLOR_AZUL = "#1A3980"
COLOR_GRIS = "#F5F7FA"
COLOR_BORDE = "#D9DEE7"

card_style = {
    "backgroundColor": "white",
    "border": f"1px solid {COLOR_BORDE}",
    "borderRadius": "12px",
    "padding": "22px",
    "boxShadow": "0px 3px 8px rgba(0,0,0,0.08)",
    "marginBottom": "20px"}


# ======================================================
# Header con logos
# ======================================================

header = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "18px 28px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "borderBottom": f"2px solid {COLOR_BORDE}",
        "backgroundColor": "white"
    },
    children=[
        html.Div(
            style={
                "display": "flex",
                "gap": "18px",
                "alignItems": "center"
            },
            children=[
                html.Img(
                    src="/assets/logo_andes.png",
                    style={"height": "70px"}
                ),
                html.Img(
                    src="/assets/logo_cauca.png",
                    style={"height": "70px"}
                )
            ]
        ),

        html.H2(
            "Tablero de Análisis Educativo - Cauca",
            style={
                "color": COLOR_AZUL,
                "margin": "0px",
                "textAlign": "center"
            }
        )
    ]
)


# ======================================================
# Tab 0 - Portada
# ======================================================

tab_portada = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": COLOR_GRIS
    },
    children=[
        html.Div(
            style=card_style,
            children=[
                html.H1(
                    "Análisis predictivo del desempeño académico en el Cauca",
                    style={
                        "color": COLOR_AZUL,
                        "textAlign": "center",
                        "marginBottom": "25px"
                    }
                ),

                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "gap": "40px",
                        "flexWrap": "wrap"
                    },
                    children=[
                        html.Div(
                            style={
                                "maxWidth": "720px",
                                "textAlign": "justify",
                                "lineHeight": "1.7",
                                "fontSize": "16px"
                            },
                            children=[
                                html.P(
                                    "Este tablero presenta una herramienta de analítica desarrollada "
                                    "para la Secretaría de Educación y Cultura del Cauca. Su objetivo es apoyar "
                                    "la identificación de factores asociados al desempeño académico de los "
                                    "estudiantes en las Pruebas Saber 11."
                                ),
                                html.P(
                                    "A partir de modelos predictivos, la herramienta permite explorar cómo las condiciones "
                                    "institucionales, la brecha digital, el contexto del hogar y las características "
                                    "sociodemográficas se relacionan con el desempeño esperado de los estudiantes del departamento."
                                )
                            ]
                        ),
                        html.Img(
                            src="/assets/foto_portada.jpg",
                            style={
                                "width": "330px",
                                "height": "230px",
                                "objectFit": "cover",
                                "borderRadius": "14px",
                                "boxShadow": "0px 4px 10px rgba(0,0,0,0.15)"
                            }
                        )

                    ]
                )
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                "gap": "20px"
            },
            children=[
                html.Div(
                    style=card_style,
                    children=[
                        html.H3(
                            "¿Qué encontrará en el tablero?",
                            style={"color": COLOR_AZUL}
                        ),
                        html.P(
                            "Predicciones, visualizaciones y resultados de modelos asociados "
                            "al desempeño académico de los estudiantes del Cauca con énfasis  "
                            "en condiciones institucionales, brecha digital y características sociodemográficas."
                        )
                    ]
                ),

                html.Div(
                    style=card_style,
                    children=[
                        html.H3(
                            "Preguntas de negocio",
                            style={"color": COLOR_AZUL}
                        ),
                        html.P(
                            "Cada pestaña responde una pregunta específica:  desempeño según condiciones institucionales,"
                            "riesgo de bajo desempeño asociado a brecha digital y percentil nacional esperado según "
                            "características sociodemográficas."
                        )
                    ]
                ),

                html.Div(
                    style=card_style,
                    children=[
                        html.H3(
                            "Usuario final",
                            style={"color": COLOR_AZUL}
                        ),
                        html.P(
                            "Secretaría de Educación y Cultura del Cauca, equipos técnicos "
                            "y tomadores de decisiones del sector educativo."
                        )
                    ]
                )
            ]
        )
    ]
)


# ======================================================
# Tabs para las preguntas
# ======================================================

tab_pregunta_1 = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": COLOR_GRIS
    },
    children=[
        html.H2("Desempeño institucional", style={"color": COLOR_AZUL}),
        html.Div(
            style=card_style,
            children=[
                html.H3("Descripción del problema", style={"color": COLOR_AZUL}),
                html.P(
                    "En esta sección se integrará el primer modelo predictivo. "
                    "Aquí se incluirán los campos de entrada, el resultado del modelo "
                    "y el hallazgo principal asociado a la pregunta de negocio."
                )
            ]
        )
    ]
)


tab_pregunta_2 = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": COLOR_GRIS
    },
    children=[
        html.H2("Riesgo por brecha digital", style={"color": COLOR_AZUL}),
        html.Div(
            style=card_style,
            children=[
                html.H3("Descripción del problema", style={"color": COLOR_AZUL}),
                html.P(
                    "En esta sección se integrará el segundo modelo predictivo. "
                    "La estructura puede incluir descripción, entradas del usuario, "
                    "predicción del modelo y visualizaciones de apoyo."
                )
            ]
        )
    ]
)


tab_pregunta_3 = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": COLOR_GRIS
    },
    children=[
        html.H2("Percentil esperado", style={"color": COLOR_AZUL}),
        html.Div(
            style=card_style,
            children=[
                html.H3("Descripción del problema", style={"color": COLOR_AZUL}),
                html.P(
                    "En esta sección se integrará el tercer modelo predictivo. "
                    "Aquí se presentarán los parámetros requeridos, la salida del modelo "
                    "y una conclusión orientada a la toma de decisiones."
                )
            ]
        )
    ]
)


# ======================================================
# Layout principal
# ======================================================

app.layout = html.Div(
    style={
        "backgroundColor": COLOR_GRIS,
        "minHeight": "100vh"
    },
    children=[
        header,

        dcc.Tabs(
            id="tabs",
            value="portada",
            children=[
                dcc.Tab(label="Portada", value="portada"),
                dcc.Tab(label="Desempeño institucional", value="pregunta_1"),
                dcc.Tab(label="Riesgo por brecha digital", value="pregunta_2"),
                dcc.Tab(label="Percentil esperado", value="pregunta_3")
            ],
            style={"fontFamily": "Arial"}
        ),

        html.Div(id="contenido-tabs")
    ]
)


# ======================================================
# Callback para cambiar de pestaña
# ======================================================

@app.callback(
    Output("contenido-tabs", "children"),
    Input("tabs", "value")
)
def renderizar_tabs(tab):
    if tab == "portada":
        return tab_portada
    elif tab == "pregunta_1":
        return tab_pregunta_1
    elif tab == "pregunta_2":
        return tab_pregunta_2
    elif tab == "pregunta_3":
        return tab_pregunta_3

    return tab_portada


# ======================================================
# Ejecución local
# ======================================================

if __name__ == "__main__":
    app.run(debug=True)