"""
PixelQuest Studios - BI sobre datos de jugadores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analisis_inteligencia_gamer import (
    agregar_metricas_valor,
    cargar_limpio,
    consultas_dataframes,
    hallazgos_negocio,
    tabla_olap,
)

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_CRUDO = BASE_DIR / "datos_gamer.csv"

st.set_page_config(
    page_title="PixelQuest Studios — Inteligencia de datos",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def cargar_crudos() -> pd.DataFrame | None:
    if not ARCHIVO_CRUDO.is_file():
        return None
    return pd.read_csv(ARCHIVO_CRUDO, encoding="utf-8")


@st.cache_data
def cargar_analitica() -> pd.DataFrame:
    df = cargar_limpio()
    return agregar_metricas_valor(df)


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; }
        div[data-testid="stTabs"] button { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("PixelQuest Studios")
    st.caption("Panel de inteligencia de negocio · datos de jugadores")

    try:
        df = cargar_analitica()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Ejecuta primero `python procesar_datos_gamer.py` para generar "
            "`datos_gamer_limpios.csv` desde `datos_gamer.csv`."
        )
        return

    df_crudos = cargar_crudos()
    tablas, meta = consultas_dataframes(df)
    pivot = tabla_olap(df)
    hallazgos = hallazgos_negocio(df)

    with st.sidebar:
        st.subheader("Resumen rápido")
        st.metric("Jugadores (panel limpio)", f"{len(df):,}")
        st.metric("Gasto mensual total (USD)", f"{df['gasto_mensual_usd'].sum():,.2f}")
        st.metric("Horas/semana (media)", f"{df['horas_semanales'].mean():.1f}")
        st.divider()
        st.caption("Fuente analítica: `datos_gamer_limpios.csv` + métricas de valor.")

    t1, t2, t3, t4 = st.tabs(
        ["1. Datos crudos / limpios", "2. KPIs y consultas", "3. Tabla OLAP", "4. Gráficas interactivas"]
    )

    with t1:
        st.subheader("Exploración de datos")
        modo = st.radio(
            "Vista",
            ["Datos crudos (`datos_gamer.csv`)", "Datos limpios enriquecidos"],
            horizontal=True,
        )
        if modo.startswith("Datos crudos"):
            if df_crudos is None:
                st.warning(f"No se encontró `{ARCHIVO_CRUDO.name}`. Colócalo en la carpeta del proyecto.")
            else:
                st.dataframe(df_crudos, width="stretch", height=420)
                st.caption(f"{len(df_crudos)} filas · {len(df_crudos.columns)} columnas")
        else:
            cols_show = [
                c
                for c in [
                    "id_jugador",
                    "nombre",
                    "plataforma",
                    "genero_juego",
                    "horas_semanales",
                    "gasto_mensual_usd",
                    "tipo_jugador",
                    "riesgo_abandono",
                    "score_valor_jugador",
                    "segmento_valor",
                    "rank_valor",
                ]
                if c in df.columns
            ]
            st.dataframe(df[cols_show].sort_values("rank_valor"), width="stretch", height=420)
            st.caption(
                "Incluye segmento de valor, ranking y variables del pipeline de limpieza (`procesar_datos_gamer.py`)."
            )

    with t2:
        st.subheader("Indicadores y consultas estratégicas")
        c1, c2, c3, c4 = st.columns(4)
        gasto_total = df["gasto_mensual_usd"].sum()
        riesgo_alto_pct = (df["riesgo_abandono"] == "Alto").mean() * 100
        plat_top = (
            df.groupby("plataforma", observed=False)["gasto_mensual_usd"]
            .sum()
            .idxmax()
        )
        r_corr = meta["correlacion_horas_gasto"]
        with c1:
            st.metric("Gasto mensual total", f"${gasto_total:,.0f}")
        with c2:
            st.metric("% jugadores riesgo Alto", f"{riesgo_alto_pct:.1f}%")
        with c3:
            st.metric("Plataforma top (suma USD)", str(plat_top))
        with c4:
            st.metric("Corr. horas ↔ gasto", f"{r_corr:.3f}")

        st.markdown("##### Hallazgos automáticos")
        for i, h in enumerate(hallazgos[:8], start=1):
            st.markdown(f"{i}. {h}")

        st.divider()
        st.markdown("##### Consultas Q1–Q5")
        with st.expander("Q1 — Ingresos por plataforma", expanded=True):
            st.dataframe(tablas["q1_ingresos_plataforma"], width="stretch")
        with st.expander("Q2 — Engagement y valor por género de juego"):
            st.dataframe(tablas["q2_genero_engagement"], width="stretch")
        with st.expander("Q3 — Gasto por tipo de jugador"):
            st.dataframe(tablas["q3_tipo_jugador_gasto"], width="stretch")
        with st.expander("Q4 — Riesgo de abandono por plataforma (% fila)"):
            st.dataframe(tablas["q4_riesgo_por_plataforma_pct"], width="stretch")
        with st.expander("Q5 — Segmento de valor vs riesgo alto"):
            st.dataframe(tablas["q5_segmento_valor_crm"], width="stretch")

        st.markdown("##### Q6 — Horas y gasto")
        st.write(meta["interpretacion_correlacion"])

    with t3:
        st.subheader("Cubo OLAP — Plataforma × Género")
        st.caption("Suma de `gasto_mensual_usd` y `horas_semanales` por celda.")
        pivot_display = pivot.copy()
        st.dataframe(pivot_display, width="stretch")
        csv_bytes = pivot_display.to_csv().encode("utf-8")
        st.download_button(
            "Descargar pivot (CSV)",
            data=csv_bytes,
            file_name="olap_plataforma_genero.csv",
            mime="text/csv",
        )

    with t4:
        st.subheader("Visualizaciones interactivas")
        ing = (
            df.groupby("plataforma", observed=False)["gasto_mensual_usd"]
            .sum()
            .reset_index()
            .sort_values("gasto_mensual_usd", ascending=True)
        )
        fig_bar = px.bar(
            ing,
            x="gasto_mensual_usd",
            y="plataforma",
            orientation="h",
            title="Facturación agregada por plataforma",
            labels={"gasto_mensual_usd": "Gasto mensual total (USD)", "plataforma": "Plataforma"},
            color="gasto_mensual_usd",
            color_continuous_scale="Viridis",
        )
        fig_bar.update_layout(height=380, showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, width="stretch")
        st.caption(
            "La plataforma con barra más larga concentra ingreso directo; útil para alianzas y priorización de ofertas."
        )

        gasto_pivot = pd.pivot_table(
            df,
            values="gasto_mensual_usd",
            index="plataforma",
            columns="genero_juego",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=gasto_pivot.values,
                x=[str(c) for c in gasto_pivot.columns],
                y=[str(i) for i in gasto_pivot.index],
                colorscale="YlOrRd",
                hovertemplate="Plataforma=%{y}<br>Género=%{x}<br>Gasto USD=%{z:.0f}<extra></extra>",
            )
        )
        fig_hm.update_layout(
            title="Gasto mensual por plataforma y género",
            xaxis_title="Género de juego",
            yaxis_title="Plataforma",
            height=max(420, len(gasto_pivot.index) * 36),
        )
        st.plotly_chart(fig_hm, width="stretch")

        orden = ["Casual", "Frecuente", "Hardcore"]
        orden = [x for x in orden if x in df["tipo_jugador"].unique()]
        fig_box = px.box(
            df,
            x="tipo_jugador",
            y="gasto_mensual_usd",
            category_orders={"tipo_jugador": orden} if orden else None,
            color="tipo_jugador",
            title="Distribución de gasto por tipo de jugador",
            labels={"tipo_jugador": "Tipo", "gasto_mensual_usd": "Gasto mensual (USD)"},
        )
        fig_box.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

        st.scatter_chart(
            df,
            x="horas_semanales",
            y="gasto_mensual_usd",
            color="segmento_valor" if "segmento_valor" in df.columns else None,
            size="score_valor_jugador" if "score_valor_jugador" in df.columns else None,
        )
        st.caption("Dispersión horas vs gasto (tamaño ~ score de valor cuando está disponible).")


if __name__ == "__main__":
    main()
