"""
PixelQuest Studios — Panel de BI (Streamlit) sobre la base limpia y métricas de valor.

Orquesta carga, KPIs, export OLAP y visualización interactiva reutilizando el núcleo analítico compartido.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analisis_inteligencia_gamer import (
    agregar_metricas_valor,
    calcular_kpis,
    cargar_limpio,
    consultas_dataframes,
    hallazgos_negocio,
    tabla_olap,
    tabla_olap_pais,
    tabla_olap_juego_genero,
    tabla_olap_pais_genero,
)

# ==========================================
# CONFIGURACIÓN Y RUTAS
# ==========================================
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

    # Cálculos
    tablas, meta = consultas_dataframes(df)
    pivot = tabla_olap(df)
    hallazgos = hallazgos_negocio(df)
    kpis = calcular_kpis(df)

    # ==========================================
    # SIDEBAR
    # ==========================================
    with st.sidebar:
        st.subheader("Resumen rápido")
        st.metric("Jugadores", f"{kpis.get('total_jugadores', len(df)):,}")
        st.metric("Ingreso total (MXN)", f"${kpis.get('ingreso_total_mxn', 0):,.0f}")
        st.metric("Horas/semana (media)", f"{kpis.get('horas_promedio', 0):.1f}")
        st.metric("% Riesgo Alto", f"{kpis.get('pct_riesgo_alto', 0):.1f}%")

        if "pct_activos" in kpis:
            st.metric("% Activos", f"{kpis['pct_activos']:.1f}%")
        if "pct_compras_internas" in kpis:
            st.metric("% Compran", f"{kpis['pct_compras_internas']:.1f}%")
        if "calificacion_promedio" in kpis:
            st.metric("Calificación media", f"{kpis['calificacion_promedio']:.2f}/5.0")
        st.divider()
        st.caption("Fuente analítica: `datos_gamer_limpios.csv` + métricas de valor.")

    # ==========================================
    # TABS
    # ==========================================
    t1, t2, t3, t4, t5 = st.tabs(
        ["1. Datos", "2. KPIs y Consultas", "3. OLAP", "4. Gráficas", "5. Preguntas de Negocio"]
    )

    # ==========================================
    # TAB 1 — Exploración de datos
    # ==========================================
    with t1:
        st.subheader("Exploración de datos")
        modo = st.radio(
            "Vista",
            ["Datos crudos (`datos_gamer.csv`)", "Datos limpios enriquecidos"],
            horizontal=True,
        )
        if modo.startswith("Datos crudos"):
            if df_crudos is None:
                st.warning(f"No se encontró `{ARCHIVO_CRUDO.name}`.")
            else:
                st.dataframe(df_crudos, width="stretch", height=420)
                st.caption(f"{len(df_crudos)} filas · {len(df_crudos.columns)} columnas")
        else:
            cols_show = [c for c in [
                "id_jugador", "nombre", "pais", "plataforma", "genero_juego",
                "nombre_juego", "horas_semanales", "ingreso_mensual_mxn",
                "nivel_alcanzado", "calificacion", "compras_internas",
                "estado_jugador", "antiguedad_dias", "dias_ultima_conexion",
                "tipo_jugador", "riesgo_abandono",
                "score_valor_jugador", "score_valor_compuesto",
                "segmento_valor", "rank_valor", "alto_valor",
            ] if c in df.columns]
            st.dataframe(df[cols_show].sort_values("rank_valor"), width="stretch", height=420)
            st.caption("Incluye segmento de valor, ranking y variables del pipeline de limpieza.")

    # ==========================================
    # TAB 2 — KPIs y consultas
    # ==========================================
    with t2:
        st.subheader("Indicadores estratégicos")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Ingreso mensual total", f"${kpis.get('ingreso_total_mxn', 0):,.0f}")
        with c2:
            st.metric("% Riesgo Alto", f"{kpis.get('pct_riesgo_alto', 0):.1f}%")
        with c3:
            st.metric("Corr. horas/ingreso", f"{kpis.get('corr_horas_ingreso', 0):.3f}")
        with c4:
            st.metric("% Compras internas", f"{kpis.get('pct_compras_internas', 'N/A')}%")

        if "pct_activos" in kpis or "calificacion_promedio" in kpis:
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.metric("% Activos", f"{kpis.get('pct_activos', 'N/A')}%")
            with c6:
                st.metric("Calificación media", f"{kpis.get('calificacion_promedio', 'N/A')}")
            with c7:
                st.metric("Nivel promedio", f"{kpis.get('nivel_promedio', 'N/A')}")
            with c8:
                st.metric("País top activos", str(kpis.get('pais_top_activos', 'N/A')))

        st.markdown("##### Hallazgos automáticos")
        for i, h in enumerate(hallazgos[:12], start=1):
            st.markdown(f"{i}. {h}")

        st.divider()
        st.markdown("##### Consultas Q1–Q5")
        with st.expander("Q1 — Ingresos por plataforma", expanded=True):
            st.dataframe(tablas["q1_ingresos_plataforma"], width="stretch")
        with st.expander("Q2 — Engagement y valor por género"):
            st.dataframe(tablas["q2_genero_engagement"], width="stretch")
        with st.expander("Q3 — Ingreso por tipo de jugador"):
            st.dataframe(tablas["q3_tipo_jugador_ingreso"], width="stretch")
        with st.expander("Q4 — Riesgo de abandono por plataforma (% fila)"):
            st.dataframe(tablas["q4_riesgo_por_plataforma_pct"], width="stretch")
        with st.expander("Q5 — Segmento de valor vs riesgo"):
            st.dataframe(tablas["q5_segmento_valor_crm"], width="stretch")

        st.markdown("##### Consultas Q6–Q10")
        for key, title in [
            ("q6_pais_activos", "Q6 — País con más jugadores activos"),
            ("q7_compras_tipo_jugador", "Q7 — Compras internas por tipo de jugador"),
            ("q8_calificacion_juego", "Q8 — Calificación promedio por juego"),
            ("q9_plataforma_genero_ingresos", "Q9 — Ingresos por plataforma × género"),
            ("q10_alto_valor_riesgo", "Q10 — Jugadores de alto valor vs riesgo"),
        ]:
            if key in tablas:
                with st.expander(title):
                    st.dataframe(tablas[key], width="stretch")

        st.markdown("##### Correlación horas ↔ ingreso")
        st.write(meta["interpretacion_correlacion"])

    # ==========================================
    # TAB 3 — OLAP
    # ==========================================
    with t3:
        st.subheader("Cubos OLAP")
        st.caption("Suma de `ingreso_mensual_mxn` y `horas_semanales` por celda. Exportables a CSV.")

        with st.expander("Plataforma × Género", expanded=True):
            st.dataframe(pivot, width="stretch")
            csv_bytes = pivot.to_csv().encode("utf-8")
            st.download_button("Descargar pivot (CSV)", data=csv_bytes,
                               file_name="olap_plataforma_genero.csv", mime="text/csv")

        pivot_pais = tabla_olap_pais(df)
        if pivot_pais is not None:
            with st.expander("País × Plataforma"):
                st.dataframe(pivot_pais, width="stretch")
                st.download_button("Descargar (CSV)", data=pivot_pais.to_csv().encode("utf-8"),
                                   file_name="olap_pais_plataforma.csv", mime="text/csv")

        pivot_juego = tabla_olap_juego_genero(df)
        if pivot_juego is not None:
            with st.expander("Juego × Género"):
                st.dataframe(pivot_juego, width="stretch")
                st.download_button("Descargar (CSV)", data=pivot_juego.to_csv().encode("utf-8"),
                                   file_name="olap_juego_genero.csv", mime="text/csv")

        pivot_pg = tabla_olap_pais_genero(df)
        if pivot_pg is not None:
            with st.expander("País × Género"):
                st.dataframe(pivot_pg, width="stretch")
                st.download_button("Descargar (CSV)", data=pivot_pg.to_csv().encode("utf-8"),
                                   file_name="olap_pais_genero.csv", mime="text/csv")

    # ==========================================
    # TAB 4 — Gráficas interactivas
    # ==========================================
    with t4:
        st.subheader("Visualizaciones interactivas")

        # G1 — Ingreso por plataforma
        ing = (
            df.groupby("plataforma", observed=False)["ingreso_mensual_mxn"]
            .sum()
            .reset_index()
            .sort_values("ingreso_mensual_mxn", ascending=True)
        )
        fig_bar = px.bar(
            ing, x="ingreso_mensual_mxn", y="plataforma", orientation="h",
            title="Facturación agregada por plataforma",
            labels={"ingreso_mensual_mxn": "Ingreso mensual total (MXN)", "plataforma": "Plataforma"},
            color="ingreso_mensual_mxn", color_continuous_scale="Viridis",
        )
        fig_bar.update_layout(height=380, showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, width="stretch")

        # G2 — Heatmap plataforma × género
        ingreso_pivot = pd.pivot_table(
            df, values="ingreso_mensual_mxn", index="plataforma", columns="genero_juego",
            aggfunc="sum", fill_value=0, observed=False,
        )
        fig_hm = go.Figure(data=go.Heatmap(
            z=ingreso_pivot.values,
            x=[str(c) for c in ingreso_pivot.columns],
            y=[str(i) for i in ingreso_pivot.index],
            colorscale="YlOrRd",
            hovertemplate="Plataforma=%{y}<br>Género=%{x}<br>Ingreso MXN=%{z:.0f}<extra></extra>",
        ))
        fig_hm.update_layout(
            title="Ingreso mensual por plataforma y género",
            xaxis_title="Género", yaxis_title="Plataforma",
            height=max(420, len(ingreso_pivot.index) * 36),
        )
        st.plotly_chart(fig_hm, width="stretch")

        # G3 — Boxplot ingreso por tipo de jugador
        orden = [x for x in ["Casual", "Frecuente", "Hardcore"] if x in df["tipo_jugador"].unique()]
        fig_box = px.box(
            df, x="tipo_jugador", y="ingreso_mensual_mxn",
            category_orders={"tipo_jugador": orden} if orden else None,
            color="tipo_jugador",
            title="Distribución de ingreso por tipo de jugador",
            labels={"tipo_jugador": "Tipo", "ingreso_mensual_mxn": "Ingreso mensual (MXN)"},
        )
        fig_box.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

        # G4 — Jugadores activos/inactivos por país
        if "pais" in df.columns and "estado_jugador" in df.columns:
            pais_act = (
                df.groupby("pais", observed=False)
                .agg(activos=("estado_jugador", lambda s: (s == "Activo").sum()),
                     inactivos=("estado_jugador", lambda s: (s == "Inactivo").sum()))
                .reset_index()
                .melt(id_vars="pais", var_name="Estado", value_name="Cantidad")
            )
            fig_pais = px.bar(
                pais_act, y="pais", x="Cantidad", color="Estado",
                orientation="h", barmode="stack",
                title="Jugadores activos e inactivos por país",
                color_discrete_map={"activos": "#2ecc71", "inactivos": "#e74c3c"},
                labels={"pais": "País", "Cantidad": "Jugadores"},
            )
            fig_pais.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_pais, width="stretch")

        # G5 — Calificación promedio por juego
        if "calificacion" in df.columns and "nombre_juego" in df.columns:
            calif = (
                df.groupby("nombre_juego", observed=False)["calificacion"]
                .mean()
                .reset_index()
                .sort_values("calificacion", ascending=True)
            )
            fig_cal = px.bar(
                calif, x="calificacion", y="nombre_juego", orientation="h",
                title="Calificación promedio por videojuego",
                labels={"calificacion": "Calificación (1-5)", "nombre_juego": "Juego"},
                color="calificacion", color_continuous_scale="RdYlGn",
                text=calif["calificacion"].round(2),
            )
            fig_cal.update_layout(height=400, showlegend=False, yaxis={"categoryorder": "total ascending"})
            fig_cal.update_traces(textposition="outside")
            st.plotly_chart(fig_cal, width="stretch")

        # G6 — Compras internas por tipo de jugador
        if "compras_internas" in df.columns:
            compras = (
                df.groupby("tipo_jugador", observed=False)["compras_internas"]
                .apply(lambda s: (s == "Sí").mean() * 100)
                .reset_index()
                .sort_values("compras_internas", ascending=True)
                .rename(columns={"compras_internas": "tasa_conversion"})
            )
            fig_comp = px.bar(
                compras, x="tasa_conversion", y="tipo_jugador", orientation="h",
                title="Tasa de compras internas por tipo de jugador (%)",
                labels={"tasa_conversion": "% que compra", "tipo_jugador": "Tipo"},
                color="tasa_conversion", color_continuous_scale="Blues",
                text=compras["tasa_conversion"].round(1).astype(str) + "%",
            )
            fig_comp.update_layout(height=300, showlegend=False)
            fig_comp.update_traces(textposition="outside")
            st.plotly_chart(fig_comp, width="stretch")

        # G7 — Scatter horas vs ingreso (con color por segmento)
        if "segmento_valor" in df.columns:
            fig_scat = px.scatter(
                df, x="horas_semanales", y="ingreso_mensual_mxn",
                color="segmento_valor",
                size="score_valor_jugador" if "score_valor_jugador" in df.columns else None,
                title="Horas vs Ingreso por segmento de valor",
                labels={"horas_semanales": "Horas/semana", "ingreso_mensual_mxn": "Ingreso mensual (MXN)"},
                opacity=0.7,
                color_discrete_map={"Bronce": "#cd7f32", "Plata": "#c0c0c0", "Oro": "#ffd700", "Platino": "#e5e4e2"},
            )
            fig_scat.update_layout(height=450)
            st.plotly_chart(fig_scat, width="stretch")

        # G8 — Riesgo de abandono por país
        if "pais" in df.columns:
            riesgo_pais = (
                pd.crosstab(df["pais"], df["riesgo_abandono"], normalize="index") * 100
            ).round(1).reset_index()
            if "Alto" in riesgo_pais.columns:
                riesgo_pais = riesgo_pais.sort_values("Alto", ascending=True)
            riesgo_melt = riesgo_pais.melt(id_vars="pais", var_name="Riesgo", value_name="Porcentaje")
            fig_riesgo = px.bar(
                riesgo_melt, y="pais", x="Porcentaje", color="Riesgo",
                orientation="h", barmode="stack",
                title="Distribución de riesgo de abandono por país (%)",
                color_discrete_map={"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"},
                labels={"pais": "País", "Porcentaje": "% de jugadores"},
            )
            fig_riesgo.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_riesgo, width="stretch")

    # ==========================================
    # TAB 5 — Preguntas de Negocio
    # ==========================================
    with t5:
        st.subheader("Respuestas a las 10 preguntas del negocio")

        with st.expander("1. Qué plataforma genera mayor ingreso promedio?", expanded=True):
            ing_q = tablas["q1_ingresos_plataforma"].sort_values("ingreso_promedio", ascending=False)
            st.dataframe(ing_q, width="stretch")
            top_plat = ing_q.index[0]
            st.success(f"**{top_plat}** lidera con ingreso promedio de **${ing_q.iloc[0]['ingreso_promedio']:.2f}**")

        with st.expander("2. Qué género tiene más horas jugadas?"):
            gen_q = tablas["q2_genero_engagement"].sort_values("horas_promedio", ascending=False)
            st.dataframe(gen_q, width="stretch")
            top_gen = gen_q.index[0]
            st.success(f"**{top_gen}** lidera con **{gen_q.iloc[0]['horas_promedio']:.1f}** horas/semana promedio")

        with st.expander("3. Qué tipo de jugador realiza más compras internas?"):
            if "q7_compras_tipo_jugador" in tablas:
                st.dataframe(tablas["q7_compras_tipo_jugador"], width="stretch")
                top_comp = tablas["q7_compras_tipo_jugador"].index[0]
                st.success(f"**{top_comp}** tiene la mayor tasa de compras internas")
            else:
                st.info("Datos de compras internas no disponibles en este dataset.")

        with st.expander("4. Qué país concentra más jugadores activos?"):
            if "q6_pais_activos" in tablas:
                st.dataframe(tablas["q6_pais_activos"], width="stretch")
                top_pais = tablas["q6_pais_activos"].index[0]
                st.success(f"**{top_pais}** concentra la mayor cantidad de jugadores activos "
                           f"({tablas['q6_pais_activos'].iloc[0]['activos']} activos)")
            else:
                st.info("Datos de país no disponibles en este dataset.")

        with st.expander("5. Qué jugadores tienen mayor riesgo de abandono?"):
            riesgo_df = df[["id_jugador", "nombre", "pais", "plataforma", "riesgo_abandono",
                            "horas_semanales", "ingreso_mensual_mxn", "dias_ultima_conexion"]]
            top_riesgo = riesgo_df[riesgo_df["riesgo_abandono"] == "Alto"].sort_values(
                ["horas_semanales", "ingreso_mensual_mxn"]
            )
            st.dataframe(top_riesgo.head(20), width="stretch")
            st.warning(f"{len(top_riesgo)} jugadores tienen riesgo Alto: bajas horas + bajo ingreso. "
                       "Segmento prioritario para reactivación.")
            if "dias_ultima_conexion" in df.columns:
                inactivos = (df["dias_ultima_conexion"] > 30).sum()
                st.info(f"{inactivos} jugadores llevan más de 30 días sin conectarse.")

        with st.expander("6. Qué videojuego tiene mejor calificación promedio?"):
            if "q8_calificacion_juego" in tablas:
                st.dataframe(tablas["q8_calificacion_juego"], width="stretch")
                top_juego = tablas["q8_calificacion_juego"].index[0]
                st.success(f"**{top_juego}** tiene la mejor calificación: "
                           f"**{tablas['q8_calificacion_juego'].iloc[0]['calificacion_promedio']:.2f}/5.0**")
            else:
                st.info("Datos de calificación no disponibles en este dataset.")

        with st.expander("7. Qué combinación de plataforma y género genera más ingresos?"):
            if "q9_plataforma_genero_ingresos" in tablas:
                q9 = tablas["q9_plataforma_genero_ingresos"]
                st.dataframe(q9, width="stretch")
                max_val = q9.max().max()
                max_cell = q9.stack().idxmax()
                st.success(f"La combinación **{max_cell[0]} × {max_cell[1]}** genera **${max_val:,.0f}** en total")
            else:
                st.info("Realizando análisis...")
                q9_alt = pd.pivot_table(
                    df, values="ingreso_mensual_mxn", index="plataforma",
                    columns="genero_juego", aggfunc="sum", fill_value=0, observed=False,
                )
                st.dataframe(q9_alt, width="stretch")

        with st.expander("8. Qué segmento debería recibir promociones?"):
            promos = df.groupby("segmento_valor", observed=False).agg(
                jugadores=("id_jugador", "count"),
                ingreso_medio=("ingreso_mensual_mxn", "mean"),
                horas_medio=("horas_semanales", "mean"),
                pct_riesgo_alto=("riesgo_abandono", lambda s: (s == "Alto").mean() * 100),
            ).round(2)
            st.dataframe(promos, width="stretch")
            st.info(
                "Segmentos con alto % de riesgo de abandono + bajo ingreso son candidatos a promociones. "
                "Bronce y Plata con riesgo > 30% deben recibir campañas de retención con recompensas de regreso."
            )

        with st.expander("9. Qué jugadores son de alto valor?"):
            altos = df[df["alto_valor"] == True] if "alto_valor" in df.columns else pd.DataFrame()
            if not altos.empty:
                cols_av = [c for c in [
                    "id_jugador", "nombre", "pais", "plataforma", "nombre_juego",
                    "horas_semanales", "ingreso_mensual_mxn", "nivel_alcanzado",
                    "calificacion", "compras_internas", "segmento_valor",
                ] if c in altos.columns]
                st.dataframe(altos[cols_av].sort_values("ingreso_mensual_mxn", ascending=False), width="stretch")
                st.success(
                    f"{len(altos)} jugadores son de alto valor (top 20% en score compuesto). "
                    f"Ingreso promedio: ${altos['ingreso_mensual_mxn'].mean():.2f}"
                )
            else:
                st.info("No se pudo determinar la clasificación de alto valor.")

        with st.expander("10. Recomendaciones para PixelQuest Studios"):
            st.markdown("### Recomendaciones de retención y crecimiento")
            recs = [
                f"**Reactivar jugadores inactivos:** "
                f"{kpis.get('pct_inactivos', 'N/A')}% de la base está inactiva. "
                "Campaña de correo con recompensa de regreso (7 días de items premium).",
                f"**Fidelizar al segmento de alto valor:** "
                "Programa VIP con acceso anticipado a contenido, soporte prioritario y descuentos exclusivos.",
                f"**Expandir en el país top:** "
                f"'{kpis.get('pais_top_activos', 'N/A')}' tiene la mayor base activa. "
                "Campañas de marketing localizadas y precios regionales.",
                f"**Invertir en el género más rentable:** "
                "Reforzar el desarrollo de contenido para los géneros con mayor engagement y ingreso.",
                f"**Convertir jugadores frecuentes en compradores:** "
                "Ofrecer bundles personalizados basados en su estilo de juego (nivel, horas, género favorito).",
                f"**Monitorear el juego mejor calificado:** "
                "Identificar los elementos de diseño que generan mayor satisfacción y replicarlos.",
                f"**Estrategia por plataforma:** "
                "La plataforma líder en ingreso debe recibir ofertas exclusivas; "
                "las de bajo rendimiento necesitan optimización de precios.",
                f"**Reducir la fricción en onboarding:** "
                "Los jugadores con nivel bajo y calificación baja abandonan más rápido. "
                "Tutoriales simplificados y recompensas tempranas.",
            ]
            for i, r in enumerate(recs, 1):
                st.markdown(f"{i}. {r}")

            st.divider()
            st.caption("Estas recomendaciones se basan en los datos actuales del panel. "
                       "Se recomienda validar con tests A/B antes de implementar a escala.")


if __name__ == "__main__":
    main()
