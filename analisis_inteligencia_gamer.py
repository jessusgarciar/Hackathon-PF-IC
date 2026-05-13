"""
PixelQuest Studios — Inteligencia de negocio sobre datos_gamer_limpios.csv

Minería: ranking/segmento de valor (horas + ingreso + nivel).
Consultas estratégicas (Q1–Q10), tablas OLAP (pivot) y dashboard con interpretación.

Requisitos: pip install pandas matplotlib seaborn
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_LIMPIO = BASE_DIR / "datos_gamer_limpios.csv"

PESO_HORAS_EN_SCORE = 0.35
PESO_INGRESO_EN_SCORE = 0.65


def cargar_limpio() -> pd.DataFrame:
    if not ARCHIVO_LIMPIO.is_file():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_LIMPIO}. Ejecute primero procesar_datos_gamer.py."
        )
    return pd.read_csv(ARCHIVO_LIMPIO, encoding="utf-8")


def min_max(s: pd.Series) -> pd.Series:
    r = s.max() - s.min()
    if r == 0 or np.isnan(r):
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / r


def agregar_metricas_valor(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["horas_norm"] = min_max(out["horas_semanales"])
    out["ingreso_norm"] = min_max(out["ingreso_mensual_mxn"])
    out["score_valor_jugador"] = (
        PESO_HORAS_EN_SCORE * out["horas_norm"]
        + PESO_INGRESO_EN_SCORE * out["ingreso_norm"]
    )

    if "nivel_alcanzado" in out.columns and "calificacion" in out.columns:
        out["nivel_norm"] = min_max(out["nivel_alcanzado"].astype(float))
        out["calif_norm"] = min_max(out["calificacion"].astype(float))
        out["score_valor_compuesto"] = (
            0.25 * out["horas_norm"]
            + 0.40 * out["ingreso_norm"]
            + 0.20 * out["nivel_norm"]
            + 0.15 * out["calif_norm"]
        )
    else:
        out["score_valor_compuesto"] = out["score_valor_jugador"]

    bins = [0, 0.25, 0.5, 0.75, 1.0001]
    labels = ["Bronce", "Plata", "Oro", "Platino"]
    out["segmento_valor"] = pd.cut(
        out["score_valor_jugador"], bins=bins, labels=labels, include_lowest=True
    ).astype("string")
    out["rank_valor"] = out["score_valor_jugador"].rank(ascending=False, method="min").astype(int)
    h = out["horas_semanales"].replace(0, np.nan)
    out["indice_afinidad_genero"] = out["ingreso_mensual_mxn"] / h
    out["indice_afinidad_genero"] = out["indice_afinidad_genero"].replace([np.inf, -np.inf], np.nan)

    umbral_alto = out["score_valor_compuesto"].quantile(0.80)
    out["alto_valor"] = out["score_valor_compuesto"] >= umbral_alto

    return out


def pct_mas(fraccion: float) -> str:
    if fraccion >= 0:
        return f"{fraccion * 100:.1f}% más"
    return f"{-fraccion * 100:.1f}% menos"


# ==========================================
# HALLAZGOS DE NEGOCIO
# ==========================================
def hallazgos_negocio(df: pd.DataFrame) -> list[str]:
    hallazgos: list[str] = []

    hc = df["tipo_jugador"] == "Hardcore"
    cas = df["tipo_jugador"] == "Casual"
    if hc.any() and cas.any():
        diff = df.loc[hc, "ingreso_mensual_mxn"].mean() / df.loc[cas, "ingreso_mensual_mxn"].mean() - 1
        hallazgos.append(
            f"Los jugadores Hardcore generan en promedio un {pct_mas(diff)} en ingresos que los Casual "
            "(oportunidad: bundles premium y pases de temporada para Hardcore)."
        )

    pc = df["plataforma"] == "pc"
    regla_pc_hardcore = False
    if hc.any() and pc.any():
        mask = hc & pc
        if mask.any():
            otros_hc = hc & ~pc
            if otros_hc.any():
                r_pc = df.loc[mask, "ingreso_mensual_mxn"].mean()
                r_ot = df.loc[otros_hc, "ingreso_mensual_mxn"].mean()
                if r_ot > 0:
                    d = r_pc / r_ot - 1
                    hallazgos.append(
                        f"Entre Hardcore, los de PC generan un {pct_mas(d)} en ingresos respecto al Hardcore "
                        "en otras plataformas (reforzar tienda/cosméticos en PC)."
                    )
                    regla_pc_hardcore = True
    if not regla_pc_hardcore:
        fre = df["tipo_jugador"] == "Frecuente"
        if fre.any() and cas.any():
            d = df.loc[fre, "ingreso_mensual_mxn"].mean() / df.loc[cas, "ingreso_mensual_mxn"].mean() - 1
            hallazgos.append(
                f"En promedio, un Frecuente genera un {pct_mas(d)} en ingresos que un Casual "
                "(mid-core: potencial con tutoriales y ofertas por tiempo de sesión)."
            )

    riesgo_alto = df["riesgo_abandono"] == "Alto"
    if riesgo_alto.any():
        share = riesgo_alto.mean()
        hallazgos.append(
            f"El {share * 100:.1f}% de la base muestra riesgo de abandono Alto "
            "(bajas horas y bajo ingreso): priorizar campañas de reactivación."
        )

    plat = df.groupby("plataforma", observed=False)["ingreso_mensual_mxn"].sum()
    if not plat.empty:
        top = plat.idxmax()
        conc = plat.max() / plat.sum() if plat.sum() else 0
        hallazgos.append(
            f"La plataforma con mayor facturación agregada es '{top}', concentrando el "
            f"{conc * 100:.1f}% del ingreso total (diversificar o doblar apuesta según estrategia)."
        )

    seg = df.groupby("segmento_valor", observed=False)["ingreso_mensual_mxn"].mean().sort_values(ascending=False)
    if len(seg) >= 2:
        mejor, peor = seg.index[0], seg.index[-1]
        ratio = seg.iloc[0] / seg.iloc[-1] - 1 if seg.iloc[-1] else 0
        hallazgos.append(
            f"El segmento de valor '{mejor}' genera un {pct_mas(ratio)} en ingresos que '{peor}' en promedio: "
            "diseñar CRM y beneficios escalonados hacia Oro/Platino."
        )

    fr_whale = (df["ingreso_mensual_mxn"] >= df["ingreso_mensual_mxn"].quantile(0.75)).mean()
    hallazgos.append(
        f"Aproximadamente el {fr_whale * 100:.1f}% del panel está en el cuartil superior de ingreso "
        "(focus whale-friendly: soporte VIP, acceso anticipado)."
    )

    # Nuevos hallazgos
    if "pais" in df.columns and "estado_jugador" in df.columns:
        activos_pais = (
            df[df["estado_jugador"] == "Activo"]
            .groupby("pais", observed=False)["id_jugador"]
            .count()
            .sort_values(ascending=False)
        )
        if not activos_pais.empty:
            top_pais = activos_pais.index[0]
            pct_activos = (df["estado_jugador"] == "Activo").mean() * 100
            hallazgos.append(
                f"'{top_pais}' concentra la mayor cantidad de jugadores activos "
                f"({activos_pais.iloc[0]}). El {pct_activos:.1f}% de la base está activa."
            )

    if "compras_internas" in df.columns:
        compras_tipo = (
            df.groupby("tipo_jugador", observed=False)["compras_internas"]
            .apply(lambda s: (s == "Sí").mean() * 100)
            .sort_values(ascending=False)
        )
        if not compras_tipo.empty:
            top_tipo_c = compras_tipo.index[0]
            hallazgos.append(
                f"El {compras_tipo.iloc[0]:.1f}% de los jugadores '{top_tipo_c}' realizan compras internas "
                "(top en conversión de microtransacciones)."
            )

    if "calificacion" in df.columns and "nombre_juego" in df.columns:
        calif_juego = df.groupby("nombre_juego", observed=False)["calificacion"].mean().sort_values(ascending=False)
        if not calif_juego.empty:
            top_juego = calif_juego.index[0]
            hallazgos.append(
                f"'{top_juego}' tiene la mejor calificación promedio ({calif_juego.iloc[0]:.2f}/5.0). "
                "Analizar qué elementos de diseño generan mayor satisfacción."
            )

    if "dias_ultima_conexion" in df.columns:
        inactivos_30 = (df["dias_ultima_conexion"] > 30).mean() * 100
        hallazgos.append(
            f"El {inactivos_30:.1f}% de jugadores lleva más de 30 días sin conectarse. "
            "Campaña de reactivación con recompensa de regreso sugerida."
        )

    return hallazgos


def _interpretacion_correlacion_horas_ingreso(r: float) -> str:
    if pd.isna(r):
        return "No es posible calcular la correlación con los datos actuales."
    if r >= 0.5:
        return "Asociación fuerte positiva: campañas que aumenten sesión suelen acompañar ticket."
    if r >= 0.2:
        return "Asociación moderada: hay jugadores que grindean sin pagar y otros que pagan sin grind."
    return "Relación débil en este panel: revisar con más datos o por estratos (plataforma/género)."


# ==========================================
# CONSULTAS Q1–Q10
# ==========================================
def consultas_dataframes(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, float | str]]:
    # Q1 — Ingresos por plataforma
    ing = df.groupby("plataforma", observed=False)["ingreso_mensual_mxn"].agg(["sum", "mean", "count"])
    ing = ing.rename(columns={"sum": "ingreso_total_mxn", "mean": "ingreso_promedio", "count": "jugadores"})
    ing = ing.sort_values("ingreso_total_mxn", ascending=False)

    # Q2 — Engagement por género
    gen = (
        df.groupby("genero_juego", observed=False)
        .agg(
            horas_promedio=("horas_semanales", "mean"),
            ingreso_promedio=("ingreso_mensual_mxn", "mean"),
            jugadores=("id_jugador", "count"),
        )
        .sort_values("horas_promedio", ascending=False)
    )
    gen = gen.copy()
    gen["score_engagement_genero"] = min_max(gen["horas_promedio"]) * 0.5 + min_max(gen["ingreso_promedio"]) * 0.5
    gen = gen.sort_values("score_engagement_genero", ascending=False)

    # Q3 — Ingreso por tipo de jugador
    tipo = (
        df.groupby("tipo_jugador", observed=False)["ingreso_mensual_mxn"]
        .agg(promedio="mean", total="sum", n="count")
        .sort_values("promedio", ascending=False)
    )

    # Q4 — Riesgo de abandono por plataforma
    riesgo = (pd.crosstab(df["plataforma"], df["riesgo_abandono"], normalize="index") * 100).round(1)

    # Q5 — Segmento de valor vs riesgo
    crm = (
        df.groupby("segmento_valor", observed=False)
        .agg(
            jugadores=("id_jugador", "count"),
            ingreso_medio=("ingreso_mensual_mxn", "mean"),
            pct_alto_riesgo=("riesgo_abandono", lambda s: (s == "Alto").mean() * 100),
        )
        .sort_values("ingreso_medio", ascending=False)
    ).round(2)

    r = float(df["horas_semanales"].corr(df["ingreso_mensual_mxn"]))

    tablas = {
        "q1_ingresos_plataforma": ing,
        "q2_genero_engagement": gen,
        "q3_tipo_jugador_ingreso": tipo,
        "q4_riesgo_por_plataforma_pct": riesgo,
        "q5_segmento_valor_crm": crm,
    }

    # Q6 — País con más jugadores activos
    if "pais" in df.columns and "estado_jugador" in df.columns:
        q6 = (
            df.groupby("pais", observed=False)
            .agg(
                total=("id_jugador", "count"),
                activos=("estado_jugador", lambda s: (s == "Activo").sum()),
                inactivos=("estado_jugador", lambda s: (s == "Inactivo").sum()),
                ingreso_promedio=("ingreso_mensual_mxn", "mean"),
                rating_promedio=("calificacion", "mean") if "calificacion" in df.columns else pd.NA,
            )
            .sort_values("activos", ascending=False)
        )
        q6["pct_activos"] = (q6["activos"] / q6["total"] * 100).round(1)
        tablas["q6_pais_activos"] = q6

    # Q7 — Compras internas por tipo de jugador
    if "compras_internas" in df.columns:
        q7 = (
            df.groupby("tipo_jugador", observed=False)
            .agg(
                total=("id_jugador", "count"),
                compran=("compras_internas", lambda s: (s == "Sí").sum()),
                no_compran=("compras_internas", lambda s: (s == "No").sum()),
                ingreso_promedio=("ingreso_mensual_mxn", "mean"),
                monto_promedio_compradores=("ingreso_mensual_mxn", lambda x: x[df["compras_internas"] == "Sí"].mean()),
            )
            .sort_values("compran", ascending=False)
        )
        q7["tasa_conversion"] = (q7["compran"] / q7["total"] * 100).round(1)
        tablas["q7_compras_tipo_jugador"] = q7

    # Q8 — Calificación promedio por juego
    if "calificacion" in df.columns and "nombre_juego" in df.columns:
        q8 = (
            df.groupby("nombre_juego", observed=False)
            .agg(
                calificacion_promedio=("calificacion", "mean"),
                jugadores=("id_jugador", "count"),
                horas_promedio=("horas_semanales", "mean"),
                ingreso_promedio=("ingreso_mensual_mxn", "mean"),
            )
            .sort_values("calificacion_promedio", ascending=False)
        )
        q8["calificacion_promedio"] = q8["calificacion_promedio"].round(2)
        tablas["q8_calificacion_juego"] = q8

    # Q9 — Combinación plataforma × género (ingresos)
    if "plataforma" in df.columns and "genero_juego" in df.columns:
        q9 = pd.pivot_table(
            df,
            values="ingreso_mensual_mxn",
            index="plataforma",
            columns="genero_juego",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        tablas["q9_plataforma_genero_ingresos"] = q9

    # Q10 — Jugadores de alto valor (top 20) con riesgo
    if "alto_valor" in df.columns:
        q10 = (
            df.groupby("alto_valor", observed=False)
            .agg(
                jugadores=("id_jugador", "count"),
                ingreso_promedio=("ingreso_mensual_mxn", "mean"),
                horas_promedio=("horas_semanales", "mean"),
                pct_riesgo_alto=("riesgo_abandono", lambda s: (s == "Alto").mean() * 100),
            )
            .round(2)
        )
        tablas["q10_alto_valor_riesgo"] = q10

    meta = {
        "correlacion_horas_ingreso": r,
        "interpretacion_correlacion": _interpretacion_correlacion_horas_ingreso(r),
    }

    return tablas, meta


def consultas_estrategicas(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CONSULTAS ESTRATEGICAS (resultados sobre datos_gamer_limpios)")
    print("=" * 72)

    tablas, meta = consultas_dataframes(df)

    print("\nQ1) Que plataforma genera mayores ingresos (suma de ingreso mensual)?")
    print(tablas["q1_ingresos_plataforma"].to_string())

    print("\nQ2) Que genero muestra mayor engagement y valor?")
    print(tablas["q2_genero_engagement"].to_string())

    print("\nQ3) Que tipo de jugador genera mas ingresos (promedio y total)?")
    print(tablas["q3_tipo_jugador_ingreso"].to_string())

    print("\nQ4) Distribucion de riesgo de abandono por plataforma (% por fila):")
    print(tablas["q4_riesgo_por_plataforma_pct"].to_string())

    print("\nQ5) Segmento de valor vs intensidad de riesgo alto (%):")
    print(tablas["q5_segmento_valor_crm"].to_string())

    r = meta["correlacion_horas_ingreso"]
    print(f"\nQ6) Correlacion horas vs ingreso:")
    print(f"    correlacion(horas_semanales, ingreso_mensual_mxn) = {r:.3f}")
    if pd.notna(r):
        print(f"    Lectura: {meta['interpretacion_correlacion']}")

    if "q6_pais_activos" in tablas:
        print("\nQ7) Pais con mas jugadores activos:")
        print(tablas["q6_pais_activos"].to_string())

    if "q7_compras_tipo_jugador" in tablas:
        print("\nQ8) Compras internas por tipo de jugador:")
        print(tablas["q7_compras_tipo_jugador"].to_string())

    if "q8_calificacion_juego" in tablas:
        print("\nQ9) Calificacion promedio por videojuego:")
        print(tablas["q8_calificacion_juego"].to_string())

    if "q9_plataforma_genero_ingresos" in tablas:
        print("\nQ10) Combinacion plataforma x genero con mas ingresos:")
        print(tablas["q9_plataforma_genero_ingresos"].to_string())


# ==========================================
# TABLAS OLAP
# ==========================================
def tabla_olap(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pd.pivot_table(
        df,
        values=["ingreso_mensual_mxn", "horas_semanales"],
        index="plataforma",
        columns="genero_juego",
        aggfunc={"ingreso_mensual_mxn": "sum", "horas_semanales": "sum"},
        fill_value=0,
        observed=False,
    )
    return pivot


def tabla_olap_pais(df: pd.DataFrame) -> pd.DataFrame | None:
    if "pais" not in df.columns:
        return None
    pivot = pd.pivot_table(
        df,
        values="ingreso_mensual_mxn",
        index="pais",
        columns="plataforma",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    return pivot


def tabla_olap_juego_genero(df: pd.DataFrame) -> pd.DataFrame | None:
    if "nombre_juego" not in df.columns:
        return None
    pivot = pd.pivot_table(
        df,
        values=["ingreso_mensual_mxn", "calificacion"] if "calificacion" in df.columns else ["ingreso_mensual_mxn"],
        index="nombre_juego",
        columns="genero_juego",
        aggfunc={"ingreso_mensual_mxn": "sum", "calificacion": "mean"} if "calificacion" in df.columns else "sum",
        fill_value=0,
        observed=False,
    )
    return pivot


def tabla_olap_pais_genero(df: pd.DataFrame) -> pd.DataFrame | None:
    if "pais" not in df.columns:
        return None
    pivot = pd.pivot_table(
        df,
        values=["ingreso_mensual_mxn", "horas_semanales"],
        index="pais",
        columns="genero_juego",
        aggfunc={"ingreso_mensual_mxn": "sum", "horas_semanales": "sum"},
        fill_value=0,
        observed=False,
    )
    return pivot


# ==========================================
# KPI
# ==========================================
def calcular_kpis(df: pd.DataFrame) -> dict[str, float | str]:
    kpis: dict[str, float | str] = {}

    kpis["total_jugadores"] = len(df)
    kpis["ingreso_total_mxn"] = round(float(df["ingreso_mensual_mxn"].sum()), 0)
    kpis["ingreso_promedio_mxn"] = round(float(df["ingreso_mensual_mxn"].mean()), 2)
    kpis["horas_promedio"] = round(float(df["horas_semanales"].mean()), 1)
    kpis["pct_riesgo_alto"] = round(float((df["riesgo_abandono"] == "Alto").mean() * 100), 1)
    kpis["corr_horas_ingreso"] = round(float(df["horas_semanales"].corr(df["ingreso_mensual_mxn"])), 3)

    if "estado_jugador" in df.columns:
        kpis["pct_activos"] = round(float((df["estado_jugador"] == "Activo").mean() * 100), 1)
        kpis["pct_inactivos"] = round(float((df["estado_jugador"] == "Inactivo").mean() * 100), 1)

    if "compras_internas" in df.columns:
        kpis["pct_compras_internas"] = round(float((df["compras_internas"] == "Sí").mean() * 100), 1)

    if "calificacion" in df.columns:
        kpis["calificacion_promedio"] = round(float(df["calificacion"].mean()), 2)

    if "nivel_alcanzado" in df.columns:
        kpis["nivel_promedio"] = round(float(df["nivel_alcanzado"].mean()), 0)

    if "pais" in df.columns:
        kpis["pais_top_activos"] = str(
            df[df["estado_jugador"] == "Activo"]["pais"].mode(dropna=True).iloc[0]
            if "estado_jugador" in df.columns
            else df["pais"].mode(dropna=True).iloc[0]
        )

    if "nombre_juego" in df.columns and "calificacion" in df.columns:
        kpis["juego_top_calificacion"] = str(
            df.groupby("nombre_juego", observed=False)["calificacion"].mean().idxmax()
        )

    return kpis


# ==========================================
# DASHBOARD (gráficas)
# ==========================================
def configurar_estilo() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titlesize"] = 14


def dashboard(df: pd.DataFrame) -> None:
    configurar_estilo()
    out_dir = BASE_DIR / "salida_analisis_gamer"
    out_dir.mkdir(exist_ok=True)

    # 01 — Ingreso por plataforma
    fig1, ax1 = plt.subplots()
    ing = df.groupby("plataforma", observed=False)["ingreso_mensual_mxn"].sum().sort_values(ascending=True)
    ing.plot(kind="barh", ax=ax1, color=sns.color_palette("viridis", n_colors=len(ing)))
    ax1.set_xlabel("Ingreso mensual total (MXN)")
    ax1.set_ylabel("Plataforma")
    ax1.set_title("Facturacion agregada por plataforma")
    fig1.tight_layout()
    p1 = out_dir / "01_ingreso_total_por_plataforma.png"
    fig1.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Grafica 1] Archivo: {p1}")

    # 02 — Heatmap plataforma × género
    ingreso_pivot = pd.pivot_table(
        df, values="ingreso_mensual_mxn", index="plataforma", columns="genero_juego",
        aggfunc="sum", fill_value=0, observed=False,
    )
    fig2, ax2 = plt.subplots(figsize=(max(8, ingreso_pivot.shape[1] * 1.2), 5))
    sns.heatmap(ingreso_pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax2, cbar_kws={"label": "MXN (suma)"})
    ax2.set_title("Ingreso mensual por plataforma y genero de juego")
    fig2.tight_layout()
    p2 = out_dir / "02_heatmap_plataforma_genero_ingreso.png"
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Grafica 2] Archivo: {p2}")

    # 03 — Ingreso por tipo de jugador
    fig3, ax3 = plt.subplots()
    orden = [x for x in ["Casual", "Frecuente", "Hardcore"] if x in df["tipo_jugador"].unique()]
    sns.boxplot(data=df, x="tipo_jugador", y="ingreso_mensual_mxn", order=orden, ax=ax3)
    sns.stripplot(data=df, x="tipo_jugador", y="ingreso_mensual_mxn", order=orden, color=".25", alpha=0.7, ax=ax3)
    ax3.set_xlabel("Tipo de jugador")
    ax3.set_ylabel("Ingreso mensual (MXN)")
    ax3.set_title("Distribucion de ingreso segun intensidad de juego")
    fig3.tight_layout()
    p3 = out_dir / "03_ingreso_por_tipo_jugador.png"
    fig3.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print(f"[Grafica 3] Archivo: {p3}")

    # 04 — Jugadores activos por país
    if "pais" in df.columns and "estado_jugador" in df.columns:
        fig4, ax4 = plt.subplots()
        pais_act = (
            df.groupby("pais", observed=False)
            .agg(activos=("estado_jugador", lambda s: (s == "Activo").sum()),
                 inactivos=("estado_jugador", lambda s: (s == "Inactivo").sum()))
            .sort_values("activos", ascending=False)
        )
        pais_act.plot(kind="barh", stacked=True, ax=ax4, color=["#2ecc71", "#e74c3c"])
        ax4.set_xlabel("Cantidad de jugadores")
        ax4.set_ylabel("Pais")
        ax4.set_title("Jugadores activos e inactivos por pais")
        ax4.legend(["Activo", "Inactivo"])
        fig4.tight_layout()
        p4 = out_dir / "04_activos_por_pais.png"
        fig4.savefig(p4, dpi=120, bbox_inches="tight")
        plt.close(fig4)
        print(f"[Grafica 4] Archivo: {p4}")

    # 05 — Calificación promedio por juego
    if "calificacion" in df.columns and "nombre_juego" in df.columns:
        fig5, ax5 = plt.subplots()
        calif = df.groupby("nombre_juego", observed=False)["calificacion"].mean().sort_values(ascending=True)
        colors = sns.color_palette("RdYlGn", n_colors=len(calif))
        calif.plot(kind="barh", ax=ax5, color=colors)
        ax5.set_xlabel("Calificacion promedio (1-5)")
        ax5.set_ylabel("Juego")
        ax5.set_title("Calificacion promedio por videojuego")
        for i, v in enumerate(calif.values):
            ax5.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
        fig5.tight_layout()
        p5 = out_dir / "05_calificacion_por_juego.png"
        fig5.savefig(p5, dpi=120, bbox_inches="tight")
        plt.close(fig5)
        print(f"[Grafica 5] Archivo: {p5}")

    # 06 — Compras internas por tipo de jugador
    if "compras_internas" in df.columns:
        fig6, ax6 = plt.subplots()
        compras = (
            df.groupby("tipo_jugador", observed=False)["compras_internas"]
            .apply(lambda s: (s == "Sí").mean() * 100)
            .sort_values(ascending=True)
        )
        colors6 = sns.color_palette("Blues_d", n_colors=len(compras))
        compras.plot(kind="barh", ax=ax6, color=colors6)
        ax6.set_xlabel("% que realiza compras internas")
        ax6.set_ylabel("Tipo de jugador")
        ax6.set_title("Tasa de compras internas por tipo de jugador")
        for i, v in enumerate(compras.values):
            ax6.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=10)
        fig6.tight_layout()
        p6 = out_dir / "06_compras_internas_por_tipo.png"
        fig6.savefig(p6, dpi=120, bbox_inches="tight")
        plt.close(fig6)
        print(f"[Grafica 6] Archivo: {p6}")

    # 07 — Heatmap país × género (ingreso)
    if "pais" in df.columns:
        pg_pivot = pd.pivot_table(
            df, values="ingreso_mensual_mxn", index="pais", columns="genero_juego",
            aggfunc="sum", fill_value=0, observed=False,
        )
        if not pg_pivot.empty:
            fig7, ax7 = plt.subplots(figsize=(max(8, pg_pivot.shape[1] * 1.2), max(6, pg_pivot.shape[0] * 0.5)))
            sns.heatmap(pg_pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax7, cbar_kws={"label": "MXN (suma)"})
            ax7.set_title("Ingreso mensual por pais y genero de juego")
            fig7.tight_layout()
            p7 = out_dir / "07_heatmap_pais_genero_ingreso.png"
            fig7.savefig(p7, dpi=120, bbox_inches="tight")
            plt.close(fig7)
            print(f"[Grafica 7] Archivo: {p7}")

    # 08 — Scatter nivel vs ingreso
    if "nivel_alcanzado" in df.columns:
        fig8, ax8 = plt.subplots()
        scatter = ax8.scatter(
            df["nivel_alcanzado"].astype(float),
            df["ingreso_mensual_mxn"],
            c=df["horas_semanales"],
            cmap="viridis",
            alpha=0.6,
        )
        ax8.set_xlabel("Nivel alcanzado")
        ax8.set_ylabel("Ingreso mensual (MXN)")
        ax8.set_title("Nivel maximo vs Ingreso mensual")
        plt.colorbar(scatter, ax=ax8, label="Horas semanales")
        fig8.tight_layout()
        p8 = out_dir / "08_nivel_vs_ingreso.png"
        fig8.savefig(p8, dpi=120, bbox_inches="tight")
        plt.close(fig8)
        print(f"[Grafica 8] Archivo: {p8}")

    # 09 — Distribución de calificaciones por género
    if "calificacion" in df.columns and "genero_juego" in df.columns:
        fig9, ax9 = plt.subplots()
        sns.boxplot(data=df, x="genero_juego", y="calificacion", ax=ax9, palette="Set3")
        ax9.set_xlabel("Genero de juego")
        ax9.set_ylabel("Calificacion (1-5)")
        ax9.set_title("Distribucion de calificaciones por genero")
        ax9.tick_params(axis="x", rotation=45)
        fig9.tight_layout()
        p9 = out_dir / "09_calificacion_por_genero.png"
        fig9.savefig(p9, dpi=120, bbox_inches="tight")
        plt.close(fig9)
        print(f"[Grafica 9] Archivo: {p9}")

    # 10 — Riesgo de abandono por país
    if "pais" in df.columns:
        fig10, ax10 = plt.subplots()
        riesgo_pais = (pd.crosstab(df["pais"], df["riesgo_abandono"], normalize="index") * 100).round(1)
        if "Alto" in riesgo_pais.columns:
            riesgo_pais = riesgo_pais.sort_values("Alto", ascending=True)
        riesgo_pais.plot(kind="barh", stacked=True, ax=ax10, color=["#2ecc71", "#f39c12", "#e74c3c"])
        ax10.set_xlabel("% de jugadores")
        ax10.set_ylabel("Pais")
        ax10.set_title("Distribucion de riesgo de abandono por pais")
        ax10.legend(["Bajo", "Medio", "Alto"])
        fig10.tight_layout()
        p10 = out_dir / "10_riesgo_abandono_por_pais.png"
        fig10.savefig(p10, dpi=120, bbox_inches="tight")
        plt.close(fig10)
        print(f"[Grafica 10] Archivo: {p10}")


# ==========================================
# EXPORTACIONES ADICIONALES
# ==========================================
def exportar_olaps(df: pd.DataFrame) -> None:
    out_dir = BASE_DIR / "salida_analisis_gamer"
    out_dir.mkdir(exist_ok=True)

    p = tabla_olap_pais(df)
    if p is not None:
        p.to_csv(out_dir / "olap_pais_plataforma.csv")
        print(f"Exportado: olap_pais_plataforma.csv")

    p = tabla_olap_juego_genero(df)
    if p is not None:
        p.to_csv(out_dir / "olap_juego_genero.csv")
        print(f"Exportado: olap_juego_genero.csv")

    p = tabla_olap_pais_genero(df)
    if p is not None:
        p.to_csv(out_dir / "olap_pais_genero.csv")
        print(f"Exportado: olap_pais_genero.csv")


def exportar_kpis(df: pd.DataFrame) -> None:
    kpis = calcular_kpis(df)
    kpi_df = pd.DataFrame(list(kpis.items()), columns=["KPI", "Valor"])
    out_dir = BASE_DIR / "salida_analisis_gamer"
    out_dir.mkdir(exist_ok=True)
    kpi_df.to_csv(out_dir / "kpis_negocio.csv", index=False, encoding="utf-8")
    print("\n=== KPIs DE NEGOCIO ===")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
    print(f"\nKPIs exportados: {out_dir / 'kpis_negocio.csv'}")


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    df = cargar_limpio()
    df = agregar_metricas_valor(df)

    print("=== Mineria: score de valor y segmento ===")
    cols = [c for c in [
        "id_jugador", "plataforma", "genero_juego", "nombre_juego", "pais",
        "horas_semanales", "ingreso_mensual_mxn", "nivel_alcanzado",
        "calificacion", "compras_internas", "estado_jugador",
        "tipo_jugador", "riesgo_abandono",
        "score_valor_jugador", "score_valor_compuesto",
        "segmento_valor", "rank_valor", "alto_valor",
    ] if c in df.columns]
    print(df[cols].sort_values("rank_valor").to_string(index=False))

    print("\n=== KPIs DE NEGOCIO ===")
    kpis = calcular_kpis(df)
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    print("\n=== HALLAZGOS / REGLAS DE NEGOCIO ===")
    for i, h in enumerate(hallazgos_negocio(df), start=1):
        print(f"{i}. {h}")

    consultas_estrategicas(df)

    # OLAP
    pivot = tabla_olap(df)
    print("\n" + "=" * 72)
    print("TABLA OLAP: plataforma x genero")
    print("=" * 72)
    print(pivot.to_string())
    pivot.to_csv(BASE_DIR / "olap_plataforma_genero.csv")
    print(f"\nExportado: olap_plataforma_genero.csv")

    exportar_olaps(df)
    exportar_kpis(df)

    print("\n=== DASHBOARD (graficas + interpretacion) ===")
    dashboard(df)
    print("\nListo.")


if __name__ == "__main__":
    main()
