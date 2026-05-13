"""
PixelQuest Studios — Inteligencia de negocio sobre datos_gamer_limpios.csv

Minería: ranking/segmento de valor (horas + gasto).
Consultas estratégicas, tabla OLAP (pivot) y dashboard con interpretación.

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

# Peso relativo: PixelQuest prioriza monetización directa frente a tiempo de sesión en el score compuesto.
PESO_HORAS_EN_SCORE = 0.35
PESO_GASTO_EN_SCORE = 0.65


def cargar_limpio() -> pd.DataFrame:
    """
    Propósito (PixelQuest): Materializa el dataset ya gobernado por el ETL como única fuente de verdad analítica.

    Regla de negocio: Solo se consume `datos_gamer_limpios.csv` — ningún cálculo de valor opera sobre filas crudas sin reglas de limpieza.
    """
    if not ARCHIVO_LIMPIO.is_file():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_LIMPIO}. Ejecute primero procesar_datos_gamer.py."
        )
    return pd.read_csv(ARCHIVO_LIMPIO, encoding="utf-8")


def min_max(s: pd.Series) -> pd.Series:
    """
    Propósito (PixelQuest): Pone horas y gasto en escala común para un índice de valor comparable entre jugadores.

    Regla de negocio: Rango cero o degenerado → neutro 0.5 para no dominar el score cuando la muestra no discrimina.
    """
    r = s.max() - s.min()
    if r == 0 or np.isnan(r):
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / r


def agregar_metricas_valor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Propósito (PixelQuest): Enriquece el panel con segmentación de valor, ranking y proxy de afinidad para CRM y ofertas.

    Regla de negocio:
    - score_valor_jugador = 0.35×horas_norm + 0.65×gasto_norm (mayor peso al ARPU).
    - segmento_valor en cuartiles sobre el score (Bronce → Platino).
    - indice_afinidad_genero: gasto/hora como proxy de «dispuesto a pagar por hora de engagement» cuando horas>0.
    """
    out = df.copy()
    out["horas_norm"] = min_max(out["horas_semanales"])
    out["gasto_norm"] = min_max(out["gasto_mensual_usd"])
    out["score_valor_jugador"] = (
        PESO_HORAS_EN_SCORE * out["horas_norm"]
        + PESO_GASTO_EN_SCORE * out["gasto_norm"]
    )
    bins = [0, 0.25, 0.5, 0.75, 1.0001]
    labels = ["Bronce", "Plata", "Oro", "Platino"]
    out["segmento_valor"] = pd.cut(
        out["score_valor_jugador"], bins=bins, labels=labels, include_lowest=True
    ).astype("string")
    out["rank_valor"] = out["score_valor_jugador"].rank(ascending=False, method="min").astype(int)
    h = out["horas_semanales"].replace(0, np.nan)
    out["indice_afinidad_genero"] = out["gasto_mensual_usd"] / h
    out["indice_afinidad_genero"] = out["indice_afinidad_genero"].replace(
        [np.inf, -np.inf], np.nan
    )
    return out


def pct_mas(fraccion: float) -> str:
    """
    Propósito (PixelQuest): Formatea diferencias relativas en narrativa ejecutiva para hallazgos y bullets de negocio.

    Regla de negocio: Texto estable («X% más/menos») para informes reproducibles sin depender del locale numérico.
    """
    if fraccion >= 0:
        return f"{fraccion * 100:.1f}% más"
    return f"{-fraccion * 100:.1f}% menos"


def hallazgos_negocio(df: pd.DataFrame) -> list[str]:
    """
    Propósito (PixelQuest): Produce frases accionables (priorización de canales, CRM y monetización) a partir del panel limpio.

    Regla de negocio: Orquesta comparaciones condicionales (Hardcore vs Casual, PC vs resto, concentración por plataforma,
    gap entre segmentos de valor, cuartil superior de gasto) sin asumir causalidad — solo lecturas de la muestra actual.
    """
    hallazgos: list[str] = []

    hc = df["tipo_jugador"] == "Hardcore"
    cas = df["tipo_jugador"] == "Casual"
    if hc.any() and cas.any():
        diff = df.loc[hc, "gasto_mensual_usd"].mean() / df.loc[cas, "gasto_mensual_usd"].mean() - 1
        hallazgos.append(
            f"Los jugadores Hardcore gastan en promedio un {pct_mas(diff)} que los Casual "
            "(oportunidad: bundles premium y pases de temporada para Hardcore)."
        )

    pc = df["plataforma"] == "pc"
    regla_pc_hardcore = False
    if hc.any() and pc.any():
        mask = hc & pc
        if mask.any():
            otros_hc = hc & ~pc
            if otros_hc.any():
                r_pc = df.loc[mask, "gasto_mensual_usd"].mean()
                r_ot = df.loc[otros_hc, "gasto_mensual_usd"].mean()
                if r_ot > 0:
                    d = r_pc / r_ot - 1
                    hallazgos.append(
                        f"Entre Hardcore, los de PC gastan un {pct_mas(d)} respecto al Hardcore "
                        "en otras plataformas (reforzar tienda/cosméticos en PC)."
                    )
                    regla_pc_hardcore = True
    if not regla_pc_hardcore:
        fre = df["tipo_jugador"] == "Frecuente"
        if fre.any() and cas.any():
            d = df.loc[fre, "gasto_mensual_usd"].mean() / df.loc[cas, "gasto_mensual_usd"].mean() - 1
            hallazgos.append(
                f"En promedio, un Frecuente gasta un {pct_mas(d)} que un Casual "
                "(mid-core: potencial con tutoriales y ofertas por tiempo de sesión)."
            )

    riesgo_alto = df["riesgo_abandono"] == "Alto"
    if riesgo_alto.any():
        share = riesgo_alto.mean()
        hallazgos.append(
            f"El {share * 100:.1f}% de la base muestra riesgo de abandono Alto (bajas horas y bajo gasto): "
            "priorizar campañas de reactivación y onboarding ligero."
        )

    plat = df.groupby("plataforma", observed=False)["gasto_mensual_usd"].sum()
    if not plat.empty:
        top = plat.idxmax()
        conc = plat.max() / plat.sum() if plat.sum() else 0
        hallazgos.append(
            f"La plataforma con mayor facturación agregada es '{top}', concentrando el "
            f"{conc * 100:.1f}% del gasto total del sample (diversificar o doblar apuesta según estrategia)."
        )

    seg = df.groupby("segmento_valor", observed=False)["gasto_mensual_usd"].mean().sort_values(
        ascending=False
    )
    if len(seg) >= 2:
        mejor, peor = seg.index[0], seg.index[-1]
        ratio = seg.iloc[0] / seg.iloc[-1] - 1 if seg.iloc[-1] else 0
        hallazgos.append(
            f"El segmento de valor '{mejor}' gasta un {pct_mas(ratio)} que '{peor}' en promedio: "
            "diseñar CRM y beneficios escalonados hacia Oro/Platino."
        )

    fr_whale = (df["gasto_mensual_usd"] >= df["gasto_mensual_usd"].quantile(0.75)).mean()
    hallazgos.append(
        f"Aproximadamente el {fr_whale * 100:.1f}% del panel está en el cuartil superior de gasto "
        "(focus whale-friendly: soporte VIP, acceso anticipado)."
    )

    return hallazgos


def _interpretacion_correlacion_horas_gasto(r: float) -> str:
    """
    Propósito (PixelQuest): Traduce el coeficiente Pearson entre horas y gasto en decisiones de campaña coordinadas.

    Regla de negocio: Umbrales 0.5 / 0.2 segmentan fuerza de asociación; por debajo se recomienda estratificar antes de actuar.
    """
    if pd.isna(r):
        return "No es posible calcular la correlación con los datos actuales."
    if r >= 0.5:
        return (
            "Asociación fuerte positiva: campañas que aumenten sesión suelen acompañar ticket; "
            "coordinar retención y monetización."
        )
    if r >= 0.2:
        return (
            "Asociación moderada: hay jugadores que «grindean sin pagar» y otros que «pagan sin grind»; "
            "personalizar ofertas según segmento_valor."
        )
    return (
        "Relación débil en este panel: revisar con más datos o por estratos (plataforma/género) "
        "antes de inferir causalidad."
    )


def consultas_dataframes(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, float | str]]:
    """
    Propósito (PixelQuest): Empaqueta las consultas Q1–Q5 y metadatos Q6 para el panel Streamlit y CLI.

    Regla de negocio:
    - Q1: facturación y ARPU por plataforma.
    - Q2: engagement por género con score 50/50 horas vs gasto tras min_max.
    - Q4: riesgo normalizado por fila (%) para comparar perfiles entre plataformas con bases distintas.
    - Q5: % alto riesgo dentro de cada segmento_valor para priorizar retención en tier alto.
    """
    ing = df.groupby("plataforma", observed=False)["gasto_mensual_usd"].agg(["sum", "mean", "count"])
    ing = ing.rename(columns={"sum": "gasto_total_usd", "mean": "gasto_promedio", "count": "jugadores"})
    ing = ing.sort_values("gasto_total_usd", ascending=False)

    gen = (
        df.groupby("genero_juego", observed=False)
        .agg(
            horas_promedio=("horas_semanales", "mean"),
            gasto_promedio=("gasto_mensual_usd", "mean"),
            jugadores=("id_jugador", "count"),
        )
        .sort_values("horas_promedio", ascending=False)
    )
    gen = gen.copy()
    gen["score_engagement_genero"] = min_max(gen["horas_promedio"]) * 0.5 + min_max(gen["gasto_promedio"]) * 0.5
    gen = gen.sort_values("score_engagement_genero", ascending=False)

    tipo = (
        df.groupby("tipo_jugador", observed=False)["gasto_mensual_usd"]
        .agg(promedio="mean", total="sum", n="count")
        .sort_values("promedio", ascending=False)
    )

    riesgo = (pd.crosstab(df["plataforma"], df["riesgo_abandono"], normalize="index") * 100).round(1)

    crm = (
        df.groupby("segmento_valor", observed=False)
        .agg(
            jugadores=("id_jugador", "count"),
            gasto_medio=("gasto_mensual_usd", "mean"),
            pct_alto_riesgo=("riesgo_abandono", lambda s: (s == "Alto").mean() * 100),
        )
        .sort_values("gasto_medio", ascending=False)
    ).round(2)

    r = float(df["horas_semanales"].corr(df["gasto_mensual_usd"]))

    tablas = {
        "q1_ingresos_plataforma": ing,
        "q2_genero_engagement": gen,
        "q3_tipo_jugador_gasto": tipo,
        "q4_riesgo_por_plataforma_pct": riesgo,
        "q5_segmento_valor_crm": crm,
    }
    meta = {
        "correlacion_horas_gasto": r,
        "interpretacion_correlacion": _interpretacion_correlacion_horas_gasto(r),
    }
    return tablas, meta


def consultas_estrategicas(df: pd.DataFrame) -> None:
    """
    Propósito (PixelQuest): Salida batch en consola para auditoría y demos sin UI.

    Regla de negocio: Mismas definiciones que `consultas_dataframes` — una sola verdad para terminal y dashboard.
    """
    print("\n" + "=" * 72)
    print("CONSULTAS ESTRATÉGICAS (resultados sobre datos_gamer_limpios)")
    print("=" * 72)

    tablas, meta = consultas_dataframes(df)
    ing = tablas["q1_ingresos_plataforma"]
    gen = tablas["q2_genero_engagement"]
    tipo = tablas["q3_tipo_jugador_gasto"]
    riesgo = tablas["q4_riesgo_por_plataforma_pct"]
    crm = tablas["q5_segmento_valor_crm"]
    r = meta["correlacion_horas_gasto"]

    print("\nQ1) ¿Qué plataforma genera mayores ingresos (suma de gasto mensual)?")
    print(ing.to_string())

    print(
        "\nQ2) ¿Qué género muestra mayor engagement y valor? "
        "(Proxy sin encuesta: media de horas y gasto; columna score_engagement_genero)"
    )
    print(gen.to_string())

    print("\nQ3) ¿Qué tipo de jugador gasta más (promedio y total)?")
    print(tipo.to_string())

    print("\nQ4) Distribución de riesgo de abandono por plataforma (% por fila):")
    print(riesgo.to_string())

    print("\nQ5) Segmento de valor vs intensidad de riesgo alto (%):")
    print(crm.to_string())

    print(
        "\nQ6) ¿Horas de juego y gasto van de la mano? (correlación Pearson en la muestra)"
    )
    print(f"    correlacion(horas_semanales, gasto_mensual_usd) = {r:.3f}")
    if pd.notna(r):
        print(f"    Lectura: {meta['interpretacion_correlacion']}")


def tabla_olap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Propósito (PixelQuest): Cubo plataforma × género con suma de gasto y horas para análisis multidimensional y export.

    Regla de negocio: Agregación sum — ingresos y carga de juego total por celda; celdas vacías en la muestra → 0.
    """
    pivot = pd.pivot_table(
        df,
        values=["gasto_mensual_usd", "horas_semanales"],
        index="plataforma",
        columns="genero_juego",
        aggfunc={"gasto_mensual_usd": "sum", "horas_semanales": "sum"},
        fill_value=0,
        observed=False,
    )
    return pivot


def configurar_estilo() -> None:
    """
    Propósito (PixelQuest): Unifica estética de figuras exportadas para informes externos a Streamlit.

    Regla de negocio: Tema fijo (whitegrid + context talk) para comparabilidad visual entre ejecuciones del script.
    """
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titlesize"] = 14


def dashboard(df: pd.DataFrame) -> None:
    """
    Propósito (PixelQuest): Genera PNG con narrativa de negocio en consola para storytelling offline.

    Regla de negocio: Mismos cruces que el panel (plataforma, género, tipo_jugador) — decisiones alineadas con la app web.
    """
    configurar_estilo()
    out_dir = BASE_DIR / "salida_analisis_gamer"
    out_dir.mkdir(exist_ok=True)

    fig1, ax1 = plt.subplots()
    ing = df.groupby("plataforma", observed=False)["gasto_mensual_usd"].sum().sort_values(ascending=True)
    ing.plot(kind="barh", ax=ax1, color=sns.color_palette("viridis", n_colors=len(ing)))
    ax1.set_xlabel("Gasto mensual total (USD)")
    ax1.set_ylabel("Plataforma")
    ax1.set_title("Facturación agregada por plataforma")
    fig1.tight_layout()
    p1 = out_dir / "01_gasto_total_por_plataforma.png"
    fig1.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    print(
        "\n[Gráfica 1] Interpretación negocio: la plataforma con la barra más larga concentra el ingreso "
        "directo del panel; sirve para priorizar alianzas, inventario de ofertas y soporte. Si una "
        "plataforma tiene muchos usuarios pero poca barra, hay oportunidad de monetización (conversion funnels)."
    )
    print(f"    Archivo: {p1}")

    gasto_pivot = pd.pivot_table(
        df,
        values="gasto_mensual_usd",
        index="plataforma",
        columns="genero_juego",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    fig2, ax2 = plt.subplots(figsize=(max(8, gasto_pivot.shape[1] * 1.2), 5))
    sns.heatmap(
        gasto_pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax2,
        cbar_kws={"label": "USD (suma)"},
    )
    ax2.set_title("Mapa de calor: gasto mensual por plataforma y género de juego")
    fig2.tight_layout()
    p2 = out_dir / "02_heatmap_plataforma_genero_gasto.png"
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(
        "\n[Gráfica 2] Interpretación negocio: las celdas más intensas señalan combinaciones plataforma–género "
        "que ya monetizan; pueden ampliarse con bundles cruzados. Las celdas claras con muchas horas "
        "(ver pivot de horas) indican interés sin conversión — A/B en precio o catálogo."
    )
    print(f"    Archivo: {p2}")

    fig3, ax3 = plt.subplots()
    orden = ["Casual", "Frecuente", "Hardcore"]
    orden = [x for x in orden if x in df["tipo_jugador"].unique()]
    sns.boxplot(data=df, x="tipo_jugador", y="gasto_mensual_usd", order=orden, ax=ax3)
    sns.stripplot(
        data=df,
        x="tipo_jugador",
        y="gasto_mensual_usd",
        order=orden,
        color=".25",
        alpha=0.7,
        ax=ax3,
    )
    ax3.set_xlabel("Tipo de jugador")
    ax3.set_ylabel("Gasto mensual (USD)")
    ax3.set_title("Distribución de gasto según intensidad de juego")
    fig3.tight_layout()
    p3 = out_dir / "03_gasto_por_tipo_jugador.png"
    fig3.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print(
        "\n[Gráfica 3] Interpretación negocio: si la mediana sube de Casual a Hardcore, conviene "
        "anclar retención en contenido de alta frecuencia de sesión y monetización ética (pase batalla). "
        "Si hay Hardcore con gasto bajo (puntos abajo), probar upsell por logros o competición."
    )
    print(f"    Archivo: {p3}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    # ==========================================
    # CARGA
    # ==========================================
    df = cargar_limpio()

    # ==========================================
    # MINERÍA (valor / segmento / rank)
    # ==========================================
    df = agregar_metricas_valor(df)

    print("=== Minería: score de valor y segmento ===")
    cols = [
        "id_jugador",
        "plataforma",
        "genero_juego",
        "horas_semanales",
        "gasto_mensual_usd",
        "tipo_jugador",
        "score_valor_jugador",
        "segmento_valor",
        "rank_valor",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values("rank_valor").to_string(index=False))

    print("\n=== HALLAZGOS / REGLAS DE NEGOCIO (min. 5) ===")
    for i, h in enumerate(hallazgos_negocio(df)[:8], start=1):
        print(f"{i}. {h}")

    # ==========================================
    # CONSULTAS
    # ==========================================
    consultas_estrategicas(df)

    # ==========================================
    # OLAP
    # ==========================================
    pivot = tabla_olap(df)
    print("\n" + "=" * 72)
    print("TABLA OLAP (pivot_table): plataforma × género — suma de gasto y horas")
    print("=" * 72)
    print(pivot.to_string())

    pivot.to_csv(BASE_DIR / "olap_plataforma_genero.csv")
    print(f"\nPivot exportado: {BASE_DIR / 'olap_plataforma_genero.csv'}")

    # ==========================================
    # VISUALIZACIÓN (export estático)
    # ==========================================
    print("\n=== DASHBOARD (gráficas + interpretación) ===")
    dashboard(df)
    print("\nListo.")


if __name__ == "__main__":
    main()
