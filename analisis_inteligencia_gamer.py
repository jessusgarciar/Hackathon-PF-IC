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

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_LIMPIO = BASE_DIR / "datos_gamer_limpios.csv"

# Peso del gasto vs horas en el score de valor (negocio monetiza más el gasto directo)
PESO_HORAS_EN_SCORE = 0.35
PESO_GASTO_EN_SCORE = 0.65


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
    # Proxy de «calificación» / afinidad sin encuesta: engagement + disposición a pagar
    h = out["horas_semanales"].replace(0, np.nan)
    out["indice_afinidad_genero"] = out["gasto_mensual_usd"] / h
    out["indice_afinidad_genero"] = out["indice_afinidad_genero"].replace(
        [np.inf, -np.inf], np.nan
    )
    return out


def pct_mas(fraccion: float) -> str:
    if fraccion >= 0:
        return f"{fraccion * 100:.1f}% más"
    return f"{-fraccion * 100:.1f}% menos"


def hallazgos_negocio(df: pd.DataFrame) -> list[str]:
    """Hallazgos accionables (se enumeran al imprimir)."""
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


def consultas_estrategicas(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CONSULTAS ESTRATÉGICAS (resultados sobre datos_gamer_limpios)")
    print("=" * 72)

    # Q1: Ingresos por plataforma
    ing = df.groupby("plataforma", observed=False)["gasto_mensual_usd"].agg(["sum", "mean", "count"])
    ing = ing.rename(columns={"sum": "gasto_total_usd", "mean": "gasto_promedio", "count": "jugadores"})
    print("\nQ1) ¿Qué plataforma genera mayores ingresos (suma de gasto mensual)?")
    print(ing.sort_values("gasto_total_usd", ascending=False).to_string())

    # Q2: «Calificación» por género — proxy: horas promedio (engagement) + gasto promedio
    gen = (
        df.groupby("genero_juego", observed=False)
        .agg(
            horas_promedio=("horas_semanales", "mean"),
            gasto_promedio=("gasto_mensual_usd", "mean"),
            jugadores=("id_jugador", "count"),
        )
        .sort_values("horas_promedio", ascending=False)
    )
    gen["score_engagement_genero"] = min_max(gen["horas_promedio"]) * 0.5 + min_max(
        gen["gasto_promedio"]
    ) * 0.5
    print(
        "\nQ2) ¿Qué género muestra mayor engagement y valor? "
        "(Proxy sin encuesta: media de horas y gasto; columna score_engagement_genero)"
    )
    print(gen.sort_values("score_engagement_genero", ascending=False).to_string())

    # Q3: Tipo de jugador que más gasta
    tipo = (
        df.groupby("tipo_jugador", observed=False)["gasto_mensual_usd"]
        .agg(promedio="mean", total="sum", n="count")
        .sort_values("promedio", ascending=False)
    )
    print("\nQ3) ¿Qué tipo de jugador gasta más (promedio y total)?")
    print(tipo.to_string())

    # Q4: Riesgo de abandono por plataforma
    riesgo = pd.crosstab(df["plataforma"], df["riesgo_abandono"], normalize="index") * 100
    print("\nQ4) Distribución de riesgo de abandono por plataforma (% por fila):")
    print(riesgo.round(1).to_string())

    # Q5: Segmento de valor vs retención
    crm = (
        df.groupby("segmento_valor", observed=False)
        .agg(
            jugadores=("id_jugador", "count"),
            gasto_medio=("gasto_mensual_usd", "mean"),
            pct_alto_riesgo=("riesgo_abandono", lambda s: (s == "Alto").mean() * 100),
        )
        .sort_values("gasto_medio", ascending=False)
    )
    print("\nQ5) Segmento de valor vs intensidad de riesgo alto (%):")
    print(crm.round(2).to_string())

    r = df["horas_semanales"].corr(df["gasto_mensual_usd"])
    print(
        "\nQ6) ¿Horas de juego y gasto van de la mano? (correlación Pearson en la muestra)"
    )
    print(f"    correlacion(horas_semanales, gasto_mensual_usd) = {r:.3f}")
    if pd.notna(r):
        if r >= 0.5:
            print(
                "    Lectura: asociación fuerte positiva — campañas que aumenten sesión suelen "
                "acompañar ticket; coordinar retención y monetización."
            )
        elif r >= 0.2:
            print(
                "    Lectura: asociación moderada — hay jugadores 'grindean sin pagar' y 'pagan sin grind'; "
                "personalizar ofertas según segmento_valor."
            )
        else:
            print(
                "    Lectura: relación débil en este panel — revisar con más datos o por estratos "
                "(plataforma/género) antes de inferir causalidad."
            )


def tabla_olap(df: pd.DataFrame) -> pd.DataFrame:
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
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titlesize"] = 14


def dashboard(df: pd.DataFrame) -> None:
    configurar_estilo()
    out_dir = BASE_DIR / "salida_analisis_gamer"
    out_dir.mkdir(exist_ok=True)

    # Gráfica 1: Gasto total por plataforma
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

    # Gráfica 2: Heatmap cruce plataforma × género (gasto)
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

    # Gráfica 3: Gasto por tipo de jugador (caja + puntos)
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

    df = cargar_limpio()
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

    consultas_estrategicas(df)

    pivot = tabla_olap(df)
    print("\n" + "=" * 72)
    print("TABLA OLAP (pivot_table): plataforma × género — suma de gasto y horas")
    print("=" * 72)
    print(pivot.to_string())

    pivot.to_csv(BASE_DIR / "olap_plataforma_genero.csv")
    print(f"\nPivot exportado: {BASE_DIR / 'olap_plataforma_genero.csv'}")

    print("\n=== DASHBOARD (gráficas + interpretación) ===")
    dashboard(df)
    print("\nListo.")


if __name__ == "__main__":
    main()
