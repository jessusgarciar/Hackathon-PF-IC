"""
PixelQuest Studios — Pipeline de limpieza y enriquecimiento de datos de jugadores.

Entrada:  datos_gamer.csv (mismo directorio que este script)
Salida:  datos_gamer_limpios.csv

Requisito: pip install pandas
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ==========================================
# CONFIGURACIÓN Y RUTAS
# ==========================================
# Rutas relativas al script: el pipeline debe ejecutarse en cualquier entorno sin rutas absolutas.
BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = BASE_DIR / "datos_gamer.csv"
ARCHIVO_SALIDA = BASE_DIR / "datos_gamer_limpios.csv"

# Si el CSV usa nombres distintos a los esperados, centralizar el mapeo aquí para no bifurcar lógica downstream.
RENOMBRAR_COLUMNAS: dict[str, str] = {
    # "horas_jugadas": "horas_semanales",
    # "gasto_total": "gasto_mensual_usd",
}

# Umbrales alineados con hipótesis de negocio (retención vs monetización); revisar con distribución real.
# tipo_jugador: bins en asignar_tipo_jugador (Casual / Frecuente / Hardcore).
# riesgo_abandono: bandas disjuntas para no dejar la clase intermedia vacía en muestras pequeñas.
UMBRAL_HORAS_BAJA = 10
UMBRAL_GASTO_BAJO = 10.0
UMBRAL_HORAS_ALTA = 25
UMBRAL_GASTO_ALTO = 40.0


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """
    Propósito (PixelQuest): Asegura lectura estable del inventario de jugadores para el ETL sin corrupción de caracteres.

    Regla de negocio: Se fuerza UTF-8 para preservar nombres, plataformas y géneros en idioma local en reportes CRM.
    """
    return pd.read_csv(ruta, encoding="utf-8")


def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Propósito (PixelQuest): Homogeniza la base antes de segmentación (tipo_jugador, riesgo) y handoff al BI.

    Regla de negocio:
    - Imputación numérica por mediana dentro de plataforma (fallback global) para no inflar gasto/horas con ceros
      ni sesgar churn comparando perfiles de ecosistemas distintos.
    - Dedup por id_jugador conservando la última fila como «verdad operativa» más reciente.
    - Texto y sinónimos unificados para que ingresos y OLAP por plataforma/género sean comparables en el tiempo.
    """
    out = df.copy()

    # ==========================================
    # LIMPIEZA — Normalización léxica (plataforma / género)
    # ==========================================
    # Unificar antes de imputar: la mediana por plataforma solo tiene sentido si el valor de plataforma es estable.
    for col in ("plataforma", "genero_juego"):
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )

    mapa_plataforma = {
        "pc": "pc",
        "steam": "pc",
        "playstation": "consola_sony",
        "ps5": "consola_sony",
        "play station 5": "consola_sony",
        "xbox": "consola_microsoft",
        "nintendo switch": "nintendo_switch",
        "switch": "nintendo_switch",
        "mobile": "movil",
        "movil": "movil",
    }
    if "plataforma" in out.columns:
        out["plataforma"] = out["plataforma"].replace(mapa_plataforma)

    mapa_genero = {
        "battle royale": "battle_royale",
        "rpg": "rpg",
        "aventura": "aventura",
        "adventure": "aventura",
        "shooter": "shooter",
        "moba": "moba",
        "sports": "deportes",
        "deportes": "deportes",
        "puzzle": "puzzle",
    }
    if "genero_juego" in out.columns:
        out["genero_juego"] = out["genero_juego"].replace(mapa_genero)

    # ==========================================
    # LIMPIEZA — Tipos numéricos para reglas por umbrales
    # ==========================================
    # Coerce: filas mal tipadas no deben romper bins de Hardcore ni el score downstream; quedan como NA y se imputan.
    for col in ("edad", "horas_semanales", "gasto_mensual_usd"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "edad" in out.columns:
        out["edad"] = out["edad"].astype("float64")

    # ==========================================
    # LIMPIEZA — Duplicados
    # ==========================================
    # Última ocurrencia: prioriza ingestas recientes del mismo jugador (correcciones de soporte / re-registro).
    if "id_jugador" in out.columns:
        out = out.drop_duplicates(subset=["id_jugador"], keep="last")
    else:
        out = out.drop_duplicates()

    # ==========================================
    # LIMPIEZA — Imputación (nulos en métricas clave)
    # ==========================================
    def imputar_por_grupo(serie: pd.Series, grupo: pd.Series) -> pd.Series:
        filled = serie.copy()
        mediana_global = serie.median(skipna=True)
        for g in grupo.dropna().unique():
            mask = (grupo == g) & serie.isna()
            med_g = serie[grupo == g].median(skipna=True)
            fill_val = med_g if pd.notna(med_g) else mediana_global
            filled.loc[mask] = fill_val
        filled = filled.fillna(mediana_global)
        return filled

    if "plataforma" in out.columns:
        if "edad" in out.columns:
            # Mediana por plataforma: la edad típica en Steam no debe «arrastrar» la imputación en consola (cohortes distintas).
            out["edad"] = (
                imputar_por_grupo(out["edad"], out["plataforma"]).round().astype("Int64")
            )
        if "horas_semanales" in out.columns:
            # Misma lógica: horas medias difieren por dispositivo; evita sesgar riesgo de abandono entre ecosistemas.
            out["horas_semanales"] = imputar_por_grupo(out["horas_semanales"], out["plataforma"])
        if "gasto_mensual_usd" in out.columns:
            # Gasto imputado por plataforma refleja mejor el nivel de ARPU típico de ese canal que un relleno global.
            out["gasto_mensual_usd"] = imputar_por_grupo(
                out["gasto_mensual_usd"], out["plataforma"]
            )
    else:
        if "edad" in out.columns:
            out["edad"] = (
                out["edad"].fillna(out["edad"].median()).round().astype("Int64")
            )
        if "horas_semanales" in out.columns:
            out["horas_semanales"] = out["horas_semanales"].fillna(
                out["horas_semanales"].median()
            )
        if "gasto_mensual_usd" in out.columns:
            out["gasto_mensual_usd"] = out["gasto_mensual_usd"].fillna(
                out["gasto_mensual_usd"].median()
            )

    if "nombre" in out.columns:
        # Placeholder auditable: mejor fila conservada con etiqueta que exclusión que pierde gasto/horas reales.
        out["nombre"] = out["nombre"].astype("string").fillna("Sin nombre")

    if "genero_juego" in out.columns:
        # Bucket explícito: el análisis por género no debe mezclar NA con un género real en agregaciones.
        out["genero_juego"] = out["genero_juego"].fillna("desconocido")

    if "plataforma" in out.columns:
        moda_pl = out["plataforma"].mode(dropna=True)
        # Moda: asignar canal más frecuente en la muestra evita filas huérfanas en matrices plataforma×género.
        out["plataforma"] = out["plataforma"].fillna(
            moda_pl.iloc[0] if len(moda_pl) else "desconocida"
        )

    return out


def asignar_tipo_jugador(horas: pd.Series) -> pd.Series:
    """
    Propósito (PixelQuest): Etiqueta intensidad de uso para briefings de producto y priorización de contenido.

    Regla de negocio: Cortes fijos en horas/semana (Casual < 10, Frecuente 10–34, Hardcore ≥ 35) — comunicables a stakeholders y estables entre reportes.
    """
    bins = [-0.001, 10, 35, float("inf")]
    labels = ["Casual", "Frecuente", "Hardcore"]
    return pd.cut(horas, bins=bins, labels=labels, right=False).astype("string")


def asignar_riesgo_abandono(horas: pd.Series, gasto: pd.Series) -> pd.Series:
    """
    Propósito (PixelQuest): Prioriza intervenciones de retención y CRM según combinación engagement × gasto.

    Regla de negocio:
    - Alto: horas bajas Y gasto bajo (perfil frío).
    - Bajo: horas altas O gasto alto (anclaje por hábito o monetización).
    - Medio: banda intermedia. Si ambas reglas chocan tras imputación, prevalece Bajo (jugador ya «caliente» en algún eje).
    """
    resultado = pd.Series("Medio", index=horas.index, dtype="string")
    alto = (horas < UMBRAL_HORAS_BAJA) & (gasto < UMBRAL_GASTO_BAJO)
    bajo = (horas >= UMBRAL_HORAS_ALTA) | (gasto >= UMBRAL_GASTO_ALTO)
    resultado = resultado.mask(alto, "Alto")
    resultado = resultado.mask(bajo, "Bajo")
    resultado = resultado.mask(alto & bajo, "Bajo")
    return resultado


def main() -> None:
    # ==========================================
    # CARGA
    # ==========================================
    if not ARCHIVO_ENTRADA.is_file():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_ENTRADA}. Coloca datos_gamer.csv junto a este script."
        )

    df_raw = cargar_datos(ARCHIVO_ENTRADA)
    if RENOMBRAR_COLUMNAS:
        df_raw = df_raw.rename(columns=RENOMBRAR_COLUMNAS)

    print("=== Primeros registros (crudos) ===")
    print(df_raw.head(10).to_string())
    print()

    # ==========================================
    # LIMPIEZA
    # ==========================================
    df_limpio = preprocesar(df_raw)

    if "horas_semanales" not in df_limpio.columns or "gasto_mensual_usd" not in df_limpio.columns:
        raise ValueError(
            "El CSV debe incluir columnas 'horas_semanales' y 'gasto_mensual_usd' "
            "(o adapte el script a los nombres reales de su dataset)."
        )

    # ==========================================
    # VARIABLES DERIVADAS (reglas de negocio)
    # ==========================================
    df_limpio["tipo_jugador"] = asignar_tipo_jugador(df_limpio["horas_semanales"])
    df_limpio["riesgo_abandono"] = asignar_riesgo_abandono(
        df_limpio["horas_semanales"],
        df_limpio["gasto_mensual_usd"],
    )

    df_limpio.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8")

    print("=== Primeros registros (limpios + variables nuevas) ===")
    print(df_limpio.head(10).to_string())
    print()
    print(f"Archivo guardado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
