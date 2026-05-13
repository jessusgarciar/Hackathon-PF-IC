"""
PixelQuest Studios — Pipeline de limpieza y enriquecimiento de datos de jugadores.

Entrada:  datos_gamer.csv (mismo directorio que este script)
Salida:  datos_gamer_limpios.csv

Requisito: pip install pandas
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas (relativas al archivo del script para que sea portable)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = BASE_DIR / "datos_gamer.csv"
ARCHIVO_SALIDA = BASE_DIR / "datos_gamer_limpios.csv"

# Si tu CSV usa otros nombres de columnas, complétalo aquí (clave = nombre en tu archivo).
RENOMBRAR_COLUMNAS: dict[str, str] = {
    # "horas_jugadas": "horas_semanales",
    # "gasto_total": "gasto_mensual_usd",
}

# Umbrales para variables derivadas (ajustar según negocio / análisis exploratorio)
# tipo_jugador: ver bins en asignar_tipo_jugador (Casual / Frecuente / Hardcore)
# riesgo_abandono: umbrales de horas y gasto en USD/mes (tres bandas no vacías)
UMBRAL_HORAS_BAJA = 10       # por debajo: jugador poco activo si no compensa con gasto
UMBRAL_GASTO_BAJO = 10.0
UMBRAL_HORAS_ALTA = 25       # por encima: alta retención por engagement
UMBRAL_GASTO_ALTO = 40.0     # por encima: alto compromiso monetario (whale / fan)


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Carga el CSV usando UTF-8 para nombres y textos con tildes."""
    return pd.read_csv(ruta, encoding="utf-8")


def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesamiento — al menos 3 técnicas documentadas:

    1) Manejo de valores nulos
       - Decisión: para columnas numéricas clave (edad, horas_semanales, gasto_mensual_usd)
         usamos la mediana imputada por grupo de plataforma cuando exista;
         si el grupo no tiene valores, caemos a la mediana global. Así reducimos sesgo
         frente a rellenar siempre con 0 (que distorsionaría gasto/horas) o borrar filas
         con muchos nulos (perderíamos jugadores reales con un solo campo faltante).
       - Para 'nombre' vacío tras deduplicar, rellenamos con 'Sin nombre' para trazabilidad
         en reportes sin eliminar el registro si aporta métricas de comportamiento.

    2) Eliminación (deduplicación) de registros repetidos
       - Decisión: eliminar duplicados exactos y, si hay id_jugador, conservar la última
         ocurrencia (supone que la entrada más reciente corrige datos). Si no hubiera id,
         se podría deduplicar por (nombre, edad, plataforma) según política de datos.

    3) Normalización de textos inconsistentes (plataforma, género)
       - Decisión: strip, minúsculas, colapso de espacios y mapeo de sinónimos
         (ej. PS5 / playstation / Play Station 5 → playstation_5) para analítica homogénea
         y cruces correctos en tablas/agrupaciones.

    4) Conversión y saneo de tipos (formato)
       - Decisión: coercear a numérico con errors='coerce' y revisar; edad/horas/gasto
         deben ser numéricos para reglas de negocio (umbrales, scores).
    """
    out = df.copy()

    # --- (3) Normalización temprana de texto para que agrupaciones de imputación sean útiles ---
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

    # --- (4) Tipos numéricos ---
    # edad se trata como float durante imputación; al final se redondea a entero nullable
    for col in ("edad", "horas_semanales", "gasto_mensual_usd"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "edad" in out.columns:
        out["edad"] = out["edad"].astype("float64")

    # --- (2) Duplicados ---
    if "id_jugador" in out.columns:
        out = out.drop_duplicates(subset=["id_jugador"], keep="last")
    else:
        out = out.drop_duplicates()

    # --- (1) Nulos en numéricos: imputación por mediana de plataforma ---
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
            out["edad"] = (
                imputar_por_grupo(out["edad"], out["plataforma"]).round().astype("Int64")
            )
        if "horas_semanales" in out.columns:
            out["horas_semanales"] = imputar_por_grupo(out["horas_semanales"], out["plataforma"])
        if "gasto_mensual_usd" in out.columns:
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
        out["nombre"] = out["nombre"].astype("string").fillna("Sin nombre")

    if "genero_juego" in out.columns:
        out["genero_juego"] = out["genero_juego"].fillna("desconocido")

    if "plataforma" in out.columns:
        moda_pl = out["plataforma"].mode(dropna=True)
        out["plataforma"] = out["plataforma"].fillna(
            moda_pl.iloc[0] if len(moda_pl) else "desconocida"
        )

    return out


def asignar_tipo_jugador(horas: pd.Series) -> pd.Series:
    """
    Segmenta por intensidad de juego semanal.
    Umbrales fijos facilitan comunicación con negocio; calibrar con distribución real.
    """
    bins = [-0.001, 10, 35, float("inf")]
    labels = ["Casual", "Frecuente", "Hardcore"]
    return pd.cut(horas, bins=bins, labels=labels, right=False).astype("string")


def asignar_riesgo_abandono(horas: pd.Series, gasto: pd.Series) -> pd.Series:
    """
    Combinación horas + gasto (hipótesis operativa):

    - Alto riesgo: poca actividad Y bajo gasto → perfil frío, más propenso a dejar de jugar.
    - Bajo riesgo: muchas horas O gasto alto → anclado por engagement o monetización.
    - Medio: compromiso intermedio (no es «frío» ni claramente «anclado»).

    Separar umbrales «bajos» y «altos» evita que la región intermedia quede vacía.
    """
    resultado = pd.Series("Medio", index=horas.index, dtype="string")
    alto = (horas < UMBRAL_HORAS_BAJA) & (gasto < UMBRAL_GASTO_BAJO)
    bajo = (horas >= UMBRAL_HORAS_ALTA) | (gasto >= UMBRAL_GASTO_ALTO)
    resultado = resultado.mask(alto, "Alto")
    resultado = resultado.mask(bajo, "Bajo")
    # Si una fila cumple ambos (datos extremos tras imputaciones raras), prioriza riesgo bajo
    resultado = resultado.mask(alto & bajo, "Bajo")
    return resultado


def main() -> None:
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

    df_limpio = preprocesar(df_raw)

    if "horas_semanales" not in df_limpio.columns or "gasto_mensual_usd" not in df_limpio.columns:
        raise ValueError(
            "El CSV debe incluir columnas 'horas_semanales' y 'gasto_mensual_usd' "
            "(o adapte el script a los nombres reales de su dataset)."
        )

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
