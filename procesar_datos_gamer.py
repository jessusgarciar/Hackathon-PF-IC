"""
PixelQuest Studios — Pipeline de limpieza y enriquecimiento de datos de jugadores.

Entrada:  datos_gamer.csv (mismo directorio que este script)
Salida:  datos_gamer_limpios.csv

Requisito: pip install pandas
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

# ==========================================
# CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = BASE_DIR / "datos_gamer.csv"
ARCHIVO_SALIDA = BASE_DIR / "datos_gamer_limpios.csv"

# Fecha de referencia para antigüedad y días de inactividad
FECHA_REFERENCIA = date(2026, 4, 30)

RENOMBRAR_COLUMNAS: dict[str, str] = {}

UMBRAL_HORAS_BAJA = 10
UMBRAL_INGRESO_BAJO = 175.0
UMBRAL_HORAS_ALTA = 25
UMBRAL_INGRESO_ALTO = 700.0

# Mapa juego_id -> nombre_juego canónico
MAPA_JUEGOS: dict[str, str] = {
    "g001": "BattleZone",
    "g002": "ShadowQuest",
    "g003": "DragonRealm",
    "g004": "PixelRoyale",
    "g005": "CyberRacer",
    "g006": "MysticLegends",
    "g007": "StarFleet",
    "g008": "ThunderArena",
}

MAPA_GENERO_POR_JUEGO: dict[str, str] = {
    "BattleZone": "battle_royale",
    "ShadowQuest": "rpg",
    "DragonRealm": "rpg",
    "PixelRoyale": "battle_royale",
    "CyberRacer": "deportes",
    "MysticLegends": "aventura",
    "StarFleet": "shooter",
    "ThunderArena": "moba",
}


def cargar_datos(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, encoding="utf-8")


def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ==========================================
    # LIMPIEZA — Normalización léxica (plataforma, género, país, juego)
    # ==========================================
    for col in ("plataforma", "genero_juego", "pais", "nombre_juego"):
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

    mapa_paises = {
        "méxico": "México",
        "mexico": "México",
        "españa": "España",
        "espana": "España",
        "spain": "España",
        "argentina": "Argentina",
        "colombia": "Colombia",
        "chile": "Chile",
        "perú": "Perú",
        "peru": "Perú",
        "estados unidos": "Estados Unidos",
        "usa": "Estados Unidos",
        "estados unidos de américa": "Estados Unidos",
        "ee.uu.": "Estados Unidos",
        "eeuu": "Estados Unidos",
        "brasil": "Brasil",
        "brazil": "Brasil",
        "ecuador": "Ecuador",
        "uruguay": "Uruguay",
        "costa rica": "Costa Rica",
        "guatemala": "Guatemala",
        "panamá": "Panamá",
        "panama": "Panamá",
        "bolivia": "Bolivia",
    }
    if "pais" in out.columns:
        out["pais"] = out["pais"].replace(mapa_paises)

    # Normalizar juego_id
    if "juego_id" in out.columns:
        out["juego_id"] = (
            out["juego_id"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    # Normalizar compras_internas
    if "compras_internas" in out.columns:
        out["compras_internas"] = (
            out["compras_internas"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"sí": "Sí", "si": "Sí", "yes": "Sí", "no": "No", "1": "Sí", "0": "No"})
        )

    # Normalizar estado_jugador
    if "estado_jugador" in out.columns:
        out["estado_jugador"] = (
            out["estado_jugador"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"activo": "Activo", "inactivo": "Inactivo"})
        )

    # ==========================================
    # LIMPIEZA — Tipos numéricos
    # ==========================================
    for col in ("edad", "horas_semanales", "ingreso_mensual_mxn", "nivel_alcanzado", "calificacion"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "edad" in out.columns:
        out["edad"] = out["edad"].astype("float64")
    if "calificacion" in out.columns:
        out["calificacion"] = out["calificacion"].clip(1.0, 5.0)

    # ==========================================
    # LIMPIEZA — Fechas (parseo inicial antes de duplicados)
    # ==========================================
    if "fecha_registro" in out.columns:
        out["fecha_registro_raw"] = pd.to_datetime(
            out["fecha_registro"].astype("string"), format="%d/%m/%Y", errors="coerce", dayfirst=True
        )
    if "fecha_ultima_conexion" in out.columns:
        out["fecha_ultima_conexion_raw"] = pd.to_datetime(
            out["fecha_ultima_conexion"].astype("string"), format="%d/%m/%Y", errors="coerce", dayfirst=True
        )

    # ==========================================
    # LIMPIEZA — Duplicados
    # ==========================================
    if "id_jugador" in out.columns:
        out = out.drop_duplicates(subset=["id_jugador"], keep="last")
    else:
        out = out.drop_duplicates()

    # ==========================================
    # LIMPIEZA — Imputación
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
        grp = out["plataforma"]
    else:
        grp = pd.Series("global", index=out.index)

    if "edad" in out.columns:
        out["edad"] = imputar_por_grupo(out["edad"], grp).round().astype("Int64")
    if "horas_semanales" in out.columns:
        out["horas_semanales"] = imputar_por_grupo(out["horas_semanales"], grp)
    if "ingreso_mensual_mxn" in out.columns:
        out["ingreso_mensual_mxn"] = imputar_por_grupo(out["ingreso_mensual_mxn"], grp)
    if "nivel_alcanzado" in out.columns:
        out["nivel_alcanzado"] = (
            imputar_por_grupo(out["nivel_alcanzado"], grp).round().astype("Int64")
        )
    if "calificacion" in out.columns:
        out["calificacion"] = imputar_por_grupo(out["calificacion"], grp).round(1)

    # Imputar texto
    if "nombre" in out.columns:
        out["nombre"] = out["nombre"].astype("string").fillna("Sin nombre")
    if "pais" in out.columns:
        moda_p = out["pais"].mode(dropna=True)
        out["pais"] = out["pais"].fillna(moda_p.iloc[0] if len(moda_p) else "México")
    if "plataforma" in out.columns:
        moda_pl = out["plataforma"].mode(dropna=True)
        out["plataforma"] = out["plataforma"].fillna(
            moda_pl.iloc[0] if len(moda_pl) else "desconocida"
        )
    if "genero_juego" in out.columns:
        out["genero_juego"] = out["genero_juego"].fillna("desconocido")

    # juego_id y nombre_juego — consistencia mutua
    if "juego_id" in out.columns:
        moda_jid = out["juego_id"].mode(dropna=True)
        out["juego_id"] = out["juego_id"].fillna(
            moda_jid.iloc[0] if len(moda_jid) else "g001"
        )
        # Normalizar formato: G001
        out["juego_id"] = out["juego_id"].str.replace(r"^g0*", "G", regex=True).str.upper()
        # Asegurar que empiecen con G y tengan 3 dígitos
        def _fmt_gid(v: str) -> str:
            if pd.isna(v):
                return v
            v = str(v).strip().upper()
            if not v.startswith("G"):
                v = "G" + v
            if len(v) < 4:
                v = v[0] + v[1:].zfill(3)
            return v
        out["juego_id"] = out["juego_id"].apply(_fmt_gid)

    if "nombre_juego" in out.columns:
        # Reconstruir desde juego_id si falta
        if "juego_id" in out.columns:
            for gid, gname in MAPA_JUEGOS.items():
                mask = (out["nombre_juego"].isna()) & (out["juego_id"].str.lower() == gid)
                out.loc[mask, "nombre_juego"] = gname
        out["nombre_juego"] = out["nombre_juego"].astype("string").str.title()
        out["nombre_juego"] = out["nombre_juego"].fillna("BattleZone")

    # compras_internas — imputar desde ingreso
    if "compras_internas" in out.columns:
        mask_na = out["compras_internas"].isna()
        mask_ingreso_pos = out["ingreso_mensual_mxn"] > 0
        out.loc[mask_na & mask_ingreso_pos, "compras_internas"] = "Sí"
        out.loc[mask_na & ~mask_ingreso_pos, "compras_internas"] = "No"
        # Valores residuales no reconocidos
        out["compras_internas"] = out["compras_internas"].where(
            out["compras_internas"].isin(["Sí", "No"]), "No"
        )

    # ==========================================
    # FECHAS — Imputación y derivadas (usa columnas _raw ya parseadas)
    # ==========================================
    ref_ts = pd.Timestamp(FECHA_REFERENCIA)

    # fecha_registro: si NA, estimar desde fecha_ultima_conexion_raw - 180 días
    if "fecha_registro_raw" in out.columns and "fecha_ultima_conexion_raw" in out.columns:
        mask_reg_na = out["fecha_registro_raw"].isna()
        mask_ult_ok = ~out["fecha_ultima_conexion_raw"].isna()
        out.loc[mask_reg_na & mask_ult_ok, "fecha_registro_raw"] = (
            out.loc[mask_reg_na & mask_ult_ok, "fecha_ultima_conexion_raw"]
            - pd.Timedelta(days=180)
        )
        out.loc[mask_reg_na & ~mask_ult_ok, "fecha_registro_raw"] = pd.Timestamp("2025-06-15")

    # fecha_ultima_conexion: si NA, estimar desde fecha_registro_raw + 30 días
    if "fecha_ultima_conexion_raw" in out.columns and "fecha_registro_raw" in out.columns:
        mask_ult_na = out["fecha_ultima_conexion_raw"].isna()
        mask_reg_ok = ~out["fecha_registro_raw"].isna()
        max_delta = (ref_ts - out.loc[mask_ult_na & mask_reg_ok, "fecha_registro_raw"]).dt.days
        clamped = max_delta.clip(upper=30)
        out.loc[mask_ult_na & mask_reg_ok, "fecha_ultima_conexion_raw"] = (
            out.loc[mask_ult_na & mask_reg_ok, "fecha_registro_raw"]
            + pd.to_timedelta(clamped, unit="D")
        )
        out.loc[mask_ult_na & ~mask_reg_ok, "fecha_ultima_conexion_raw"] = pd.Timestamp("2026-03-01")

    # Derivar antiguedad_dias y dias_ultima_conexion
    if "fecha_registro_raw" in out.columns:
        out["antiguedad_dias"] = (ref_ts - out["fecha_registro_raw"]).dt.days

    if "fecha_ultima_conexion_raw" in out.columns:
        out["dias_ultima_conexion"] = (ref_ts - out["fecha_ultima_conexion_raw"]).dt.days

    # estado_jugador — derivar desde dias_ultima_conexion si NA
    if "estado_jugador" in out.columns and "dias_ultima_conexion" in out.columns:
        mask_estado_na = out["estado_jugador"].isna()
        out.loc[mask_estado_na, "estado_jugador"] = out.loc[
            mask_estado_na, "dias_ultima_conexion"
        ].apply(lambda d: "Inactivo" if pd.notna(d) and d > 30 else "Activo")
        out["estado_jugador"] = out["estado_jugador"].fillna("Activo")

    # Convertir fechas a string para export CSV y limpiar columnas temporales
    if "fecha_registro_raw" in out.columns:
        out["fecha_registro"] = (
            out["fecha_registro_raw"].dt.strftime("%d/%m/%Y").fillna(pd.NA)
        )
    if "fecha_ultima_conexion_raw" in out.columns:
        out["fecha_ultima_conexion"] = (
            out["fecha_ultima_conexion_raw"].dt.strftime("%d/%m/%Y").fillna(pd.NA)
        )

    for col_tmp in ["fecha_registro_raw", "fecha_ultima_conexion_raw"]:
        if col_tmp in out.columns:
            out = out.drop(columns=[col_tmp])

    return out


def asignar_tipo_jugador(horas: pd.Series) -> pd.Series:
    bins = [-0.001, 10, 35, float("inf")]
    labels = ["Casual", "Frecuente", "Hardcore"]
    return pd.cut(horas, bins=bins, labels=labels, right=False).astype("string")


def asignar_riesgo_abandono(horas: pd.Series, ingreso: pd.Series) -> pd.Series:
    """
    Propósito (PixelQuest): Prioriza intervenciones de retención y CRM según combinación engagement × ingreso.

    Regla de negocio:
    - Alto: horas bajas Y ingreso bajo (perfil frío).
    - Bajo: horas altas O ingreso alto (anclaje por hábito o monetización).
    - Medio: banda intermedia. Si ambas reglas chocan tras imputación, prevalece Bajo (jugador ya «caliente» en algún eje).
    """
    resultado = pd.Series("Medio", index=horas.index, dtype="string")
    alto = (horas < UMBRAL_HORAS_BAJA) & (ingreso < UMBRAL_INGRESO_BAJO)
    bajo = (horas >= UMBRAL_HORAS_ALTA) | (ingreso >= UMBRAL_INGRESO_ALTO)
    resultado = resultado.mask(alto, "Alto")
    resultado = resultado.mask(bajo, "Bajo")
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

    if "horas_semanales" not in df_limpio.columns or "ingreso_mensual_mxn" not in df_limpio.columns:
        raise ValueError(
            "El CSV debe incluir columnas 'horas_semanales' y 'ingreso_mensual_mxn' "
            "(o adapte el script a los nombres reales de su dataset)."
        )

    # Variables derivadas (reglas de negocio)
    df_limpio["tipo_jugador"] = asignar_tipo_jugador(df_limpio["horas_semanales"])
    df_limpio["riesgo_abandono"] = asignar_riesgo_abandono(
        df_limpio["horas_semanales"],
        df_limpio["ingreso_mensual_mxn"],
    )

    df_limpio.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8")

    print("=== Primeros registros (limpios + variables nuevas) ===")
    print(df_limpio.head(10).to_string())
    print()
    print(f"Archivo guardado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
