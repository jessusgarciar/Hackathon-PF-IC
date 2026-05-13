"""
PixelQuest Studios — Genera filas sintéticas aleatorias para `datos_gamer.csv`.

Uso típico (reemplazar el archivo usado por el ETL):
  python generar_datos_gamer_random.py -n 2000

Conservar el CSV actual y añadir más jugadores:
  python generar_datos_gamer_random.py -n 500 --anexar

Requisito: pip install pandas
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

NOMBRES = [
    "Ana García",
    "Carlos Ruiz",
    "María López",
    "Luis Fernández",
    "Pedro Sánchez",
    "Elena Martín",
    "Sofía Herrera",
    "Jorge Díaz",
    "Natalia Vargas",
    "Andrés Molina",
    "Lucía Ramos",
    "Miguel Torres",
    "Carmen Vega",
    "David Ortega",
    "Laura Gil",
]

# Variantes como en datos reales para probar normalización del ETL (`procesar_datos_gamer.py`).
PLATAFORMAS_VARIANTES = [
    "PC ",
    "pc",
    "steam",
    "Steam",
    "playstation",
    "PS5",
    "Play Station 5",
    "XBOX",
    "xbox",
    "nintendo switch",
    "Switch",
    "mobile",
    "movil",
]

GENEROS_VARIANTES = [
    "Battle Royale",
    "RPG",
    "rpg",
    "AVENTURA",
    "MOBA",
    "moba",
    "Shooter",
    "shootER",
    "sports",
    "Sports",
    "puzzle",
    "puZZle",
]

PAISES_PESO = [
    ("México", 0.25),
    ("España", 0.15),
    ("Argentina", 0.12),
    ("Colombia", 0.10),
    ("Chile", 0.08),
    ("Perú", 0.07),
    ("Estados Unidos", 0.06),
    ("Brasil", 0.05),
    ("Ecuador", 0.04),
    ("Uruguay", 0.03),
    ("Costa Rica", 0.02),
    ("Guatemala", 0.015),
    ("Panamá", 0.01),
    ("Bolivia", 0.005),
]

JUEGOS = [
    ("G001", "BattleZone"),
    ("G002", "ShadowQuest"),
    ("G003", "DragonRealm"),
    ("G004", "PixelRoyale"),
    ("G005", "CyberRacer"),
    ("G006", "MysticLegends"),
    ("G007", "StarFleet"),
    ("G008", "ThunderArena"),
]

PAISES = [p for p, _ in PAISES_PESO]
PESOS_PAISES = [w for _, w in PAISES_PESO]

TASA_CAMBIO_MXN = 17.5

FECHA_INICIO = date(2024, 1, 1)
FECHA_FIN = date(2026, 4, 30)
FECHA_MAX_CONEXION = date(2026, 4, 30)


def _siguiente_id_base(df_exist: pd.DataFrame | None) -> int:
    if df_exist is None or df_exist.empty or "id_jugador" not in df_exist.columns:
        return 1
    s = pd.to_numeric(df_exist["id_jugador"], errors="coerce").dropna()
    if s.empty:
        return 1
    return int(s.max()) + 1


def _fecha_aleatoria(rng: random.Random, inicio: date, fin: date) -> date:
    delta = (fin - inicio).days
    return inicio + timedelta(days=rng.randint(0, delta))


def generar_filas(n: int, rng: random.Random, id_inicio: int) -> list[dict[str, object]]:
    filas: list[dict[str, object]] = []
    for i in range(n):
        jid = id_inicio + i
        # ~6% sin nombre para probar imputación
        nombre = rng.choice(NOMBRES) if rng.random() > 0.06 else None

        # ~4% sin edad
        if rng.random() < 0.04:
            edad = pd.NA
        else:
            edad = int(rng.triangular(14, 55, 24))

        # Horas/ingreso correlacionados débilmente
        h = rng.betavariate(1.8, 3.5) * 140
        if rng.random() < 0.02:
            h = float("nan")
        ingreso_base = rng.betavariate(1.2, 4.0) * 280
        if h > 35:
            ingreso_base *= rng.uniform(1.0, 2.8)
        g = max(0.0, ingreso_base * rng.gauss(1.0, 0.25))
        if rng.random() < 0.025:
            g = float("nan")
        elif rng.random() < 0.15:
            g = round(g, 2)
        else:
            g = round(g, 2)
        g_mxn = g * TASA_CAMBIO_MXN if not (isinstance(g, float) and pd.isna(g)) else g
        g = round(g_mxn, 2) if not (isinstance(g, float) and pd.isna(g)) else g

        pl = rng.choice(PLATAFORMAS_VARIANTES)
        gen = rng.choice(GENEROS_VARIANTES)

        # ---------- NUEVAS VARIABLES ----------

        # país (con pesos)
        pais = rng.choices(PAISES, weights=PESOS_PAISES, k=1)[0]

        # juego_id y nombre_juego
        juego_id, nombre_juego = rng.choice(JUEGOS)

        # compras_internas — correlacionado con ingreso; si ingreso > 0, alta probabilidad de Sí
        g_val = g if not (isinstance(g, float) and pd.isna(g)) else 0
        if g_val > 0:
            compras_internas = "Sí" if rng.random() < 0.82 else "No"
        else:
            compras_internas = "Sí" if rng.random() < 0.08 else "No"

        # nivel_alcanzado — correlacionado con horas
        nivel_base = 10 + (h if pd.notna(h) else rng.betavariate(1.8, 3.5) * 140) * 0.7
        nivel_alcanzado = int(max(1, min(100, round(rng.gauss(nivel_base, 18)))))
        if rng.random() < 0.03:
            nivel_alcanzado = pd.NA

        # calificacion — correlacionada con horas
        cal_base = 2.0 + (h if pd.notna(h) else rng.betavariate(1.8, 3.5) * 140) * 0.025
        calificacion = round(max(1.0, min(5.0, rng.gauss(cal_base, 0.7))), 1)
        if rng.random() < 0.03:
            calificacion = pd.NA

        # fecha_registro — entre 2024-01-01 y 2026-04-01
        fecha_registro = _fecha_aleatoria(rng, FECHA_INICIO, date(2026, 4, 1))
        if rng.random() < 0.02:
            fecha_registro = pd.NA

        # fecha_ultima_conexion — entre fecha_registro+1 y 2026-04-30
        if isinstance(fecha_registro, date):
            dias_max = (FECHA_MAX_CONEXION - fecha_registro).days
            if dias_max < 1:
                fecha_ultima_conexion = fecha_registro
            else:
                offset = int(rng.triangular(1, dias_max + 1, dias_max * 0.6))
                fecha_ultima_conexion = fecha_registro + timedelta(days=offset)
        else:
            fecha_ultima_conexion = _fecha_aleatoria(rng, FECHA_INICIO, FECHA_MAX_CONEXION)
        if rng.random() < 0.02:
            fecha_ultima_conexion = pd.NA

        # estado_jugador — basado en fecha_ultima_conexion
        if isinstance(fecha_ultima_conexion, date):
            dias_inactivo = (FECHA_MAX_CONEXION - fecha_ultima_conexion).days
            if dias_inactivo <= 30:
                estado_jugador = "Activo"
            elif dias_inactivo <= 90:
                estado_jugador = "Activo" if rng.random() < 0.4 else "Inactivo"
            else:
                estado_jugador = "Inactivo" if rng.random() < 0.85 else "Activo"
        else:
            estado_jugador = "Activo" if rng.random() < 0.65 else "Inactivo"
        if rng.random() < 0.025:
            estado_jugador = pd.NA

        filas.append(
            {
                "id_jugador": jid,
                "nombre": nombre,
                "edad": edad,
                "horas_semanales": round(h, 2) if pd.notna(h) else float("nan"),
                "ingreso_mensual_mxn": round(g, 2) if pd.notna(g) else float("nan"),
                "plataforma": pl,
                "genero_juego": gen,
                "pais": pais,
                "juego_id": juego_id,
                "nombre_juego": nombre_juego,
                "compras_internas": compras_internas,
                "nivel_alcanzado": nivel_alcanzado,
                "calificacion": calificacion,
                "fecha_registro": (
                    fecha_registro.strftime("%d/%m/%Y")
                    if isinstance(fecha_registro, date)
                    else pd.NA
                ),
                "fecha_ultima_conexion": (
                    fecha_ultima_conexion.strftime("%d/%m/%Y")
                    if isinstance(fecha_ultima_conexion, date)
                    else pd.NA
                ),
                "estado_jugador": estado_jugador,
            }
        )

    # ~2% duplicados adyacentes (mismo jugador ingestado dos veces)
    dup_slots = max(1, int(n * 0.02))
    for _ in range(dup_slots):
        idx = rng.randrange(max(1, len(filas) - 3))
        filas.insert(idx + 1, filas[idx].copy())
        filas[idx + 1]["id_jugador"] = filas[idx]["id_jugador"]
        if rng.random() < 0.5:
            filas[idx + 1]["nombre"] = filas[idx]["nombre"]
            filas[idx + 1]["edad"] = filas[idx]["edad"]

    return filas


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Genera filas aleatorias para datos_gamer.csv (compatibles con procesar_datos_gamer.py)."
    )
    ap.add_argument(
        "-n", "--filas", type=int, default=300, help="Cantidad de filas nuevas a generar (default: 300)."
    )
    ap.add_argument(
        "-o",
        "--salida",
        type=Path,
        default=BASE_DIR / "datos_gamer.csv",
        help="Ruta del CSV destino (default: datos_gamer.csv junto al script).",
    )
    ap.add_argument(
        "--anexar",
        action="store_true",
        help="Si existe el archivo destino, conserva las filas y concatena tras la última id_jugador.",
    )
    ap.add_argument("--semilla", type=int, default=None, help="Semilla RNG para reproducir el mismo conjunto.")

    args = ap.parse_args()
    if args.filas < 1:
        raise SystemExit("--filas debe ser >= 1")

    rng = random.Random(args.semilla)

    existente: pd.DataFrame | None = None
    if args.anexar and args.salida.is_file():
        existente = pd.read_csv(args.salida, encoding="utf-8")

    base_id = _siguiente_id_base(existente)
    nueva = pd.DataFrame(generar_filas(args.filas, rng, base_id))

    columnas = [
        "id_jugador",
        "nombre",
        "edad",
        "horas_semanales",
        "ingreso_mensual_mxn",
        "plataforma",
        "genero_juego",
        "pais",
        "juego_id",
        "nombre_juego",
        "compras_internas",
        "nivel_alcanzado",
        "calificacion",
        "fecha_registro",
        "fecha_ultima_conexion",
        "estado_jugador",
    ]
    nueva = nueva[columnas]

    if existente is not None:
        for c in columnas:
            if c not in existente.columns:
                existente[c] = pd.NA
        out = pd.concat([existente[columnas], nueva], ignore_index=True)
    else:
        out = nueva

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.salida, index=False, encoding="utf-8")

    modo = "anexado y guardado en" if existente is not None else "guardado en"
    print(f"OK: {len(nueva)} filas nuevas ({modo}) {args.salida.resolve()}")
    print(f"Total filas en archivo: {len(out)}")
    print("Siguiente paso: python procesar_datos_gamer.py")


if __name__ == "__main__":
    main()
