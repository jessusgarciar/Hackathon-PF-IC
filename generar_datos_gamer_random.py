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


def _siguiente_id_base(df_exist: pd.DataFrame | None) -> int:
    if df_exist is None or df_exist.empty or "id_jugador" not in df_exist.columns:
        return 1
    s = pd.to_numeric(df_exist["id_jugador"], errors="coerce").dropna()
    if s.empty:
        return 1
    return int(s.max()) + 1


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

        # Horas/gasto correlacionados débilmente (hardcores un poco más gastadores)
        h = rng.betavariate(1.8, 3.5) * 140
        if rng.random() < 0.02:
            h = float("nan")
        gasto_base = rng.betavariate(1.2, 4.0) * 280
        if h > 35:
            gasto_base *= rng.uniform(1.0, 2.8)
        g = max(0.0, gasto_base * rng.gauss(1.0, 0.25))
        if rng.random() < 0.025:
            g = float("nan")
        elif rng.random() < 0.15:
            g = round(g, 2)
        else:
            g = round(g, 2)

        pl = rng.choice(PLATAFORMAS_VARIANTES)
        gen = rng.choice(GENEROS_VARIANTES)

        filas.append(
            {
                "id_jugador": jid,
                "nombre": nombre,
                "edad": edad,
                "horas_semanales": round(h, 2) if pd.notna(h) else float("nan"),
                "gasto_mensual_usd": round(g, 2) if pd.notna(g) else float("nan"),
                "plataforma": pl,
                "genero_juego": gen,
            }
        )
    # ~2% duplicados adyacentes (mismo jugador ingestado dos veces) para probar drop_duplicates
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
    ap.add_argument("-n", "--filas", type=int, default=300, help="Cantidad de filas nuevas a generar (default: 300).")
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

    # Mismo orden de columnas que datos_gamer.csv de referencia
    columnas = [
        "id_jugador",
        "nombre",
        "edad",
        "horas_semanales",
        "gasto_mensual_usd",
        "plataforma",
        "genero_juego",
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
