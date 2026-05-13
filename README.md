# PixelQuest Studios — Inteligencia de datos gamer

Proyecto de **hackathon / proyecto final** orientado a **PixelQuest Studios**: pipeline de datos sobre jugadores, análisis de negocio y un **panel interactivo en Streamlit** con KPIs, consultas tipo OLAP y visualizaciones (Plotly).

## Componentes principales

| Archivo | Rol |
|--------|-----|
| `procesar_datos_gamer.py` | ETL: lee `datos_gamer.csv`, limpia y enriquece → escribe `datos_gamer_limpios.csv`. |
| `analisis_inteligencia_gamer.py` | Métricas de valor, consultas Q1–Q5, pivot/export OLAP y gráficos de salida en `salida_analisis_gamer/`. |
| `app_pixelquest.py` | Aplicación Streamlit que reutiliza el núcleo analítico para exploración y BI en el navegador. |

Los CSV incluidos (`datos_gamer.csv`, `datos_gamer_limpios.csv`, `olap_plataforma_genero.csv`) sirven como datos de trabajo y referencia para el cubo/reportes.

## Requisitos

- **Python 3.10+** (probado con 3.13 en desarrollo).
- **pip** para dependencias.

## Instalación

1. **Clonar el repositorio** y entrar al directorio del proyecto.

2. **Crear y activar un entorno virtual** (recomendado):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   En macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instalar dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Credenciales de Streamlit (opcional)**  
   Para evitar prompts de correo en local, copia el ejemplo:

   ```powershell
   Copy-Item .streamlit\credentials.toml.example .streamlit\credentials.toml
   ```

   En Unix:

   ```bash
   cp .streamlit/credentials.toml.example .streamlit/credentials.toml
   ```

   El archivo real `.streamlit/credentials.toml` está en `.gitignore` y no debe subirse al remoto.

## Uso

### 1. Regenerar datos limpios

Si modificas `datos_gamer.csv`:

```bash
python procesar_datos_gamer.py
```

### 2. Análisis por línea de comandos

Genera métricas, exportaciones y figuras en `salida_analisis_gamer/`:

```bash
python analisis_inteligencia_gamer.py
```

### 3. Panel Streamlit

Desde PowerShell en Windows puedes usar el script incluido:

```powershell
.\run_pixelquest.ps1
```

O directamente:

```bash
streamlit run app_pixelquest.py
```

La app suele abrirse en `http://localhost:8501` (puerto configurado en `.streamlit/config.toml`).

## Estructura útil

- `.streamlit/config.toml` — opciones del servidor Streamlit (puerto, uso sin navegador headless).
- `.gitignore` — excluye `__pycache__/`, entornos virtuales, salidas de gráficos y credenciales locales.
