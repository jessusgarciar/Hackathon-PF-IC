# Uso

Reemplazar datos_gamer.csv por un conjunto sintético grande:

```powershell
python generar_datos_gamer_random.py -n 2500 --semilla 42
python procesar_datos_gamer.py
```

Mantener lo que ya tienes y solo añadir filas (IDs después del máximo existente):

```powershell
python generar_datos_gamer_random.py -n 800 --anexar
python procesar_datos_gamer.py
```
