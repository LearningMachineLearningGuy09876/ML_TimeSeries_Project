# Pronóstico de Series Temporales — Ventas con ARIMA

> Pipeline de análisis de series temporales de extremo a extremo sobre un dataset de ventas: prueba de estacionariedad con el test Dickey-Fuller Aumentado, descomposición estacional, análisis de autocorrelación y un pronóstico de 60 pasos con `auto_arima` — cubriendo el flujo diagnóstico completo que precede a cualquier modelo de series temporales.

---

## Problema

Pronosticar valores futuros de ventas a partir de una serie temporal histórica. Los datos de series temporales rompen un supuesto fundamental de la mayoría de los modelos de ML — que las observaciones son independientes. Las ventas en el tiempo *t* están correlacionadas con las ventas en *t-1*, *t-2*, etc. ARIMA modela esta autocorrelación explícitamente y la usa para hacer pronósticos.

## Dataset

- **Fuente:** Dataset de ventas (4Geeks / breathecode)
- **Estructura:** Serie temporal de cifras de ventas indexada por fecha
- **Variable objetivo:** `sales` — una única serie univariada establecida como índice del DataFrame tras el parseo con `pd.to_datetime()`

## Pipeline de Análisis

| Paso | Herramienta | Hallazgo |
|---|---|---|
| Inspección visual | `sns.lineplot` | Tendencia alcista clara; la serie no revierte a una media |
| Descomposición estacional | `seasonal_decompose` | Tendencia confirmada; **no se encontró componente estacional significativo** |
| Prueba de estacionariedad | Test ADF (`adfuller`) | p-valor = **0,98** — no se rechaza H₀ → serie **no estacionaria** |
| Autocorrelación | `plot_acf` | Alta autocorrelación en todo el rango, disminuyendo lentamente con el retardo |
| Selección de modelo | `auto_arima` (pmdarima) | Búsqueda automática en el espacio (p, d, q); seasonal=False, m=7 |
| Pronóstico | `model.predict(60)` | Pronóstico de 60 pasos graficado frente a los datos históricos |

## Diagnósticos Clave

**Test ADF (Dickey-Fuller Aumentado):**
- H₀: la serie tiene una raíz unitaria (es no estacionaria)
- p-valor = 0,98 >> 0,05 → no se rechaza H₀ → **no estacionariedad confirmada**
- Esto significa que la serie no tiene una media estable a la que revertir — debe diferenciarse antes de ajustar un ARIMA estándar

**Descomposición estacional:**
- Componente de tendencia: pendiente alcista fuerte y persistente
- Componente estacional: plano — no hay patrón estacional repetitivo en este dataset
- Componente residual: ruido aleatorio restante tras eliminar la tendencia

**Gráfico ACF:**
- Alta autocorrelación positiva en todos los retardos, disminuyendo gradualmente
- Característico de una serie no estacionaria con memoria fuerte — coherente con el resultado del test ADF

**auto_arima:**
- Busca automáticamente combinaciones ARIMA(p, d, q) mediante minimización del AIC
- El orden de diferenciación `d` se determina de forma basada en datos a partir de pruebas de estacionariedad
- Selecciona el modelo más parsimonioso que ajusta la estructura de autocorrelación

## Conclusiones Clave

- **La estacionariedad es un requisito previo, no un supuesto a omitir:** ARIMA requiere que la serie sea estacionaria (media y varianza constantes). Un p-valor de 0,98 en el test ADF hace inconfundible la no estacionariedad — la diferenciación es obligatoria antes de ajustar.
- **La descomposición estacional es diagnóstica, no solo visual:** Descomponer la serie confirma que la deriva alcista es una tendencia, no un comportamiento cíclico — descartando ARIMA estacional (SARIMA) y simplificando el espacio de selección de modelos.
- **auto_arima realiza la búsqueda en cuadrícula que de otro modo sería manual:** Elegir (p, d, q) a mano requiere inspeccionar los gráficos ACF y PACF e iterar. `auto_arima` automatiza esto sistemáticamente usando criterios de información, siguiendo la misma lógica pero más rápido.

## Stack Tecnológico

`Python` · `statsmodels` · `pmdarima` · `pandas` · `Matplotlib` · `Seaborn`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_TimeSeries_MTK.git
cd ML_TimeSeries_MTK
pip install -r requirements.txt
jupyter notebook src/TimeSeries.ipynb
```

## Próximos Pasos

- Evaluar la precisión del pronóstico con datos de prueba reservados usando **MAE** y **RMSE** — un pronóstico visual de 60 pasos es convincente, pero el error cuantificado sobre datos no vistos es lo que lo hace creíble
- Aplicar **transformación logarítmica** antes de diferenciar para estabilizar la varianza de la serie con tendencia alcista
- Comparar con un modelo **Facebook Prophet** — Prophet gestiona automáticamente los cambios de tendencia y festivos, y a menudo supera a ARIMA en datos de ventas empresariales con patrones irregulares

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
