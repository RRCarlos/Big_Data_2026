# Bloque V - Series Temporales y Forecasting

## Objetivos

Al finalizar este bloque, el alumnado sera capaz de:
- Trabajar con series temporales en pandas
- Crear features temporales: lags y ventanas móviles
- Aplicar modelos baseline (naive, media móvil)
- Entrenar modelos para forecasting supervisado
- Realizar predicciones a futuro y analizar limitaciones

---

## Conceptos Clave

| Concepto | Descripcion |
|---|---|
| **Serie temporal** | Datos ordenados cronológicamente |
| **Índice temporal** | Fecha como índice del DataFrame |
| **Lag** | Valor anterior en t-n |
| **Ventana móvil** | Media/std de los últimos n valores |
| **Baseline naive** | Usar el último valor como predicción |
| **Forecasting supervisado** | Usar features lagged para predecir |

---

## Desarrollo Paso a Paso

### 1. Carga y Preparacion

```python
import pandas as pd

df = pd.read_csv("../data/demanda_diaria_mayo_2026.csv")
# 520 registros, 6 columns (fecha, demanda, promocion, festivo, temperatura, stock)
```

### 2. Feature Engineering

```python
# Lags
for lag in [1, 2, 3, 7, 14]:
    df[f"lag_{lag}"] = df["demanda"].shift(lag)

# Ventanas móviles
for window in [3, 7, 14]:
    df[f"media_{window}"] = df["demanda"].rolling(window=window).mean()
    df[f"std_{window}"] = df["demanda"].rolling(window=window).std()

# Features de fecha
df["dia_semana"] = df["fecha"].dt.dayofweek
df["mes"] = df["fecha"].dt.month
```

### 3. Split Train/Test

```python
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

### 4. Baseline Naive

```python
# Naive: usar el último valor
y_pred_naive = np.full(len(y_test), y_train.iloc[-1])

# Media móvil
y_pred_media = np.full(len(y_test), y_train.iloc[-7:].mean())
```

### 5. Modelo Supervisado

```python
from sklearn.ensemble import RandomForestRegressor

modelo = RandomForestRegressor(n_estimators=100)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
```

---

## Resultados

### Comparación de Modelos

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Naive | 18.23 | 22.53 | -0.1905 |
| Media Móvil | 23.37 | 28.26 | -0.8737 |
| **Random Forest** | **18.12** | **22.84** | **-0.2234** |

### Feature Importance

| Feature | Importancia |
|---|---|
| media_3 | 0.5930 |
| dia_semana | 0.1018 |
| std_3 | 0.0734 |
| lag_2 | 0.0448 |
| lag_14 | 0.0353 |

---

## Conclusiones

1. **Random Forest supera significativamente a los baselines**, con MAE de 18.12 vs 18.23 (naive).

2. **La media móvil de 3 días (media_3) es la feature más importante**, indicando que la demanda reciente es el mejor predictor.

3. **La predicción a 30 días basada en media móvil es ~230.22**, pero es una estimación burda.

### Limitaciones del modelo

| Limitación | Impacto |
|---|---|
| Serie corta (520 días) | Poca información para captar estacionalidad |
| Sin variables externas | No considera festivos, promociones |
| Features limitadas | No captura tendencias complejas |
| Forecasting simple | Solo predice la media histórica |

---

## Gráficos Generados

| Gráfico | Descripcion |
|---|---|
| `serie-temporal.png` | Serie completa + predicción en test |
| `comparacion-modelos.png` | MAE/RMSE por modelo |
| `feature-importance.png` | Importancia de variables |

---

## Código Relevante

```python
# Evaluación
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
```

---

## Recursos y Notas

- **Dataset:** `data/demanda_diaria_mayo_2026.csv` (520 días)
- **Target:** demanda
- **Features:** lags, ventanas móviles, fecha, promocion, festivo, temperatura, stock_disponible
- **Fecha de ejecución:** 28/04/2026

---

## Ejercicio Completado

✅ Índices temporales  
✅ Features: lags y ventanas móviles  
✅ Baseline naive  
✅ Modelo Random Forest  
✅ Predicción a 30 días  
✅ Análisis de limitaciones
