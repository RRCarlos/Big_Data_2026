# Bloque II — Regresión y Comparación de Modelos

## Objetivos

Al finalizar este bloque, el alumnado será capaz de:
- Construir modelos de regresión para predecir una variable numérica continua
- Comparar distintos algoritmos de regresión mediante métricas
- Justificar la elección del modelo según los resultados
- Identificar el público objetivo basándose en análisis regional

---

## Conceptos Clave

| Concepto | Descripción |
|---|---|
| **Regresión** | Modelo predictivo para variables continuas |
| **Train/Test Split** | Separación de datos para entrenamiento y evaluación |
| **Pipeline** | Secuencia de transformadores + modelo |
| **Preprocesado** | Imputación, escalado y codificación de variables |
| **MAE** | Mean Absolute Error — error absoluto medio |
| **RMSE** | Root Mean Squared Error — raíz del error cuadrático medio |
| **R²** | Coeficiente de determinación — varianza explicada |
| **Regularización** | Técnica para evitar sobreajuste (Ridge, Lasso) |

---

## Desarrollo Paso a Paso

### 1. Carga y Preparación (Dataset CORRECTO)

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Usar dataset CORRECTO
df = pd.read_csv("../data/ventas_mayo_2026.csv")

df = df.drop_duplicates()
df["fecha"] = pd.to_datetime(df["fecha"])
df["mes"] = df["fecha"].dt.month
```

**Dataset:** `ventas_mayo_2026.csv` con 505 registros

### 2. Definición de Features

```python
target = "importe"  # Cambiado de 'ventas' a 'importe'

features_num = ["unidades", "precio_unitario", "descuento", "antiguedad_cliente_meses"]
features_cat = ["region", "canal", "categoria"]

X = df[features_num + features_cat]
y = df[target]
```

### 3. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 4. Pipeline de Preprocesado

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, features_num),
        ("cat", categorical_transformer, features_cat)
    ])
```

### 5. Modelos de Regresión

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Linear Regression
modelo_lr = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Ridge
modelo_ridge = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

# Random Forest
modelo_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])
```

### 6. Métricas de Evaluación

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
```

---

## Resultados Comparativos

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | 281.45 | 364.12 | 0.4856 |
| Linear Regression | 295.33 | 378.91 | 0.4521 |
| Ridge | 298.17 | 381.45 | 0.4478 |

### Análisis de Resultados

| Métrica | Significado | Mejor Valor |
|---|---|---|
| **MAE** | Error medio en euros | Bajo |
| **RMSE** | Penaliza errores grandes | Bajo |
| **R²** | % de varianza explicada | Alto (cerca de 1) |

---

## Análisis de Región (Público Objetivo)

### Importe Total por Región

| Región | Importe Total |
|---|---:|
| **Madrid** | 135,629.17 € |
| Valencia | 130,746.69 € |
| Andalucía | 123,773.33 € |
| Cataluña | 106,247.09 € |
| Castilla-La Mancha | 106,185.85 € |

**Conclusión clave:** Madrid es la región con mayor importe total (135,629.17 €), siendo el público objetivo principal.

---

## Conclusiones

1. **El modelo Random Forest presenta el menor MAE (281.45 €)**, indicando las predicciones más precisas para este dataset.

2. **El R² de 0.4856 indica que el modelo explica el 48.6% de la variabilidad en las ventas**. Este valor es razonable y superior al obtenido con el dataset incorrecto.

3. **Madrid es la región con mayor importe total (135,629.17 €)**, confirmando que debe ser el público objetivo principal para las campañas de marketing.

4. **Random Forest captura mejor las relaciones no lineales entre features**, especialmente con el dataset completo de 505 registros.

---

## Gráficos Generados

| Gráfico | Descripción |
|---|---|
| `comparacion_modelos.png` | Comparación visual de métricas |
| `predicciones_vs_reales.png` | Dispersión de predicciones vs valores reales |
| `importe_por_region.png` | Importe total por región (NUEVO) |

---

## Código Relevante

```python
# Pipeline completo con mejores features
modelo_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

modelo_rf.fit(X_train, y_train)
predicciones = modelo_rf.predict(X_test)
```

---

## Recursos y Notas

- **Dataset:** `data/ventas_mayo_2026.csv` (CORRECTO - 505 registros)
- **Notebook original:** `notebooks/02_Bloque_II_Regresion_Comparacion_Modelos_3h.ipynb`
- **Entregable:** Análisis de regresión completo
- **Fecha de ejecución:** 28/04/2026 (CORREGIDO)
- **Features utilizadas:** unidades, precio_unitario, descuento, antiguedad_cliente_meses, region, canal, categoria

---

## Ejercicio Completado

✅ Separación train/test (80/20)  
✅ Pipeline de preprocesado  
✅ Entrenamiento de 3 modelos (LR, Ridge, RF)  
✅ Comparación de métricas MAE, RMSE, R²  
✅ Análisis regional para identificar público objetivo  
✅ Conclusión: Madrid es el público objetivo (135,629.17 €)  

---

## Nota sobre la Corrección

El análisis anterior usaba un dataset incorrecto (`datasets/ventas_mayo2026.csv` con solo 180 registros y estructura diferente). Se ha corregido para usar el dataset oficial (`data/ventas_mayo_2026.csv` con 505 registros).

**Cambios principales:**
- Target cambiado de `ventas` a `importe`
- Añadidas nuevas features: `categoria`, `unidades`, `precio_unitario`, `descuento`, `antiguedad_cliente_meses`
- Nueva conclusión: **Madrid es la región con mayor importe (135,629.17 €)**, siendo el público objetivo principal.
