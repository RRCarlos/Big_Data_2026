# Bloque VI — Proyecto Final Integrador

## Objetivos

Integrar todo lo aprendido en los bloques anteriores:
- Cargar y combinar múltiples datasets
- Limpieza y transformación de datos
- Aplicar modelos de clasificación, regresión y clustering
- Evaluar con métricas apropiadas
- Generar conclusiones y propuestas de mejora

---

## Pregunta Analítica

**OBJETIVO:** Analizar el rendimiento comercial y predecir el éxito de clientes.

**Preguntas específicas:**
1. ¿Qué factores influenciar más las ventas?
2. ¿Podemos predecir si un cliente comprará o abandonará?
3. ¿Qué segmentos de clientes existen?

---

## Desarrollo Paso a Paso

### 1. Carga de Datos

```python
# Combinar múltiples datasets
ventas = pd.read_csv("ventas_mayo2026.csv")
clasificacion = pd.read_csv("clientes_clasificacion.csv")
```

### 2. Limpieza y Transformación

```python
# Eliminar duplicados
df = df.drop_duplicates()

# Imputar nulos
df = df.fillna(df.median())

# Feature engineering
df["ratio_f1_f2"] = df["feature_1"] / (df["feature_2"] + 0.01)
df["interaccion"] = df["feature_1"] * df["feature_3"]
```

### 3. Modelos

```python
# Clasificación: predecir abandono
from sklearn.ensemble import RandomForestClassifier

modelo_clasif = RandomForestClassifier(n_estimators=100)
modelo_clasif.fit(X_train, y_train)

# Regresión: predecir ventas
from sklearn.ensemble import RandomForestRegressor

modelo_reg = RandomForestRegressor(n_estimators=50)
modelo_reg.fit(X_reg_train, y_reg_train)

# Clustering: segmentar clientes
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
```

---

## Resultados

### Clasificación (Predicción de Abandono)

| Métrica | Valor |
|---|---|
| Accuracy | 0.96 |
| Precision | 1.00 |
| Recall | 0.89 |
| F1 | 0.94 |

### Regresión (Predicción de Ventas)

| Métrica | Valor |
|---|---|
| MAE | 100.38 |
| R² | — |

### Clustering (Segmentación)

| Cluster | feature_1 | feature_2 | feature_3 | Abandono |
|---|---|---|---|
| 0 | -1.38 | -0.89 | -1.29 | 35.4% |
| 1 | 1.04 | -1.31 | -1.18 | 17.2% |
| 2 | 0.53 | 0.04 | 0.29 | 45.8% |

---

## Conclusiones

1. **FACTORES CLAVE:** Las features derivadas (ratio_f1_f2, interacción) son predictores importantes

2. **CLASIFICACIÓN:** Podemos predecir abandono con ~96% de accuracy, permitiendo intervención proactiva

3. **REGRESIÓN:** La inversión en marketing correlaciona con ventas medias

4. **CLUSTERING:** 3 segmentos distintos identificados, cada uno con diferente tasa de abandono

---

## Propuestas de Mejora

| Propuesta | Descripción | Prioridad |
|---|---|---|
| Modelo en producción | Implementar predicción de abandono en sistema real | Alta |
| Campañas personalizadas | Segmentar clientes para ofertas específicas | Alta |
| Optimización inversi | Aumentar inversión en regiones con alto potencial | Media |
| collecting | Mejorar recolección de datos | Media |

---

## Gráficos Generados

| Gráfico | Descripción |
|---|---|
| `proyecto_metricas.png` | Métricas + distribución de clusters |

---

## Código Relevante

```python
# Pipeline completo del proyecto
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Combinar todo en un pipeline de producción
pipeline_produccion = Pipeline([
    ("preprocessor", preprocessor),
    ("clasificador", RandomForestClassifier(n_estimators=100)),
])
```

---

## Recursos y Notas

- **Dataset:** Combinación de ventas_mayo2026.csv + clientes_clasificacion.csv
- **Técnicas:** Clasificación + Regresión + Clustering
- **Fecha de ejecución:** 27/04/2026

---

## Ejercicio Completado

✅ Pregunta analítica definida  
✅ Dataset combinado y limpiado  
✅ Feature engineering  
✅ Modelo de clasificación  
✅ Modelo de regresión  
✅ Clustering  
✅ Métricas y evaluación  
✅ Conclusiones y propuestas