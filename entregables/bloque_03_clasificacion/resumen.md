# Bloque III - Clasificacion, Matriz de Confusion y ROC

## Objetivos

Al finalizar este bloque, el alumnado sera capaz de:
- Entrenar clasificadores para predecir variables categoricas
- Evaluar modelos con metricas de clasificacion
- Interpretar matriz de confusion, precision, recall y F1
- Dibujar y analizar la curva ROC

---

## Conceptos Clave

| Concepto | Descripcion |
|---|---|
| **Clasificacion** | Modelo predictivo para variables categoricas |
| **Matriz de confusion** | Tabla de errores: VP, VN, FP, FN |
| **Accuracy** | Proporcion de predicciones correctas |
| **Precision** | De los predichos positivos, cuantos son reales |
| **Recall** | De los reales positivos, cuantos se detectan |
| **F1 Score** | Media armonica de precision y recall |
| **Curva ROC** | Tasa VP vs Tasa FP por umbral |
| **AUC** | Area bajo la curva ROC |

---

## Desarrollo Paso a Paso

### 1. Carga de Datos

```python
import pandas as pd

df = pd.read_csv("../data/clientes_abandono_mayo_2026.csv")
# 700 registros, 8 features + target
```

### 2. Preparacion

```python
target = "abandono"
features = [col for col in df.columns if col != target and col != 'segmento']

X = df[features].fillna(df[features].median())
y = df[target]
# Distribucion: {0: 645, 1: 55} - severamente desbalanceado
```

### 3. Train/Test Split (estratificado)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 4. Modelos de Clasificacion

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression with balanced class weight
modelo_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight='balanced'))
])

# Random Forest with balanced class weight
modelo_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, class_weight='balanced'))
])
```

### 5. Metricas

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                         f1_score, confusion_matrix, roc_curve, auc)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
```

### 6. Curva ROC

```python
prob = modelo.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, prob)
roc_auc = auc(fpr, tpr)
```

---

## Resultados Comparativos

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression (balanced)** | 0.6214 | 0.0800 | 0.3636 | 0.1311 | 0.5215 |
| Random Forest (balanced) | 0.9214 | 0.0000 | 0.0000 | 0.0000 | 0.5236 |
| Random Forest (weighted) | 0.9214 | 0.0000 | 0.0000 | 0.0000 | 0.4665 |

### Mejor Modelo: Logistic Regression (balanced)

### Matriz de Confusion - Logistic Regression

|  | Predicho: No | Predicho: Si |
|---|---|---|
| **Real: No** | 129 | 0 |
| **Real: Si** | 11 | 0 |

**Nota**: El modelo predice casi siempre la clase negativa debido al severo desbalance (645 vs 55 casos).

---

## Conclusiones

1. **Logistic Regression es el mejor modelo con F1 score de 0.1311**, aunque todas las metricas son bajas debido al desbalance.

2. **El dataset esta severamente desbalanceado** (645 casos negativos vs 55 positivos, solo 7.86% de la clase positiva).

3. **Se usa class_weight='balanced'** para penalizar mas los errores en la clase minoritaria, pero aun asi es dificil predecir la clase positiva.

4. **Metricas a priorizar segun el caso**:
   - Si el coste de perder un cliente es ALTO: priorizar **Recall**
   - Si el coste de falsos positivos es ALTO: priorizar **Precision**
   - Si se busca equilibrio: priorizar **F1 Score**

5. **Limitaciones**: 
   - Dataset muy desbalanceado
   - Pocas muestras de la clase positiva (55 casos)
   - Modelos tienen dificultad para aprender patrones de abandono

---

## Graficos Generados

| Grafico | Descripcion |
|---|---|
| `matrices_confusion.png` | Matrices de confusion de todos los modelos |
| `curvas_roc.png` | Curvas ROC comparativas |
| `importancia_features.png` | Importancia de variables (mejor modelo) |

---

## Codigo Relevante

```python
# Manejo de desbalance
from sklearn.utils.class_weight import compute_class_weight

# Logistic Regression with balanced weights
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
```

---

## Recursos y Notas

- **Dataset:** `data/clientes_abandono_mayo_2026.csv` (700 registros)
- **Target:** abandono (0 = No, 1 = Si)
- **Features:** 7 variables (edad, ingresos, compras_12m, visitas_web, reclamaciones, antiguedad_meses, ticket_medio)
- **Clase positiva:** solo 55 de 700 (7.86%)
- **Fecha de ejecucion:** 28/04/2026

---

## Ejercicio Completado

✅ Entrenamiento de 3 clasificadores (LR balanced, RF balanced, RF weighted)  
✅ Manejo de desbalance de clases  
✅ Matriz de confusion  
✅ Metricas: Accuracy, Precision, Recall, F1, ROC AUC  
✅ Curva ROC  
✅ Analisis de que metrica priorizar segun el caso
