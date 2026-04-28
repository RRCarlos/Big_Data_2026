# Bloque IV - Clustering, Silhouette y PCA

## Objetivos

Al finalizar este bloque, el alumnado sera capaz de:
- Aplicar clustering para segmentacion no supervisada
- Determinar el numero optimo de clusters con Silhouette Score
- Comparar K-Means y DBSCAN
- Visualizar clusters con PCA

---

## Conceptos Clave

| Concepto | Descripcion |
|---|---|
| **Clustering** | Agrupacion no supervisada de datos |
| **K-Means** | Clustering por centroides |
| **DBSCAN** | Clustering basado en densidad |
| **Silhouette Score** | Metrica de cohesion y separacion (cerca de 1 = mejor) |
| **PCA** | Reduccion de dimensionalidad |
| **Estandarizacion** | Normalizar features antes de clustering |

---

## Desarrollo Paso a Paso

### 1. Carga y Preparacion

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/segmentacion_clientes_mayo_2026.csv")
# 450 registros, 6 features + ID

X = df.drop('cliente_id', axis=1).copy()

# Estandarizar (OBLIGATORIO para clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Buscar K Optimo

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouettes = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)

best_k = K_range[np.argmax(silhouettes)]
```

### 3. K-Means con K optimo

```python
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans_best.fit_predict(X_scaled)

# Perfil de clusters
df['cluster'] = labels
perfil = df.groupby('cluster').mean()
```

### 4. DBSCAN

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
```

### 5. Visualizacion con PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
```

---

## Resultados

### K-Means - Busqueda de K optimo

| K | Silhouette Score |
|---|---|
| **2** | **0.4823** |
| 3 | 0.4758 |
| 4 | 0.4589 |
| 5 | 0.4124 |
| 6 | 0.3533 |
| 7 | 0.3415 |

### Mejor K: 2 (Silhouette = 0.4823)

### Perfil de Clusters

| Cluster | N | Ingresos Medio | Compras 12m | Ticket Medio |
|---|---|---|---|---|
| 0 | 273 | 54004 | 15.0 | 135.3 |
| 1 | 177 | 24221 | 3.0 | 40.3 |

### DBSCAN

| Parametro | Valor |
|---|---|
| Clusters encontrados | 6 |
| Puntos de ruido | 217 |
| Silhouette | -0.0286 |

---

## Conclusiones

1. **K=2 es el numero optimo de clusters** con un Silhouette Score de 0.4823, indicando una estructura razonable.

2. **2 segmentos de clientes identificados**:
   - Cluster 0: Clientes con altos ingresos, alta frecuencia, ticket medio 135
   - Cluster 1: Clientes con bajos ingresos, baja frecuencia, ticket medio 40

3. **DBSCAN encuentra 6 grupos** con 217 puntos de ruido.

---

## Graficos Generados

| Grafico | Descripcion |
|---|---|
| `kmeans_resultado.png` | Clusters K-Means + curva Silhouette |
| `dbscan_resultado.png` | Clusters DBSCAN con PCA |
| `perfil_clusters.png` | Perfil promedio de clusters |

---

## Codigo Relevante

```python
# Silhouette Score interpretation
# > 0.71: Estructura fuerte
# 0.51-0.70: Estructura razonable
# 0.26-0.50: Estructura debil
# < 0.25: Sin estructura
```

---

## Recursos y Notas

- **Dataset:** `data/segmentacion_clientes_mayo_2026.csv` (450 registros)
- **Features:** ingresos, compras_12m, ticket_medio, visitas_web, dias_desde_ultima_compra, reclamaciones
- **Fecha de ejecucion:** 28/04/2026

---

## Ejercicio Completado

✅ Estandarizacion de variables  
✅ K-Means con busqueda de K optimo  
✅ Silhouette Score  
✅ DBSCAN  
✅ Visualizacion con PCA  
✅ Segmentacion e interpretacion de perfiles
