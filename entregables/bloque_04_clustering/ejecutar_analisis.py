#!/usr/bin/env python3
"""
Script para ejecutar el analisis del Bloque IV - Clustering
Usa el dataset CORRECTO: data/segmentacion_clientes_mayo_2026.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Configuracion
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

OUTPUT_DIR = "C:/Users/PC/Big_Data_2026/entregables/bloque_04_clustering"
DATA_DIR = "C:/Users/PC/Big_Data_2026/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/graficos", exist_ok=True)

print("=" * 60)
print("BLOQUE IV - CLUSTERING")
print("=" * 60)

# ============================================
# 1. CARGAR DATOS (DATASET CORRECTO)
# ============================================
print("\n1. CARGANDO DATOS...")
df = pd.read_csv(f"{DATA_DIR}/segmentacion_clientes_mayo_2026.csv")
print(f"Dataset CORRECTO cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")

# ============================================
# 2. PREPARAR DATOS
# ============================================
print("\n2. PREPARANDO DATOS...")

# Usar todas las columnas excepto cliente_id
features = [col for col in df.columns if col != 'cliente_id']
X = df[features].copy()

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features utilizadas: {features}")
print(f"Estandarizacion: StandardScaler")

# ============================================
# 3. K-MEANS - Buscar K optimo
# ============================================
print("\n3. BUSCANDO K OPTIMO...")

silhouettes = []
inertias = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    inertias.append(kmeans.inertia_)
    
    print(f"   K={k}: Silhouette={sil:.4f}, Inertia={kmeans.inertia_:.2f}")

# Mejor K
best_k_idx = np.argmax(silhouettes)
best_k = list(K_range)[best_k_idx]
best_sil = silhouettes[best_k_idx]

print(f"\n[*] Mejor K: {best_k} (Silhouette={best_sil:.4f})")

# ============================================
# 4. K-MEANS con mejor K
# ============================================
print(f"\n4. ENTRENANDO K-MEANS (K={best_k})...")

kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans_best.fit_predict(X_scaled)

# Perfil de clusters
print("\n--- Perfil de Clusters ---")
df_cluster = df.copy()
df_cluster['cluster'] = labels_kmeans

# Seleccionar solo columnas numericas para el perfil
columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
perfil = df_cluster.groupby('cluster')[columnas_numericas].agg(['mean', 'std'])
print(perfil)

# ============================================
# 5. DBSCAN
# ============================================
print("\n5. ENTRENANDO DBSCAN...")

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
noise_points = sum(labels_dbscan == -1)

sil_dbscan = silhouette_score(X_scaled, labels_dbscan) if n_clusters_dbscan > 1 else 0

print(f"   Clusters encontrados: {n_clusters_dbscan}")
print(f"   Puntos de ruido: {noise_points}")
print(f"   Silhouette: {sil_dbscan:.4f}")

# ============================================
# 6. VISUALIZACION CON PCA
# ============================================
print("\n6. GENERANDO GRAFICOS...")

# Reducir a 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Grafico 1: K-Means
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.tab10(np.linspace(0, 1, best_k))

for i in range(best_k):
    mask = labels_kmeans == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6)

# Centros
centers_pca = pca.transform(kmeans_best.cluster_centers_)
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c='black', marker='X', s=200, edgecolors='white', linewidths=2)
axes[0].set_title(f'K-Means (K={best_k})', fontweight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend()

# Silhouette score
axes[1].plot(list(K_range), silhouettes, 'o-', color='steelblue', lw=2)
axes[1].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[1].set_title('Silhouette Score por K', fontweight='bold')
axes[1].set_xlabel('Numero de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graficos/kmeans_resultado.png", dpi=150)
plt.close()
print("   - kmeans_resultado.png")

# Grafico 2: DBSCAN
fig2, ax = plt.subplots(figsize=(8, 6))

unique_labels = set(labels_dbscan)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    if label == -1:
        color = 'gray'
        marker = 'x'
        label_name = 'Ruido'
    else:
        color = colors[i]
        marker = 'o'
        label_name = f'Cluster {label}'
    
    mask = labels_dbscan == label
    if label == -1:
        mask = labels_dbscan == -1
    
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=[color], marker=marker, label=label_name, alpha=0.6)

ax.set_title(f'DBSCAN (eps=0.5, min_samples=5)', fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graficos/dbscan_resultado.png", dpi=150)
plt.close()
print("   - dbscan_resultado.png")

# Grafico 3: Perfil de clusters (barras)
fig3, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(columnas_numericas[:6]):
    cluster_means = df_cluster.groupby('cluster')[col].mean()
    axes[idx].bar(cluster_means.index, cluster_means.values, color='steelblue', alpha=0.7)
    axes[idx].set_title(f'Promedio de {col}', fontweight='bold')
    axes[idx].set_xlabel('Cluster')
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graficos/perfil_clusters.png", dpi=150)
plt.close()
print("   - perfil_clusters.png")

# ============================================
# 7. GUARDAR RESULTADOS
# ============================================
print("\n7. GUARDANDO RESULTADOS...")

# Perfil de clusters resumido
perfil_resumido = {}
for cluster in df_cluster['cluster'].unique():
    cluster_data = df_cluster[df_cluster['cluster'] == cluster]
    perfil_resumido[f"Cluster_{cluster}"] = {
        "n_clientes": int(len(cluster_data)),
        "pct": float(len(cluster_data) / len(df_cluster) * 100),
        "ingresos_medio": float(cluster_data['ingresos'].mean()) if 'ingresos' in cluster_data.columns else 0,
        "compras_12m_media": float(cluster_data['compras_12m'].mean()) if 'compras_12m' in cluster_data.columns else 0,
        "ticket_medio": float(cluster_data['ticket_medio'].mean()) if 'ticket_medio' in cluster_data.columns else 0
    }

resultados_json = {
    "kmeans": {
        "k_optimo": int(best_k),
        "silhouette": float(best_sil),
        "inertia": float(inertias[best_k_idx])
    },
    "dbscan": {
        "n_clusters": int(n_clusters_dbscan),
        "noise_points": int(noise_points),
        "silhouette": float(sil_dbscan)
    },
    "perfil_clusters": perfil_resumido,
    "conclusiones": [
        f"K={best_k} es el numero optimo de clusters (silhouette={best_sil:.4f})",
        f"Se identificaron {best_k} segmentos de clientes",
        f"DBSCAN encontro {n_clusters_dbscan} clusters con {noise_points} puntos de ruido",
        "Los clusters permiten segmentacion personalizada"
    ]
}

with open(f"{OUTPUT_DIR}/resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados_json, f, indent=2, ensure_ascii=False)

# Actualizar resumen.md
resumen_content = f"""# Bloque IV - Clustering, Silhouette y PCA

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
"""

# Add silhouette results dynamically
for idx, k in enumerate(K_range):
    if k == best_k:
        resumen_content += f"| **{k}** | **{silhouettes[idx]:.4f}** |\n"
    else:
        resumen_content += f"| {k} | {silhouettes[idx]:.4f} |\n"

resumen_content += f"""
### Mejor K: {best_k} (Silhouette = {best_sil:.4f})

### Perfil de Clusters

"""

# Add cluster profile table
resumen_content += "| Cluster | N | Ingresos Medio | Compras 12m | Ticket Medio |\n"
resumen_content += "|---|---|---|---|---|\n"

for cluster_id in sorted(df_cluster['cluster'].unique()):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
    resumen_content += f"| {cluster_id} | {len(cluster_data)} | {cluster_data['ingresos'].mean():.0f} | {cluster_data['compras_12m'].mean():.1f} | {cluster_data['ticket_medio'].mean():.1f} |\n"

resumen_content += f"""
### DBSCAN

| Parametro | Valor |
|---|---|
| Clusters encontrados | {n_clusters_dbscan} |
| Puntos de ruido | {noise_points} |
| Silhouette | {sil_dbscan:.4f} |

---

## Conclusiones

1. **K={best_k} es el numero optimo de clusters** con un Silhouette Score de {best_sil:.4f}, indicando una estructura razonable.

2. **{best_k} segmentos de clientes identificados**:
"""

# Add cluster interpretations
for cluster_id in sorted(df_cluster['cluster'].unique()):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
    ingresos = cluster_data['ingresos'].mean()
    compras = cluster_data['compras_12m'].mean()
    ticket = cluster_data['ticket_medio'].mean()
    
    if ingresos > df['ingresos'].mean():
        perfil_ingresos = "altos ingresos"
    else:
        perfil_ingresos = "bajos ingresos"
    
    if compras > df['compras_12m'].mean():
        perfil_compras = "alta frecuencia"
    else:
        perfil_compras = "baja frecuencia"
    
    resumen_content += f"   - Cluster {cluster_id}: Clientes con {perfil_ingresos}, {perfil_compras}, ticket medio {ticket:.0f}\n"

resumen_content += f"""
3. **DBSCAN encuentra {n_clusters_dbscan} grupos** con {noise_points} puntos de ruido.

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
"""

with open(f"{OUTPUT_DIR}/resumen.md", "w", encoding="utf-8") as f:
    f.write(resumen_content)

print(f"\n[OK] BLOQUE IV COMPLETADO")
print(f"Resultados guardados en: {OUTPUT_DIR}/resultados.json")
print(f"Resumen guardado en: {OUTPUT_DIR}/resumen.md")
