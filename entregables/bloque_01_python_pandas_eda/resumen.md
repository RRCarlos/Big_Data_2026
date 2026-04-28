# Bloque I — Python, pandas y Análisis Descriptivo

## Objetivos

Al finalizar este bloque, el alumnado será capaz de:
- Cargar, inspeccionar, limpiar, transformar, describir y visualizar un dataset tabular con Python y `pandas`
- Generar conclusiones iniciales útiles para un proyecto de análisis de datos

---

## Conceptos Clave

| Concepto | Descripción |
|---|---|
| **EDA (Exploratory Data Analysis)** | Análisis exploratorio de datos para entender su estructura y contenido |
| **pandas** | Librería principal para trabajar con datos tabulares (DataFrames) |
| **numpy** | Librería para operaciones numéricas y arrays |
| **matplotlib** | Librería para visualización de datos |
| **Limpieza de datos** | Tratamiento de nulos, duplicados, tipos y outliers |
| **Variables derivadas** | Nuevas columnas calculadas a partir de existentes |
| **Análisis descriptivo** | Métricas resumen: media, mediana, desviación, percentiles |

---

## Desarrollo Paso a Paso

### 1. Carga y Exploración

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar dataset CORRECTO
df = pd.read_csv("../data/ventas_mayo_2026.csv")

# Exploración básica
df.head()           # Primeras 5 filas
df.shape            # Dimensiones (505 filas, 10 columnas)
df.info()           # Tipos de datos y nulos
df.describe()       # Estadísticas básicas
```

**Dataset:** `ventas_mayo_2026.csv` con 505 registros de ventas

### 2. Limpieza de Datos

```python
# Copia para no modificar original
df_limpio = df.copy()

# Eliminar duplicados
df_limpio = df_limpio.drop_duplicates()

# Convertir fecha a datetime
df_limpio["fecha"] = pd.to_datetime(df_limpio["fecha"])

# Imputar nulos
df_limpio["importe"] = df_limpio["importe"].fillna(df_limpio["importe"].median())
df_limpio["canal"] = df_limpio["canal"].fillna("Sin informar")

# Crear variables derivadas
df_limpio["mes"] = df_limpio["fecha"].dt.month
df_limpio["anio"] = df_limpio["fecha"].dt.year
df_limpio["dia_semana"] = df_limpio["fecha"].dt.day_name()
```

### 3. Análisis Descriptivo

```python
# Métricas de importe (ventas)
metricas = df_limpio["importe"].agg(["count", "mean", "median", "std", "min", "max"])

# Importe por región
importe_region = df_limpio.groupby("region")["importe"].sum().sort_values(ascending=False)

# Importe por canal
importe_canal = df_limpio.groupby("canal").agg(
    importe_total=("importe", "sum"),
    importe_medio=("importe", "mean"),
    num_operaciones=("fecha", "count")
).sort_values("importe_total", ascending=False)

# Ticket medio por región
ticket_region = df_limpio.groupby("region")["importe"].mean().sort_values(ascending=False)
```

### 4. Visualización

```python
# Histograma
plt.hist(df_limpio["importe"], bins=30)
plt.title("Distribución del Importe de Ventas")
plt.show()

# Boxplot por canal
df_limpio.boxplot(column="importe", by="canal")

# Barras por canal
importe_canal.plot(kind="bar")
```

---

## Resultados Principales

| Métrica | Valor |
|---|---:|
| **Importe total (Ventas)** | 619,527.35 € |
| **Importe medio** | 1,226.79 € |
| **Importe mediana** | 1,007.81 € |
| **Número de registros** | 505 |
| **Número de canales** | 3 |
| **Número de regiones** | 5 |

### Importe por Canal

| Canal | Importe Total |
|---|---:|
| **Online** | 261,714.70 € |
| Tienda | 220,526.13 € |
| Distribuidor | 137,286.52 € |

### Importe por Región

| Región | Importe Total |
|---|---:|
| **Madrid** | 135,629.17 € |
| Valencia | 130,746.69 € |
| Andalucía | 123,773.33 € |
| Cataluña | 106,247.09 € |
| Castilla-La Mancha | 106,185.85 € |

### Ticket Medio por Región

| Región | Ticket Medio |
|---|---:|
| Valencia | 1,269.39 € |
| Madrid | 1,255.83 € |
| Castilla-La Mancha | 1,249.25 € |
| Andalucía | 1,201.68 € |
| Cataluña | 1,106.74 € |

---

## Conclusiones de Negocio

1. **El canal Online es el que genera mayores ventas totales (261,714.70 €)**, representando la principal vía de facturación y el segmento principal para la empresa. Esto confirma que la estrategia digital es altamente efectiva.

2. **La región Madrid lidera en volumen de ventas (135,629.17 €)**, indicando mayor penetración de mercado o eficiencia comercial en esa zona. Madrid debe ser el foco de las próximas campañas de marketing.

3. **El ticket medio general es de 1,226.79 €, con la región Valencia presentando el mayor ticket medio (1,269.39 €)**. Esto sugiere que los clientes de Valencia tienen mayor poder adquisitivo o responden mejor a productos de mayor valor.

---

## Gráficos Generados

| Gráfico | Descripción |
|---|---|
| `histograma_ventas.png` | Distribución del importe de ventas |
| `boxplot_canal.png` | Distribución de ventas por canal |
| `ventas_por_canal.png` | Importe total por canal (barras) |
| `ventas_por_region.png` | Importe total por región (barras) |

---

## Recursos y Notas

- **Dataset:** `data/ventas_mayo_2026.csv` (CORRECTO - 505 registros)
- **Notebook original:** `notebooks/01_Bloque_I_Python_Pandas_Analisis_Descriptivo_3h.ipynb`
- **Entregable:** Notebook ejecutado + análisis completo
- **Fecha de ejecución:** 28/04/2026 (CORREGIDO)

---

## Ejercicio Integrador Completado

✅ Calcular importe total por canal (Online: 261,714.70 €)  
✅ Calcular ticket medio por región (Valencia: 1,269.39 €)  
✅ Identificar categoría con mayor número de operaciones (Online)  
✅ Generar gráfico de barras con ventas por canal  
✅ Redactar tres conclusiones de negocio  

---

## Nota sobre la Corrección

El análisis anterior usaba un dataset incorrecto (`datasets/ventas_mayo2026.csv` con solo 180 registros). Se ha corregido para usar el dataset oficial (`data/ventas_mayo_2026.csv` con 505 registros). Esto cambia las conclusiones:

- **Antes:** El canal web/Online tenía 171,402.70 €
- **Ahora:** El canal Online tiene **261,714.70 €** (valor correcto)

**Conclusión principal:** Online es el canal con mayores ventas (261,714.70 €), confirmando que es la segmentación principal para la empresa.
