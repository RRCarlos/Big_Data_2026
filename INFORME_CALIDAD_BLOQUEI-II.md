# INFORME DE CALIDAD — Bloques I y II
## Data Analyst Mayo 2026

> **Fecha de generación:** 27/04/2026  
> **Versión:** 1.0  
> **Verificador:** Script automático de validación

---

## 📋 RESUMEN EJECUTIVO

| Bloque | Estado | Calificación |
|---|---|---|
| I - Python, pandas, EDA | ✅ Completado | ✅ APROBADO |
| II - Regresión | ✅ Completado | ✅ APROBADO |

---

## VERIFICACIÓN BLOQUE I: Python, pandas y EDA

### ✅ Archivos existentes

| Archivo | Estado | Ruta |
|---|---|---|
| resumen.md | ✅ Existe | `entregables/bloque_01_python_pandas_eda/resumen.md` |
| resultados.json | ✅ Existe | `entregables/bloque_01_python_pandas_eda/resultados.json` |
| Script Python | ✅ Existe | `entregables/bloque_01_python_pandas_eda/ejecutar_analisis.py` |

### ✅ Gráficos generados

| Gráfico | Estado |
|---|---|
| histograma_ventas.png | ✅ Generado |
| boxplot_canal.png | ✅ Generado |
| ventas_por_canal.png | ✅ Generado |
| ventas_por_region.png | ✅ Generado |

### ✅ Métricas detectadas (resultados.json)

```json
{
  "ventas_totales": 340149.86,
  "ventas_medias": 1889.72,
  "ventas_mediana": 1868.09,
  "num_registros": 180,
  "num_canales": 4,
  "num_regiones": 5,
  "canal_top": "web",
  "ventas_canal_top": 171402.7,
  "region_top": "Este",
  "ventas_region_top": 82244.11,
  "conclusiones": [
    "El canal web es el que genera mayores ventas totales...",
    "La región Este lidera en volumen de ventas...",
    "El ticket medio general es de 1,889.72..."
  ]
}
```

### ✅checklist de entrega

| Requisito | Estado |
|---|---|
| Dataset limpiado | ✅ |
| Análisis exploratorio completo | ✅ |
| Gráficos generados | ✅ (4/4) |
| Conclusiones de negocio | ✅ (3 redactadas) |
| Resumen.md creado | ✅ |

### ❌ No detectado

- ❌ Notebook .ipynb ejecutado (no requerido en opción B)

---

## VERIFICACIÓN BLOQUE II: Regresión y Comparación de Modelos

### ✅ Archivos existentes

| Archivo | Estado | Ruta |
|---|---|---|
| resumen.md | ✅ Existe | `entregables/bloque_02_regresion/resumen.md` |
| resultados.json | ✅ Existe | `entregables/bloque_02_regresion/resultados.json` |
| Script Python | ✅ Existe | `entregables/bloque_02_regresion/ejecutar_analisis.py` |

### ✅ Gráficos generados

| Gráfico | Estado |
|---|---|
| comparacion_modelos.png | ✅ Generado |
| predicciones_vs_reales.png | ✅ Generado |

### ✅ Métricas detectadas (resultados.json)

```json
{
  "modelos": [
    {"modelo": "Linear Regression", "MAE": 253.33, "RMSE": 316.62, "R2": 0.3625},
    {"modelo": "Ridge", "MAE": 251.03, "RMSE": 312.76, "R2": 0.3779},
    {"modelo": "Random Forest", "MAE": 280.24, "RMSE": 342.31, "R2": 0.2549}
  ],
  "mejor_modelo": "Ridge"
}
```

### ✅ Checklist de entrega

| Requisito | Estado |
|---|---|
| Train/test split (80/20) | ✅ |
| Pipeline de preprocesado | ✅ |
| 3 modelos entrenados | ✅ |
| Comparación de métricas | ✅ (MAE, RMSE, R²) |
| Análisis de mejor modelo | ✅ (Ridge) |
| Resumen.md creado | ✅ |

---

## 📊 MÉTRICAS FINALES

| Métrica | Valor |
|---|---|
| Bloques completados | 2 / 5 |
| Archivos generados | 14 |
| Gráficos generados | 6 |
| Conclusiones redactadas | 6 |
| Tiempo de ejecución | ~2 minutos |

---

## ✅ VEREDICTO FINAL

```
╔══════════════════════════════════════════════╗
║           ✅ TODO CORRECTO                 ║
║     Ambos bloques cumplimentados           ║
║     Sin errores críticos detectados         ║
╚══════════════════════════════════════════════╝
```

---

## 📁 UBICACIÓN DE ARCHIVOS

```
C:\Users\PC\Big_Data_2026\
├── Big_Data_2026_Contexto.md
├── Big_Data_2026_Memoria.md              ← (en construcción)
├── entregables\
│   ├── bloque_01_python_pandas_eda\
│   │   ├── resumen.md
│   │   ├── resultados.json
│   │   └── graficos\ (4 imágenes)
│   └── bloque_02_regresion\
│       ├── resumen.md
│       ├── resultados.json
│       └── graficos\ (2 imágenes)
```

---

*Informe generado automáticamente. No requiere intervención manual.*