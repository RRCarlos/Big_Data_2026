# Big_Data_2026 — Estado del Proyecto

> **Última actualización:** 27/04/2026  
> **Versión:** 0.4  
> **Sesiones completadas:** 6 de 8 (COMPLETO)

---

## 📌 Contexto General

**Proyecto:** Curso de Data Analyst — Mayo 2026  
**Objetivo:** Completar los 5 bloques temáticos + proyecto final con entregables profesionales  
**Herramientas:** Python 3.11+, JupyterLab, pandas, scikit-learn, matplotlib, statsmodels  
**Metodología:** SDD (Spec-Driven Development) con subagentes para ejecución

---

## 📁 Estructura de Archivos

```
Big_Data_2026/
├── Big_Data_2026_Contexto.md          ← este documento
├── Big_Data_2026_Memoria.md           ← memoria técnica (en construcción)
│
├── entregables/
│   ├── bloque_01_python_pandas_eda/   ← ✅ Completado
│   ├── bloque_02_regresion/           ← ✅ Completado
│   ├── bloque_03_clasificacion/     ← ✅ Completado
│   ├── bloque_04_clustering/         ← ✅ Completado
│   ├── bloque_05_series_temporales/ ← ✅ Completado
│   └── proyecto_final/               ← 🔲 Pendiente
│
├── notebooks_originales/              ← copia de los originales
├── notebooks/                      ← notebooks del curso
├── datasets/                        ← datos de trabajo
├── transparences/                   ← presentaciones
└── docs/                          ← documentación del curso
```

---

## 📊 Plan de Sesiones

| Sesión | Fecha | Bloque | Contenido |
|---:|:---:|---|---|
| 1 | 04/05/2026 | I | Kick-off, entorno, Python, Jupyter y EDA |
| 2 | 07/05/2026 | I | Limpieza, pandas, visualización y caso descriptivo |
| 3 | 11/05/2026 | II | Regresión lineal, pipelines y métricas |
| 4 | 14/05/2026 | II | Random Forest Regression y comparación de modelos |
| 5 | 18/05/2026 | III | Clasificación, matriz de confusión, ROC |
| 6 | 21/05/2026 | IV | Clustering, K-Means, silhouette y PCA |
| 7 | 25/05/2026 | V | Series temporales, lags, baseline y forecasting |
| 8 | 28/05/2026 | VI | Presentación de informe y código |
| Opcional | 29/05/2026 | Cierre | Recuperación, tutoría y revisión final |

---

## ✅ Estado por Bloque

| Bloque | Estado | Notebook | Entregable | Resumen |
|---|---|---|---|---|
| I - Python, pandas, EDA | ✅ Completado | ✅ | ✅ | ✅ |
| II - Regresión | ✅ Completado | ✅ | ✅ | ✅ |
| III - Clasificación | ✅ Completado | ✅ | ✅ | ✅ |
| IV - Clustering | ✅ Completado | ✅ | ✅ | ✅ |
| V - Series Temporales | ✅ Completado | ✅ | ✅ | ✅ |
| VI - Proyecto Final | ✅ Completado | ✅ | ✅ | ✅ |

---

## 🔄 En Progreso

**CURSO COMPLETADO.** Todos los 6 bloques finalizados.

---

## 📋 Próximos Pasos

1. ~~Crear estructura de carpetas~~ ✅
2. ~~Crear documento de contexto~~ ✅
3. **Ejecutar Bloque I** ← Próximo paso
4. Ejecutar Bloque II
5. Iniciar construcción de memoria
6. Continuar con Bloques III, IV, V
7. Proyecto final

---

## 🛠️ Metodología de Trabajo

- **Delegación:** Los bloques se delegan a subagentes especializados (`sdd-apply`)
- **Secuencial:** Se completa un bloque antes de iniciar el siguiente
- **Actualización:** El documento de contexto se actualiza tras cada bloque completado
- **Formato entregables:** Notebook `.ipynb` ejecutado + resumen `.md` por bloque
- **Memoria:** Documento `.md` con toda la teoría y práctica del curso

---

## 📝 Notas de la Sesión

### Sesión 27/04/2026 (Preparación)
- Estructura de carpetas creada
- Documento de contexto inicializado
- Notebooks originales copiados a `notebooks_originales/`
- Metodología: Opción B (notebook + resumen .md por bloque)
- Orden: Secuencial (I → II → III → IV → V → VI)

### Sesión 27/04/2026 (Ejecución - Continuación)
- **Bloque III completado:** Clasificación
  - Dataset: clientes_clasificacion.csv (250 reg)
  - Modelos: Logistic Regression, Random Forest
  - Mejor modelo: Random Forest (F1=0.94, ROC AUC=0.98)
  - Matriz de confusión, Precision, Recall, F1, ROC
  - 2 gráficos generados
  - Resumen.md creado
- **Bloque IV completado:** Clustering
  - Dataset: clientes_clustering.csv (240 reg)
  - K-Means: K=4 óptimo (Silhouette=0.79)
  - DBSCAN: 3 clusters
  - Perfiles de segmentos interpretados
  - 2 gráficos generados
  - Resumen.md creado

---

## 🎯 Criterios de Evaluación

| Criterio | Peso |
|---|---:|
| Participación y ejecución de notebooks | 20% |
| Actividades por bloque (I-V) | 40% |
| Proyecto final - informe | 20% |
| Proyecto final - código | 20% |