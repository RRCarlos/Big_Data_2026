#!/usr/bin/env python3
"""
Script para ejecutar el análisis del Bloque I y capturar todos los outputs
Usa el dataset CORRECTO: data/ventas_mayo_2026.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

# Configuración
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

# ============================================
# CARGAR DATOS (DATASET CORRECTO)
# ============================================
print("=" * 60)
print("CARGA DE DATOS")
print("=" * 60)

df = pd.read_csv("C:/Users/PC/Big_Data_2026/data/ventas_mayo_2026.csv")
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")
print("\nPrimeras 5 filas:")
print(df.head().to_string())

# ============================================
# EXPLORACIÓN INICIAL
# ============================================
print("\n" + "=" * 60)
print("EXPLORACIÓN INICIAL")
print("=" * 60)

print(f"\nForma del dataset: {df.shape}")
print(f"\nInfo del dataset:")
df.info()
print(f"\nDescribe (include='all'):")
print(df.describe(include="all").to_string())
print(f"\nValores nulos por columna:")
print(df.isnull().sum().to_string())
print(f"\nDuplicados: {df.duplicated().sum()}")

# ============================================
# LIMPIEZA DE DATOS
# ============================================
print("\n" + "=" * 60)
print("LIMPIEZA DE DATOS")
print("=" * 60)

df_limpio = df.copy()

# Eliminar duplicados
print(f"Duplicados antes: {df_limpio.duplicated().sum()}")
df_limpio = df_limpio.drop_duplicates()
print(f"Duplicados después: {df_limpio.duplicated().sum()}")

# Convertir fecha
df_limpio["fecha"] = pd.to_datetime(df_limpio["fecha"])
print("\nFecha convertida a datetime")

# Imputar nulos en importe (antes ventas)
print(f"\nNulos en 'importe' antes: {df_limpio['importe'].isnull().sum()}")
df_limpio["importe"] = df_limpio["importe"].fillna(df_limpio["importe"].median())
print(f"Nulos en 'importe' después: {df_limpio['importe'].isnull().sum()}")

# Imputar nulos en canal
print(f"\nNulos en 'canal' antes: {df_limpio['canal'].isnull().sum()}")
df_limpio["canal"] = df_limpio["canal"].fillna("Sin informar")
print(f"Nulos en 'canal' después: {df_limpio['canal'].isnull().sum()}")

# Variables derivadas
df_limpio["mes"] = df_limpio["fecha"].dt.month
df_limpio["anio"] = df_limpio["fecha"].dt.year
df_limpio["dia_semana"] = df_limpio["fecha"].dt.day_name()
print("\nVariables derivadas creadas: mes, anio, dia_semana")

print("\nDataset limpio - Primeras 5 filas:")
print(df_limpio.head().to_string())

# ============================================
# ANÁLISIS DESCRIPTIVO
# ============================================
print("\n" + "=" * 60)
print("ANÁLISIS DESCRIPTIVO")
print("=" * 60)

# Métricas de importe
print("\n--- Métricas de Importe (Ventas) ---")
metricas_importe = df_limpio["importe"].agg(["count", "mean", "median", "std", "min", "max"])
print(metricas_importe.to_string())

# Importe por región
print("\n--- Importe Total por Región ---")
importe_por_region = df_limpio.groupby("region")["importe"].sum().sort_values(ascending=False)
print(importe_por_region.to_string())

# Importe por canal
print("\n--- Importe Total por Canal ---")
importe_por_canal = df_limpio.groupby("canal")["importe"].sum().sort_values(ascending=False)
print(importe_por_canal.to_string())

# Ticket medio por región
print("\n--- Ticket Medio por Región ---")
ticket_medio_region = df_limpio.groupby("region")["importe"].mean().sort_values(ascending=False)
print(ticket_medio_region.to_string())

# Análisis por canal
print("\n--- Análisis por Canal ---")
analisis_canal = df_limpio.groupby("canal").agg(
    importe_total=("importe", "sum"),
    importe_medio=("importe", "mean"),
    num_operaciones=("fecha", "count"),
    clientes_unicos=("cliente_id", "nunique")
).sort_values("importe_total", ascending=False)
print(analisis_canal.to_string())

# ============================================
# VISUALIZACIÓN
# ============================================
print("\n" + "=" * 60)
print("GENERANDO VISUALIZACIONES")
print("=" * 60)

# Crear directorio para gráficos
os.makedirs("C:/Users/PC/Big_Data_2026/entregables/bloque_01_python_pandas_eda/graficos", exist_ok=True)

# Gráfico 1: Histograma de importe
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.hist(df_limpio["importe"], bins=30, color='steelblue', edgecolor='black')
ax1.set_title("Distribución del Importe de Ventas", fontsize=14, fontweight='bold')
ax1.set_xlabel("Importe de Ventas (€)")
ax1.set_ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("C:/Users/PC/Big_Data_2026/entregables/bloque_01_python_pandas_eda/graficos/histograma_ventas.png", dpi=150)
plt.close()
print("Gráfico 1: histograma_ventas.png")

# Gráfico 2: Boxplot por canal
fig2, ax2 = plt.subplots(figsize=(10, 5))
df_limpio.boxplot(column="importe", by="canal", ax=ax2)
ax2.set_title("Distribución de Ventas por Canal", fontsize=14, fontweight='bold')
ax2.set_xlabel("Canal")
ax2.set_ylabel("Importe (€)")
plt.suptitle("")
plt.tight_layout()
plt.savefig("C:/Users/PC/Big_Data_2026/entregables/bloque_01_python_pandas_eda/graficos/boxplot_canal.png", dpi=150)
plt.close()
print("Gráfico 2: boxplot_canal.png")

# Gráfico 3: Importe por canal (barras)
fig3, ax3 = plt.subplots(figsize=(10, 5))
importe_por_canal.plot(kind="bar", ax=ax3, color=['steelblue', 'coral', 'forestgreen'])
ax3.set_title("Importe Total por Canal", fontsize=14, fontweight='bold')
ax3.set_xlabel("Canal")
ax3.set_ylabel("Importe Total (€)")
ax3.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("C:/Users/PC/Big_Data_2026/entregables/bloque_01_python_pandas_eda/graficos/ventas_por_canal.png", dpi=150)
plt.close()
print("Gráfico 3: ventas_por_canal.png")

# Gráfico 4: Importe por región
fig4, ax4 = plt.subplots(figsize=(10, 5))
importe_por_region.plot(kind="bar", ax=ax4, color='steelblue')
ax4.set_title("Importe Total por Región", fontsize=14, fontweight='bold')
ax4.set_xlabel("Región")
ax4.set_ylabel("Importe Total (€)")
ax4.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("C:/Users/PC/Big_Data_2026/entregables/bloque_01_python_pandas_eda/graficos/ventas_por_region.png", dpi=150)
plt.close()
print("Gráfico 4: ventas_por_region.png")

# ============================================
# EJERCICIO INTEGRADOR
# ============================================
print("\n" + "=" * 60)
print("EJERCICIO INTEGRADOR")
print("=" * 60)

# 1. Importe total por canal
print("\n1. IMPORTE TOTAL POR CANAL:")
print(importe_por_canal.to_string())

# 2. Ticket medio por región
print("\n2. TICKET MEDIO POR REGIÓN:")
print(ticket_medio_region.to_string())

# 3. Canal con mayor número de operaciones
print("\n3. NÚMERO DE OPERACIONES POR CANAL:")
operaciones_canal = df_limpio.groupby("canal")["cliente_id"].count().sort_values(ascending=False)
print(operaciones_canal.to_string())
canal_mas_operaciones = operaciones_canal.index[0]
print(f"\nCanal con mayor número de operaciones: {canal_mas_operaciones}")

# 4. Gráfico de barras con ventas por canal (ya generado)

# 5. Conclusiones de negocio
print("\n" + "=" * 60)
print("CONCLUSIONES DE NEGOCIO")
print("=" * 60)

canal_top = importe_por_canal.index[0]
region_top = importe_por_region.index[0]
ticket_region_top = ticket_medio_region.index[0]
importe_medio_total = df_limpio["importe"].mean()

conclusion1 = f"El canal {canal_top} es el que genera mayores ventas totales ({importe_por_canal.iloc[0]:,.2f} €), representando la principal vía de facturación."
conclusion2 = f"La región {region_top} lidera en volumen de ventas ({importe_por_region.iloc[0]:,.2f} €), indicando mayor penetración de mercado o eficiencia comercial en esa zona."
conclusion3 = f"El ticket medio general es de {importe_medio_total:,.2f} €, con la región {ticket_region_top} presentando el mayor ticket medio ({ticket_medio_region.iloc[0]:,.2f} €)."

print(f"\nConclusión 1: {conclusion1}")
print(f"\nConclusión 2: {conclusion2}")
print(f"\nConclusión 3: {conclusion3}")

# Guardar resultados clave para el resumen
resultados = {
    "importe_total": float(df_limpio["importe"].sum()),
    "importe_medio": float(df_limpio["importe"].mean()),
    "importe_mediana": float(df_limpio["importe"].median()),
    "num_registros": len(df_limpio),
    "num_canales": df_limpio["canal"].nunique(),
    "canales": df_limpio["canal"].unique().tolist(),
    "num_regiones": df_limpio["region"].nunique(),
    "regiones": [r for r in df_limpio["region"].unique().tolist() if isinstance(r, str)],
    "canal_top": canal_top,
    "importe_canal_top": float(importe_por_canal.iloc[0]),
    "region_top": region_top,
    "importe_region_top": float(importe_por_region.iloc[0]),
    "ticket_medio_region_top": float(ticket_medio_region.iloc[0]),
    "conclusiones": [conclusion1, conclusion2, conclusion3]
}

with open("C:/Users/PC/Big_Data_2026/entregables/bloque_01_python_pandas_eda/resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60)
print(f"Resultados guardados en: resultados.json")
print(f"\n[*] CONCLUSIÓN PRINCIPAL: El canal {canal_top} es el segmento principal con {importe_por_canal.iloc[0]:,.2f} € en ventas totales.")
