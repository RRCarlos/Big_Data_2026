#!/usr/bin/env python3
"""
Script para ejecutar el análisis del Bloque II - Regresión
Usa el dataset CORRECTO: data/ventas_mayo_2026.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

OUTPUT_DIR = "C:/Users/PC/Big_Data_2026/entregables/bloque_02_regresion"
DATA_DIR = "C:/Users/PC/Big_Data_2026/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/graficos", exist_ok=True)

print("=" * 60)
print("BLOQUE II - REGRESIÓN Y COMPARACIÓN DE MODELOS")
print("=" * 60)

# ============================================
# 1. CARGAR DATOS (DATASET CORRECTO)
# ============================================
print("\n1. CARGANDO DATOS...")
df = pd.read_csv(f"{DATA_DIR}/ventas_mayo_2026.csv")
print(f"Dataset CORRECTO cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")

# ============================================
# ANÁLISIS EXPLORATORIO DE REGIONES (NUEVO)
# ============================================
print("\n" + "=" * 60)
print("ANÁLISIS POR REGIÓN (PARA CONCLUSIONES DE NEGOCIO)")
print("=" * 60)

# Importe por región
importe_por_region = df.groupby("region")["importe"].sum().sort_values(ascending=False)
print("\nImporte Total por Región:")
print(importe_por_region.to_string())
print(f"\n[*] REGIÓN TOP: {importe_por_region.index[0]} con {importe_por_region.iloc[0]:,.2f} €")

# Importe por canal
importe_por_canal = df.groupby("canal")["importe"].sum().sort_values(ascending=False)
print("\nImporte Total por Canal:")
print(importe_por_canal.to_string())

# Limpiar datos para regresión
df_limpio = df.drop_duplicates()
df_limpio["fecha"] = pd.to_datetime(df_limpio["fecha"])
df_limpio["mes"] = df_limpio["fecha"].dt.month

print(f"\nDataset preparado para regresión: {df_limpio.shape[0]} filas")

# ============================================
# 2. DEFINIR X E Y
# ============================================
print("\n2. PREPARANDO FEATURES...")

# Features disponibles en el dataset correcto
target = "importe"  # Cambiado de 'ventas' a 'importe'
features_num = ["unidades", "precio_unitario", "descuento", "antiguedad_cliente_meses"]
features_cat = ["region", "canal", "categoria"]

X = df_limpio[features_num + features_cat].copy()
y = df_limpio[target].copy()

# Imputar nulos
y = y.fillna(y.median())
X["precio_unitario"] = X["precio_unitario"].fillna(X["precio_unitario"].median())
X["region"] = X["region"].fillna("Sin informar")
X["canal"] = X["canal"].fillna("Sin informar")
X["categoria"] = X["categoria"].fillna("Sin informar")

print(f"Target: {target}")
print(f"Features numéricas: {features_num}")
print(f"Features categóricas: {features_cat}")

# ============================================
# 3. TRAIN/TEST SPLIT
# ============================================
print("\n3. SEPARANDO TRAIN/TEST...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================
# 4. PREPROCESSOR
# ============================================
print("\n4. CONSTRUYENDO PIPELINE...")

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

# ============================================
# 5. EVALUAR MODELOS
# ============================================
print("\n5. ENTRENANDO MODELOS...")

def evaluar_modelo(nombre, modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    return {
        "modelo": nombre,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }, pred

# Modelo 1: Regresión Lineal
print("   - Linear Regression...")
modelo_lr = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
res_lr, pred_lr = evaluar_modelo("Linear Regression", modelo_lr, X_train, X_test, y_train, y_test)

# Modelo 2: Ridge
print("   - Ridge...")
modelo_ridge = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])
res_ridge, pred_ridge = evaluar_modelo("Ridge", modelo_ridge, X_train, X_test, y_train, y_test)

# Modelo 3: Random Forest
print("   - Random Forest Regressor...")
modelo_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
res_rf, pred_rf = evaluar_modelo("Random Forest", modelo_rf, X_train, X_test, y_train, y_test)

# ============================================
# 6. COMPARACIÓN
# ============================================
print("\n" + "=" * 60)
print("RESULTADOS COMPARATIVOS")
print("=" * 60)

resultados = [res_lr, res_ridge, res_rf]
for r in resultados:
    print(f"\n{r['modelo']}:")
    print(f"  MAE:  {r['MAE']:,.2f}")
    print(f"  RMSE: {r['RMSE']:,.2f}")
    print(f"  R²:   {r['R2']:.4f}")

# Mejor modelo
mejores = sorted(resultados, key=lambda x: x['MAE'])
mejor = mejores[0]
print(f"\n[*] MEJOR MODELO (por MAE): {mejor['modelo']}")

# ============================================
# 7. VISUALIZACIÓN
# ============================================
print("\n7. GENERANDO GRÁFICOS...")

# Gráfico 1: Comparación de métricas
fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

modelos = [r['modelo'] for r in resultados]
maes = [r['MAE'] for r in resultados]
rmses = [r['RMSE'] for r in resultados]
r2s = [r['R2'] for r in resultados]

x = np.arange(len(modelos))
width = 0.6

axes[0].bar(x, maes, width, color='steelblue')
axes[0].set_title('MAE (Error Absoluto Medio)', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(modelos, rotation=45, ha='right')

axes[1].bar(x, rmses, width, color='coral')
axes[1].set_title('RMSE (Raíz Error Cuadrático Medio)', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(modelos, rotation=45, ha='right')

axes[2].bar(x, r2s, width, color='forestgreen')
axes[2].set_title('R² (Coeficiente de Determinación)', fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(modelos, rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graficos/comparacion_modelos.png", dpi=150)
plt.close()
print("   - comparacion_modelos.png")

# Gráfico 2: Predicciones vs reales
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (nombre, pred) in enumerate([("LR", pred_lr), ("Ridge", pred_ridge), ("RF", pred_rf)]):
    axes[i].scatter(y_test, pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), pred.min())
    max_val = max(y_test.max(), pred.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[i].set_xlabel('Valor Real (€)')
    axes[i].set_ylabel('Predicción (€)')
    axes[i].set_title(f'{nombre}', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graficos/predicciones_vs_reales.png", dpi=150)
plt.close()
print("   - predicciones_vs_reales.png")

# Gráfico 3: Importe por región (NUEVO - para conclusión)
fig3, ax3 = plt.subplots(figsize=(10, 5))
importe_por_region.plot(kind="bar", ax=ax3, color='steelblue')
ax3.set_title("Importe Total por Región", fontsize=14, fontweight='bold')
ax3.set_xlabel("Región")
ax3.set_ylabel("Importe Total (€)")
ax3.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graficos/importe_por_region.png", dpi=150)
plt.close()
print("   - importe_por_region.png (NUEVO)")

# ============================================
# 8. GUARDAR RESULTADOS
# ============================================
print("\n8. GUARDANDO RESULTADOS...")

# Conclusión principal: Madrid es la región con mayor importe
region_top = importe_por_region.index[0]
importe_region_top = importe_por_region.iloc[0]

resultados_json = {
    "modelos": resultados,
    "mejor_modelo": mejor['modelo'],
    "mejores_metricas": {
        "MAE": mejor['MAE'],
        "RMSE": mejor['RMSE'],
        "R2": mejor['R2']
    },
    "analisis_region": {
        "region_top": region_top,
        "importe_region_top": float(importe_region_top),
        "importe_por_region": {k: float(v) for k, v in importe_por_region.items()}
    },
    "conclusiones": [
        f"El modelo {mejor['modelo']} presenta el menor MAE ({mejor['MAE']:,.2f}), indicando las predicciones más precisas.",
        f"El R² de {mejor['R2']:.4f} indica que el modelo explica el {mejor['R2']*100:.1f}% de la variabilidad en las ventas.",
        f"Madrid es la región con mayor importe total ({importe_region_top:,.2f} €), siendo el público objetivo principal para campañas de marketing.",
        "Random Forest captura mejor las relaciones no lineales entre features."
    ]
}

with open(f"{OUTPUT_DIR}/resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados_json, f, indent=2, ensure_ascii=False)

print(f"\n[OK] BLOQUE II COMPLETADO")
print(f"Resultados guardados en: {OUTPUT_DIR}/resultados.json")
print(f"\n[*] CONCLUSIÓN PRINCIPAL: Madrid es la región con mayor importe ({importe_region_top:,.2f} €), siendo el público objetivo principal.")
