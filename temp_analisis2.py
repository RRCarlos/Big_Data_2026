#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Cargar el dataset CORRECTO
df = pd.read_csv('data/ventas_mayo_2026.csv')

print('=== ANÁLISIS DEL DATASET CORRECTO (ventas_mayo_2026.csv) ===')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Info:')
df.info()
print(f'\nPrimeras 5 filas:')
print(df.head().to_string())

print('\n=== ANÁLISIS POR CANAL (usando importe) ===')
ventas_canal = df.groupby('canal')['importe'].sum().sort_values(ascending=False)
print(ventas_canal)
print(f"\nCanal top: {ventas_canal.index[0]} con {ventas_canal.iloc[0]:,.2f} €")

print('\n=== ANÁLISIS POR REGIÓN (usando importe) ===')
ventas_region = df.groupby('region')['importe'].sum().sort_values(ascending=False)
print(ventas_region)
print(f"\nRegión top: {ventas_region.index[0]} con {ventas_region.iloc[0]:,.2f} €")

print('\n=== ESTADÍSTICAS GENERALES ===')
print(f"Importe total: {df['importe'].sum():,.2f} €")
print(f"Importe medio: {df['importe'].mean():,.2f} €")
print(f"Importe mediana: {df['importe'].median():,.2f} €")
print(f"Número de registros: {len(df)}")
print(f"Número de canales: {df['canal'].nunique()}")
print(f"Número de regiones: {df['region'].nunique()}")

print('\n=== TICKET MEDIO POR REGIÓN ===')
ticket_medio = df.groupby('region')['importe'].mean().sort_values(ascending=False)
print(ticket_medio)

print('\n=== ANÁLISIS COMPLETO POR CANAL ===')
analisis_canal = df.groupby('canal').agg({
    'importe': ['sum', 'mean', 'count'],
    'cliente_id': 'nunique'
}).round(2)
print(analisis_canal)

print('\n=== VALORES ÚNICOS ===')
print(f"Canales: {df['canal'].unique()}")
print(f"Regiones: {df['region'].unique()}")
print(f"Categorías: {df['categoria'].unique()}")
