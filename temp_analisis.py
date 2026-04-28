#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Cargar el dataset CORRECTO
df = pd.read_csv('data/ventas_mayo_2026.csv')

print('=== ANÁLISIS DEL DATASET CORRECTO ===')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

print('\nValores únicos en canal:')
ventas_canal = df.groupby('canal')['ventas'].sum().sort_values(ascending=False)
print(ventas_canal)
print(f"\nCanal top: {ventas_canal.index[0]} con {ventas_canal.iloc[0]:,.2f} €")

print('\nValores únicos en región:')
ventas_region = df.groupby('region')['ventas'].sum().sort_values(ascending=False)
print(ventas_region)
print(f"\nRegión top: {ventas_region.index[0]} con {ventas_region.iloc[0]:,.2f} €")

print('\n=== ESTADÍSTICAS GENERALES ===')
print(f"Ventas totales: {df['ventas'].sum():,.2f} €")
print(f"Ventas medias: {df['ventas'].mean():,.2f} €")
print(f"Ventas mediana: {df['ventas'].median():,.2f} €")
print(f"Número de registros: {len(df)}")
print(f"Número de canales: {df['canal'].nunique()}")
print(f"Número de regiones: {df['region'].nunique()}")

print('\n=== TICKET MEDIO POR REGIÓN ===')
ticket_medio = df.groupby('region')['ventas'].mean().sort_values(ascending=False)
print(ticket_medio)

print('\n=== ANÁLISIS COMPLETO POR CANAL ===')
analisis_canal = df.groupby('canal').agg({
    'ventas': ['sum', 'mean', 'count'],
    'clientes': 'sum'
}).round(2)
print(analisis_canal)
