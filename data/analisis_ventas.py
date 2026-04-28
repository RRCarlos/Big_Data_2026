import pandas as pd
import numpy as np

# 1. Cargar el dataset
df = pd.read_csv('ventas_mayo_2026.csv')

print('='*60)
print('ANALISIS DEL DATASET: ventas_mayo_2026.csv')
print('='*60)

# 2. Informacion basica del dataset
print('\n1. INFORMACION BASICA DEL DATASET')
print('-'*60)
print(f'Numero de filas: {len(df)}')
print(f'Numero de columnas: {len(df.columns)}')
print(f'\nColumnas: {list(df.columns)}')
print(f'\nTipos de datos:')
print(df.dtypes)

print('\n' + '='*60)
print('2. ANALISIS DESCRIPTIVO')
print('='*60)
print('\nPrimeras 5 filas:')
print(df.head())

print('\nUltimas 5 filas:')
print(df.tail())

print('\nEstadisticas descriptivas:')
print(df.describe())

print('\nValores nulos por columna:')
print(df.isnull().sum())

print('\n' + '='*60)
print('3. MODULO I: ANALISIS POR CANAL (EDA/Segmentacion)')
print('='*60)

# Ventas totales por canal
ventas_canal = df.groupby('canal')['importe'].sum().sort_values(ascending=False)
print('\nIMPORTE TOTAL POR CANAL:')
print(ventas_canal)
print(f'\nCanal con MAYORES ventas: {ventas_canal.index[0]} con {ventas_canal.iloc[0]:,.2f}')
print(f'Porcentaje del total: {(ventas_canal.iloc[0]/ventas_canal.sum()*100):.2f}%')

print('\nUNIDADES VENDIDAS POR CANAL:')
unidades_canal = df.groupby('canal')['unidades'].sum().sort_values(ascending=False)
print(unidades_canal)
print(f'\nCanal con MAYORES unidades: {unidades_canal.index[0]} con {unidades_canal.iloc[0]:,} unidades')

print('\nNUMERO DE TRANSACCIONES POR CANAL:')
transacciones_canal = df.groupby('canal').size().sort_values(ascending=False)
print(transacciones_canal)

print('\nTicket promedio por canal:')
ticket_canal = df.groupby('canal').apply(lambda x: x['importe'].sum() / len(x)).sort_values(ascending=False)
print(ticket_canal)

print('\n' + '='*60)
print('4. MODULO II: ANALISIS POR REGION (Regresion/Publico objetivo)')
print('='*60)

# Ventas totales por region
ventas_region = df.groupby('region')['importe'].sum().sort_values(ascending=False)
print('\nIMPORTE TOTAL POR REGION:')
print(ventas_region)
print(f'\nRegion con MAYOR importe: {ventas_region.index[0]} con {ventas_region.iloc[0]:,.2f}')
print(f'Porcentaje del total: {(ventas_region.iloc[0]/ventas_region.sum()*100):.2f}%')

print('\nUNIDADES VENDIDAS POR REGION:')
unidades_region = df.groupby('region')['unidades'].sum().sort_values(ascending=False)
print(unidades_region)

print('\nNUMERO DE TRANSACCIONES POR REGION:')
transacciones_region = df.groupby('region').size().sort_values(ascending=False)
print(transacciones_region)

print('\nTicket promedio por region:')
ticket_region = df.groupby('region').apply(lambda x: x['importe'].sum() / len(x)).sort_values(ascending=False)
print(ticket_region)

print('\n' + '='*60)
print('5. ANALISIS CRUZADO: CANAL vs REGION')
print('='*60)

# Tabla pivote: Canal x Region
pivot_ventas = pd.pivot_table(df, values='importe', index='canal', columns='region', aggfunc='sum', fill_value=0)
print('\nIMPORTE TOTAL (Canal x Region):')
print(pivot_ventas)

print('\nPorcentajes por region dentro de cada canal:')
print(pivot_ventas.div(pivot_ventas.sum(axis=1), axis=0).multiply(100).round(2))

print('\n' + '='*60)
print('6. ANALISIS ADICIONAL: CATEGORIAS')
print('='*60)

ventas_categoria = df.groupby('categoria')['importe'].sum().sort_values(ascending=False)
print('\nIMPORTE TOTAL POR CATEGORIA:')
print(ventas_categoria)

print('\n' + '='*60)
print('7. VERIFICACION DE EXPECTATIVAS DEL PROFESOR')
print('='*60)

print('\nMODULO I - Es "Online" el canal con mayores ventas?')
if ventas_canal.index[0] == 'Online':
    print('CORRECTO: Online es efectivamente el canal con mayores ventas')
    print(f'   Importe: {ventas_canal.iloc[0]:,.2f}')
else:
    print('INCORRECTO: Online NO es el canal con mayores ventas')
    print(f'   El canal con mayores ventas es: {ventas_canal.index[0]}')
    print(f'   Online esta en posicion: {list(ventas_canal.index).index("Online") + 1}')
    print(f'   Importe de Online: {ventas_canal["Online"]:,.2f}')

print('\nMODULO II - Es "Madrid" la region con mayor importe?')
if ventas_region.index[0] == 'Madrid':
    print('CORRECTO: Madrid es efectivamente la region con mayor importe')
    print(f'   Importe: {ventas_region.iloc[0]:,.2f}')
else:
    print('INCORRECTO: Madrid NO es la region con mayor importe')
    print(f'   La region con mayor importe es: {ventas_region.index[0]}')
    posicion_madrid = list(ventas_region.index).index('Madrid') + 1 if 'Madrid' in ventas_region.index else 'No encontrada'
    print(f'   Madrid esta en posicion: {posicion_madrid}')
    if 'Madrid' in ventas_region.index:
        print(f'   Importe de Madrid: {ventas_region["Madrid"]:,.2f}')

print('\n' + '='*60)
print('8. RESUMEN DE METRICAS EXACTAS')
print('='*60)

print('\nIMPORTES TOTALES POR CANAL:')
for canal, importe in ventas_canal.items():
    print(f'  {canal}: {importe:,.2f} ({(importe/ventas_canal.sum()*100):.2f}%)')

print('\nIMPORTES TOTALES POR REGION:')
for region, importe in ventas_region.items():
    print(f'  {region}: {importe:,.2f} ({(importe/ventas_region.sum()*100):.2f}%)')

# Guardar resultados en CSV para referencia
print('\n' + '='*60)
print('9. GUARDANDO RESULTADOS')
print('='*60)

ventas_canal.to_csv('resultados_ventas_canal.csv')
ventas_region.to_csv('resultados_ventas_region.csv')
print('Archivos guardados: resultados_ventas_canal.csv y resultados_ventas_region.csv')

print('\n' + '='*60)
print('ANALISIS COMPLETADO')
print('='*60)
