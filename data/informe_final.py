import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv('ventas_mayo_2026.csv')

print('='*70)
print('INFORME DETALLADO: VALIDACION DE CONCLUSIONES DEL PROFESOR')
print('='*70)

# Limpiar datos (filas con region nula)
df_limpio = df.dropna(subset=['region'])
print(f'\nRegistros totales: {len(df)}')
print(f'Registros con region nula: {len(df) - len(df_limpio)}')
print(f'Registros para analisis: {len(df_limpio)}')

print('\n' + '='*70)
print('MODULO I: SEGMENTACION POR CANAL (EDA)')
print('='*70)

# Analisis por canal
ventas_canal = df_limpio.groupby('canal')['importe'].sum().sort_values(ascending=False)
unidades_canal = df_limpio.groupby('canal')['unidades'].sum().sort_values(ascending=False)
transacciones_canal = df_limpio.groupby('canal').size().sort_values(ascending=False)

print('\nTABLA COMPARATIVA POR CANAL:')
print('-'*70)
print(f'{"Canal":<15} {"Importe Total":>15} {"% del Total":>10} {"Unidades":>10} {"Transacciones":>14} {"Ticket Promedio":>15}')
print('-'*70)

total_importe = ventas_canal.sum()
for canal in ventas_canal.index:
    importe = ventas_canal[canal]
    pct = (importe / total_importe) * 100
    unid = unidades_canal[canal]
    trans = transacciones_canal[canal]
    ticket = importe / trans
    print(f'{canal:<15} {importe:>15,.2f} {pct:>9.2f}% {unid:>10,} {trans:>14} {ticket:>15,.2f}')

print('\n' + '='*70)
print('MODULO II: PUBLICO OBJETIVO POR REGION (REGRESION)')
print('='*70)

# Analisis por region
ventas_region = df_limpio.groupby('region')['importe'].sum().sort_values(ascending=False)
unidades_region = df_limpio.groupby('region')['unidades'].sum().sort_values(ascending=False)
transacciones_region = df_limpio.groupby('region').size().sort_values(ascending=False)

print('\nTABLA COMPARATIVA POR REGION:')
print('-'*70)
print(f'{"Region":<20} {"Importe Total":>15} {"% del Total":>10} {"Unidades":>10} {"Transacciones":>14} {"Ticket Promedio":>15}')
print('-'*70)

for region in ventas_region.index:
    importe = ventas_region[region]
    pct = (importe / total_importe) * 100
    unid = unidades_region[region]
    trans = transacciones_region[region]
    ticket = importe / trans
    print(f'{region:<20} {importe:>15,.2f} {pct:>9.2f}% {unid:>10,} {trans:>14} {ticket:>15,.2f}')

print('\n' + '='*70)
print('ANALISIS DE VARIANZA POR REGION')
print('='*70)

# Varianza de importes por region
for region in df_limpio['region'].unique():
    datos_region = df_limpio[df_limpio['region'] == region]['importe']
    print(f'{region:<20} - Media: {datos_region.mean():>8.2f}, Desv.Est: {datos_region.std():>8.2f}, Coef.Var: {(datos_region.std()/datos_region.mean())*100:>6.2f}%')

print('\n' + '='*70)
print('CONFIRMACION DE EXPECTATIVAS DEL PROFESOR')
print('='*70)

print('\nMODULO I - Conclusion del profesor: "Online es el canal principal"')
print('-'*50)
print(f'Ranking de canales por importe:')
for i, (canal, importe) in enumerate(ventas_canal.items(), 1):
    print(f'  {i}. {canal:<15} -> {importe:>10,.2f} ({(importe/total_importe)*100:.2f}%)')
    
if ventas_canal.index[0] == 'Online':
    print('\n[VALIDADO] Online es efectivamente el canal #1 en importe total')
    print(f'   Diferencia con el segundo (Tienda): {ventas_canal.iloc[0] - ventas_canal.iloc[1]:,.2f}')
    print(f'   Superioridad: {(ventas_canal.iloc[0]/ventas_canal.iloc[1] - 1)*100:.2f}% mas que Tienda')
else:
    print('\n[NO VALIDADO]')

print('\nMODULO II - Conclusion del profesor: "Madrid es el publico objetivo"')
print('-'*50)
print(f'Ranking de regiones por importe:')
for i, (region, importe) in enumerate(ventas_region.items(), 1):
    print(f'  {i}. {region:<20} -> {importe:>10,.2f} ({(importe/total_importe)*100:.2f}%)')

if ventas_region.index[0] == 'Madrid':
    print('\n[VALIDADO] Madrid es efectivamente la region #1 en importe total')
    print(f'   Diferencia con el segundo (Valencia): {ventas_region.iloc[0] - ventas_region.iloc[1]:,.2f}')
    print(f'   Superioridad: {(ventas_region.iloc[0]/ventas_region.iloc[1] - 1)*100:.2f}% mas que Valencia')
else:
    print('\n[NO VALIDADO]')

print('\n' + '='*70)
print('ANALISIS ADICIONAL: CORRELACIONES')
print('='*70)

# Matriz de correlacion
print('\nMatriz de correlacion (variables numericas):')
numeric_cols = ['unidades', 'precio_unitario', 'descuento', 'antiguedad_cliente_meses', 'importe']
corr_matrix = df_limpio[numeric_cols].corr()
print(corr_matrix.round(3))

print('\nCorrelacion mas fuerte con IMPORTE:')
corr_importe = corr_matrix['importe'].drop('importe').sort_values(key=abs, ascending=False)
for var, corr in corr_importe.items():
    print(f'  {var:<30}: {corr:>7.3f}')

print('\n' + '='*70)
print('METRICAS EXACTAS PARA EL INFORME')
print('='*70)
print('\n1. MODULO I (EDA/Segmentacion):')
print(f'   - Canal ganador: {ventas_canal.index[0]}')
print(f'   - Importe total: {ventas_canal.iloc[0]:,.2f}')
print(f'   - Porcentaje del total: {(ventas_canal.iloc[0]/total_importe)*100:.2f}%')
print(f'   - Unidades: {unidades_canal.iloc[0]:,}')
print(f'   - Transacciones: {transacciones_canal.iloc[0]}')
print(f'   - Ticket promedio: {ventas_canal.iloc[0]/transacciones_canal.iloc[0]:,.2f}')
print(f'   - ¿Coincide con el profesor? {"SI" if ventas_canal.index[0] == "Online" else "NO"}')

print('\n2. MODULO II (Regresion/Publico objetivo):')
print(f'   - Region ganadora: {ventas_region.index[0]}')
print(f'   - Importe total: {ventas_region.iloc[0]:,.2f}')
print(f'   - Porcentaje del total: {(ventas_region.iloc[0]/total_importe)*100:.2f}%')
print(f'   - Unidades: {unidades_region.iloc[0]:,}')
print(f'   - Transacciones: {transacciones_region.iloc[0]}')
print(f'   - Ticket promedio: {ventas_region.iloc[0]/transacciones_region.iloc[0]:,.2f}')
print(f'   - ¿Coincide con el profesor? {"SI" if ventas_region.index[0] == "Madrid" else "NO"}')

print('\n' + '='*70)
print('RESUMEN DE IMPORTES TOTALES (PARA GRAFICOS)')
print('='*70)
print('\nIMPORTES POR CANAL:')
for canal, importe in ventas_canal.items():
    print(f'  {canal}: {importe:,.2f}')

print('\nIMPORTES POR REGION:')
for region, importe in ventas_region.items():
    print(f'  {region}: {importe:,.2f}')

print('\n' + '='*70)
print('INFORME COMPLETADO')
print('='*70)
