# Bloque V — Series Temporales y Forecasting

## Objetivos
Trabajar con series temporales, crear features, aplicar forecasting.

## Conceptos Clave
Serie temporal, Lag, Ventana móvil, Baseline naive, Forecasting supervisado.


## Resultados
| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 4.00 | 5.04 | 0.71 |
| Naive | 14.02 | 16.74 | -2.16 |

Predicción a 30 días: ~138.77

## Limitaciones
- Serie corta (180 días)
- Sin variables externas

## Conclusión
Modelo RF supera significativamente al baseline.