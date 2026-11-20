import duckdb
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# 1. Conectarnos a nuestra base de datos
conexion = duckdb.connect(database="agricola.duckdb")

# 2. Consultar todos los registros de la tabla parcelas
df = conexion.execute("""
    SELECT * FROM parcelas
    ORDER BY parcela_id
""").df()

# 3. Mostrar los datos utilizando Pandas
print("DATOS DE LAS PARCELAS")
print(df.head(10))

# 4. Mostrar información general
print("\nINFORMACION GENERAL")
print(df.info())


# 5. Mostrar estadísticas descriptivas
print("\nESTADISTICAS DESCRIPTIVAS")
print(df.describe())

# 6. Calcular la matriz de correlación de Pearson
correlaciones = df[
    ["volumen_riego", "fertilizante_kg", "rendimiento_ton"]
].corr(method="pearson")

print("\nMATRIZ DE CORRELACION DE PEARSON")
print(correlaciones.round(4))

# 7. Identificar las correlaciones con el rendimiento
print("\nCORRELACION CON EL RENDIMIENTO")

print(
    "Riego:",
    correlaciones.loc["volumen_riego", "rendimiento_ton"]
)

print(
    "Fertilizante:",
    correlaciones.loc["fertilizante_kg", "rendimiento_ton"]
)


# 8. Construir el modelo de regresion lineal multiple OLS

# Variable dependiente (Y): rendimiento
Y = df["rendimiento_ton"]

# Variables independientes (X): riego y fertilizante
X = df[["volumen_riego", "fertilizante_kg"]]

# Agregar la constante o intercepto
X = sm.add_constant(X)

# Ajustar el modelo de regresion
modelo = sm.OLS(Y, X).fit()

# Mostrar el resumen estadistico
print("\nRESULTADOS DEL MODELO DE REGRESION OLS")
print(modelo.summary())


# 9. PRUEBA DE HIPOTESIS: TRATAMIENTO A VS B

# Separar los rendimientos de cada tratamiento
grupo_A = df[
    df["tipo_tratamiento"] == "Tratamiento_A"
]["rendimiento_ton"]

grupo_B = df[
    df["tipo_tratamiento"] == "Tratamiento_B"
]["rendimiento_ton"]

# Calcular los promedios
media_A = grupo_A.mean()
media_B = grupo_B.mean()

print("\nCOMPARACION DE TRATAMIENTOS")
print("Cantidad de parcelas A:", len(grupo_A))
print("Cantidad de parcelas B:", len(grupo_B))
print("Rendimiento promedio A:", round(media_A, 4))
print("Rendimiento promedio B:", round(media_B, 4))

# Ejecutar prueba t-Student de dos muestras independientes
resultado_t = stats.ttest_ind(
    grupo_B,
    grupo_A,
    equal_var=True,
    alternative="greater"
)

print("\nRESULTADOS DE LA PRUEBA T-STUDENT")
print("Estadistico t:", resultado_t.statistic)
print("P-valor:", resultado_t.pvalue)

# Calcular diferencia de medias
diferencia = media_B - media_A

# Calcular intervalo de confianza del 95%
n_A = len(grupo_A)
n_B = len(grupo_B)

var_A = grupo_A.var(ddof=1)
var_B = grupo_B.var(ddof=1)

grados_libertad = n_A + n_B - 2

varianza_combinada = (
    ((n_A - 1) * var_A + (n_B - 1) * var_B)
    / grados_libertad
)

error_estandar = (
    varianza_combinada * (1/n_A + 1/n_B)
) ** 0.5

valor_critico = stats.t.ppf(
    0.975,
    grados_libertad
)

limite_inferior = diferencia - valor_critico * error_estandar
limite_superior = diferencia + valor_critico * error_estandar

print("\nINTERVALO DE CONFIANZA DEL 95%")
print("Limite inferior:", round(limite_inferior, 4))
print("Limite superior:", round(limite_superior, 4))

# Emitir veredicto inferencial
alpha = 0.05

print("\nVEREDICTO INFERENCIAL")

if resultado_t.pvalue < alpha:
    print("Se rechaza la hipotesis nula (H0).")
    print("El Tratamiento B incrementa significativamente")
    print("el rendimiento respecto al Tratamiento A.")
else:
    print("No se puede rechazar la hipotesis nula (H0).")
    print("No existe evidencia suficiente para afirmar")
    print("que el Tratamiento B mejora el rendimiento.")


# 10. Cerrar la conexion
conexion.close()
