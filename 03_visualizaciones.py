import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

# 1. Conectarse a la base de datos
conexion = duckdb.connect(database="agricola.duckdb")

# 2. Extraer los datos
df = conexion.execute("""
    SELECT * FROM parcelas
""").df()

# 3. Calcular la matriz de correlacion
correlaciones = df[
    ["volumen_riego", "fertilizante_kg", "rendimiento_ton"]
].corr(method="pearson")

# 4. Crear el mapa de calor
plt.figure(figsize=(9, 6))

sns.heatmap(
    correlaciones,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

# 5. Agregar titulo
plt.title(
    "Matriz de Correlacion de Pearson",
    fontsize=14
)

plt.tight_layout()

# 6. Guardar la grafica
plt.savefig("grafica_1_heatmap.png", dpi=300)

# 7. Mostrar la grafica
plt.show()

# GRAFICA 2: DIAGRAMA DE DISPERSION CON REGRESION OLS

# Preparar las variables del modelo
Y = df["rendimiento_ton"]

X = df[["volumen_riego", "fertilizante_kg"]]
X = sm.add_constant(X)

# Ajustar el modelo de regresion
modelo = sm.OLS(Y, X).fit()

# Crear valores de riego para dibujar la linea
riego_nuevo = np.linspace(
    df["volumen_riego"].min(),
    df["volumen_riego"].max(),
    100
)

# Mantener el fertilizante en su valor promedio
datos_prediccion = pd.DataFrame({
    "const": 1.0,
    "volumen_riego": riego_nuevo,
    "fertilizante_kg": df["fertilizante_kg"].mean()
})

# Obtener predicciones e intervalo de confianza
predicciones = modelo.get_prediction(datos_prediccion).summary_frame()

# Crear la grafica
plt.figure(figsize=(9, 6))

# Mostrar las parcelas como puntos
plt.scatter(
    df["volumen_riego"],
    df["rendimiento_ton"],
    alpha=0.7,
    label="Parcelas"
)

# Dibujar la recta de regresion
plt.plot(
    riego_nuevo,
    predicciones["mean"],
    color="red",
    label="Regresion OLS"
)

# Agregar la banda de confianza del 95%
plt.fill_between(
    riego_nuevo,
    predicciones["mean_ci_lower"],
    predicciones["mean_ci_upper"],
    color="red",
    alpha=0.2,
    label="Intervalo de confianza 95%"
)

plt.title("Volumen de Riego vs Rendimiento")
plt.xlabel("Volumen de Riego (m³/ha)")
plt.ylabel("Rendimiento (ton/ha)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("grafica_2_dispersion.png", dpi=300)
plt.show()

# GRAFICA 3: DIAGRAMA DE CAJAS COMPARATIVO

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="tipo_tratamiento",
    y="rendimiento_ton",
    hue="tipo_tratamiento",
    palette="Set2",
    legend=False
)

plt.title("Comparacion del Rendimiento por Tratamiento")
plt.xlabel("Tipo de Tratamiento")
plt.ylabel("Rendimiento (ton/ha)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("grafica_3_boxplot.png", dpi=300)
plt.show()

# GRAFICA 4: HISTOGRAMA Y DENSIDAD DE RESIDUOS

# Calcular los residuos del modelo
residuos = modelo.resid

# Crear la grafica
plt.figure(figsize=(9, 6))

sns.histplot(
    residuos,
    kde=True,
    stat="density",
    bins=15,
    color="skyblue",
    edgecolor="black"
)

# Marcar el punto donde el residuo es cero
plt.axvline(
    x=0,
    color="red",
    linestyle="--",
    label="Residuo cero"
)

plt.title("Distribucion de los Residuos del Modelo OLS")
plt.xlabel("Residuos (ton/ha)")
plt.ylabel("Densidad")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("grafica_4_residuos.png", dpi=300)
plt.show()

# 8. Cerrar conexion
conexion.close()