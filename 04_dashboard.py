import streamlit as st
import duckdb
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuracion de la pagina
st.set_page_config(
    page_title="Dashboard Agricola",
    page_icon="🌱",
    layout="wide"
)

# Titulo principal
st.title("🌱 Dashboard de Rendimiento Agricola")
st.subheader("Analisis de rendimiento y eficiencia de riego")

# Conexion con la base de datos
conexion = duckdb.connect(database="agricola.duckdb")

df = conexion.execute("""
    SELECT * FROM parcelas
    ORDER BY parcela_id
""").df()

conexion.close()

# Modelo de regresion OLS
Y = df["rendimiento_ton"]

X = df[["volumen_riego", "fertilizante_kg"]]
X = sm.add_constant(X)

modelo = sm.OLS(Y, X).fit()

# Prueba de hipotesis
grupo_A = df[
    df["tipo_tratamiento"] == "Tratamiento_A"
]["rendimiento_ton"]

grupo_B = df[
    df["tipo_tratamiento"] == "Tratamiento_B"
]["rendimiento_ton"]

resultado_t = stats.ttest_ind(
    grupo_B,
    grupo_A,
    equal_var=True,
    alternative="greater"
)

# Calcular indicadores
rendimiento_promedio = df["rendimiento_ton"].mean()
r2_ajustado = modelo.rsquared_adj * 100
p_valor = resultado_t.pvalue

# Mostrar tarjetas KPI
st.subheader("Indicadores principales")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rendimiento promedio",
    f"{rendimiento_promedio:.2f} ton/ha"
)

col2.metric(
    "R² ajustado",
    f"{r2_ajustado:.2f}%"
)

col3.metric(
    "P-valor de la prueba",
    f"{p_valor:.6f}"
)

st.divider()

st.info("Los indicadores se calculan directamente a partir de los datos almacenados en DuckDB.")


# FILTROS INTERACTIVOS
st.sidebar.header("Filtros de parcelas")

# Selector de tratamiento
tratamiento_seleccionado = st.sidebar.selectbox(
    "Seleccionar tratamiento",
    ["Todos", "Tratamiento_A", "Tratamiento_B"]
)

# Control deslizante de volumen de riego
riego_min = float(df["volumen_riego"].min())
riego_max = float(df["volumen_riego"].max())

rango_riego = st.sidebar.slider(
    "Rango de volumen de riego (m³/ha)",
    min_value=riego_min,
    max_value=riego_max,
    value=(riego_min, riego_max),
    step=1.0
)

# Aplicar filtros
df_filtrado = df[
    (df["volumen_riego"] >= rango_riego[0]) &
    (df["volumen_riego"] <= rango_riego[1])
]

if tratamiento_seleccionado != "Todos":
    df_filtrado = df_filtrado[
        df_filtrado["tipo_tratamiento"] == tratamiento_seleccionado
    ]

# Mostrar cantidad de parcelas
st.sidebar.write(
    f"Parcelas seleccionadas: {len(df_filtrado)}"
)


# SECCION DE VISUALIZACIONES
st.subheader("Visualizaciones del analisis")

# Comprobar que existan suficientes parcelas
if len(df_filtrado) < 3:

    st.warning(
        "Selecciona al menos 3 parcelas para generar las visualizaciones."
    )

else:

    # Crear un nuevo modelo con los datos filtrados
    Y_filtrado = df_filtrado["rendimiento_ton"]

    X_filtrado = df_filtrado[
        ["volumen_riego", "fertilizante_kg"]
    ]

    X_filtrado = sm.add_constant(X_filtrado)

    modelo_filtrado = sm.OLS(
        Y_filtrado,
        X_filtrado
    ).fit()

    # Primera fila de graficas
    col1, col2 = st.columns(2)

    # GRAFICA 1: HEATMAP
    with col1:

        st.markdown("### 1. Matriz de correlacion")

        correlaciones = df_filtrado[
            ["volumen_riego", "fertilizante_kg", "rendimiento_ton"]
        ].corr(method="pearson")

        fig1, ax1 = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            correlaciones,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax1
        )

        st.pyplot(fig1)
        plt.close(fig1)

    # GRAFICA 2: DISPERSION CON REGRESION
    with col2:

        st.markdown("### 2. Riego vs rendimiento")

        riego_nuevo = np.linspace(
            df_filtrado["volumen_riego"].min(),
            df_filtrado["volumen_riego"].max(),
            100
        )

        datos_prediccion = pd.DataFrame({
            "const": 1.0,
            "volumen_riego": riego_nuevo,
            "fertilizante_kg": df_filtrado["fertilizante_kg"].mean()
        })

        predicciones = modelo_filtrado.get_prediction(
            datos_prediccion
        ).summary_frame()

        fig2, ax2 = plt.subplots(figsize=(7, 5))

        ax2.scatter(
            df_filtrado["volumen_riego"],
            df_filtrado["rendimiento_ton"],
            alpha=0.7,
            label="Parcelas"
        )

        ax2.plot(
            riego_nuevo,
            predicciones["mean"],
            color="red",
            label="Regresion OLS"
        )

        ax2.fill_between(
            riego_nuevo,
            predicciones["mean_ci_lower"],
            predicciones["mean_ci_upper"],
            color="red",
            alpha=0.2,
            label="Intervalo de confianza 95%"
        )

        ax2.set_xlabel("Volumen de riego (m³/ha)")
        ax2.set_ylabel("Rendimiento (ton/ha)")
        ax2.legend()

        st.pyplot(fig2)
        plt.close(fig2)

    # Segunda fila de graficas
    col3, col4 = st.columns(2)

    # GRAFICA 3: BOXPLOT
    with col3:

        st.markdown("### 3. Comparacion de tratamientos")

        fig3, ax3 = plt.subplots(figsize=(7, 5))

        sns.boxplot(
            data=df_filtrado,
            x="tipo_tratamiento",
            y="rendimiento_ton",
            hue="tipo_tratamiento",
            palette="Set2",
            legend=False,
            ax=ax3
        )

        ax3.set_xlabel("Tipo de tratamiento")
        ax3.set_ylabel("Rendimiento (ton/ha)")

        st.pyplot(fig3)
        plt.close(fig3)

    # GRAFICA 4: HISTOGRAMA DE RESIDUOS
    with col4:

        st.markdown("### 4. Distribucion de residuos")

        residuos = modelo_filtrado.resid

        fig4, ax4 = plt.subplots(figsize=(7, 5))

        sns.histplot(
            residuos,
            kde=True,
            stat="density",
            bins=15,
            color="skyblue",
            edgecolor="black",
            ax=ax4
        )

        ax4.axvline(
            x=0,
            color="red",
            linestyle="--",
            label="Residuo cero"
        )

        ax4.set_xlabel("Residuos (ton/ha)")
        ax4.set_ylabel("Densidad")
        ax4.legend()

        st.pyplot(fig4)
        plt.close(fig4)