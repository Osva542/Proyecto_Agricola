import duckdb

# 1. Crear la conexión con DuckDB
conexion = duckdb.connect(database="agricola.duckdb")

# 2. Establecer la semilla aleatoria
conexion.execute("SELECT setseed(0.42)")

print("Conexión con DuckDB establecida correctamente.")

# 3. Crear la tabla con 100 parcelas
conexion.execute("""
    CREATE OR REPLACE TABLE parcelas AS

    WITH datos_aleatorios AS MATERIALIZED (
        SELECT
            parcela_id,

            100 + random() * 200 AS volumen_riego,

            50 + random() * 100 AS fertilizante_kg,

            random() AS probabilidad_tratamiento,

            random() AS ruido

        FROM range(1, 101) AS serie(parcela_id)
    )

    SELECT
        parcela_id,

        ROUND(volumen_riego, 2) AS volumen_riego,

        ROUND(fertilizante_kg, 2) AS fertilizante_kg,

        CASE
            WHEN probabilidad_tratamiento < 0.50
                THEN 'Tratamiento_A'
            ELSE 'Tratamiento_B'
        END AS tipo_tratamiento,

        ROUND(
            5.0
            + (0.035 * volumen_riego)
            + (0.020 * fertilizante_kg)
            + CASE
                WHEN probabilidad_tratamiento < 0.50
                    THEN 0.0
                ELSE 1.8
              END
            + (ruido * 1.6 - 0.8),
            2
        ) AS rendimiento_ton

    FROM datos_aleatorios
""")

print("Tabla de parcelas creada correctamente.")

# 4. Consultar cuántas parcelas se generaron
total = conexion.execute(
    "SELECT COUNT(*) FROM parcelas"
).fetchone()

print("Total de parcelas:", total[0])

# 5. Mostrar los primeros 10 registros
registros = conexion.execute("""
    SELECT * FROM parcelas
    ORDER BY parcela_id
    LIMIT 10
""").fetchall()

print("\nPrimeros 10 registros:")

for registro in registros:
    print(registro)
