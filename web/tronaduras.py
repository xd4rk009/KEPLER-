# tronaduras.py
# ==========================================================
# Vista para la gestión y edición de datos de Tronaduras.
# Permite visualizar y editar la tabla 'Tronaduras' 
# (por ejemplo, del esquema Processed_Data) almacenada en DuckDB.
# ==========================================================

import streamlit as st
import duckdb
import pandas as pd

#DB_NAME = "../data/Tronaduras_vs_Sismicidad.db"
DB_NAME = r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"


def load_tronaduras():
    """Carga la tabla Tronaduras desde DuckDB."""
    conn = duckdb.connect(DB_NAME)
    query = 'SELECT * FROM "Processed_Data"."Tronaduras"'
    df = conn.execute(query).fetchdf()
    conn.close()
    # Convertir el índice en una columna para visualizarlo
    df.reset_index(inplace=True)
    return df

def update_tronaduras(updated_df):
    """Actualiza la tabla Tronaduras en DuckDB con los datos editados."""
    conn = duckdb.connect(DB_NAME)
    conn.execute('DROP TABLE IF EXISTS "Processed_Data"."Tronaduras"')
    conn.register("temp_tronaduras", updated_df)
    conn.execute('CREATE TABLE "Processed_Data"."Tronaduras" AS SELECT * FROM temp_tronaduras')
    conn.close()

def app():
    st.title("Vista Tronaduras")
    st.write("Edite los datos de la tabla Tronaduras:")
    
    df = load_tronaduras()
    edited_df = st.data_editor(df, use_container_width=True)
    
    if st.button("Guardar cambios"):
        update_tronaduras(edited_df)
        st.success("Tabla Tronaduras actualizada correctamente.")
