# sismicidad.py
# ==========================================================
# Vista para la gestión y edición de datos de Sismicidad.
# Permite visualizar y editar la tabla 'Sismicidad'
# (por ejemplo, del esquema Raw_Data) almacenada en DuckDB.
# ==========================================================

import streamlit as st
import duckdb
import pandas as pd

#DB_NAME = "../data/Tronaduras_vs_Sismicidad.db"
DB_NAME = r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"


def load_sismicidad():
    """Carga la tabla Sismicidad desde DuckDB."""
    conn = duckdb.connect(DB_NAME)
    query = 'SELECT * FROM "Raw_Data"."Sismicidad"'
    df = conn.execute(query).fetchdf()
    conn.close()
    df.reset_index(inplace=True)
    return df

def update_sismicidad(updated_df):
    """Actualiza la tabla Sismicidad en DuckDB con los datos editados."""
    conn = duckdb.connect(DB_NAME)
    conn.execute('DROP TABLE IF EXISTS "Raw_Data"."Sismicidad"')
    conn.register("temp_sismicidad", updated_df)
    conn.execute('CREATE TABLE "Raw_Data"."Sismicidad" AS SELECT * FROM temp_sismicidad')
    conn.close()

def app():
    st.title("Vista Sismicidad")
    st.write("Edite los datos de la tabla Sismicidad:")
    
    df = load_sismicidad()
    edited_df = st.data_editor(df, use_container_width=True)
    
    if st.button("Guardar cambios"):
        update_sismicidad(edited_df)
        st.success("Tabla Sismicidad actualizada correctamente.")
