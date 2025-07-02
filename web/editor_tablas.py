import streamlit as st
import duckdb
import pandas as pd
import os

#DB_NAME = "../data/Tronaduras_vs_Sismicidad.db"
DB_NAME = r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"


# Función para obtener esquemas que contienen tablas
def get_schemas_with_tables():
    conn = duckdb.connect(DB_NAME)
    query = """
    SELECT DISTINCT table_schema
    FROM information_schema.tables
    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
    """
    schemas = conn.execute(query).fetchdf()
    conn.close()

    valid_schemas = []
    for schema in schemas['table_schema']:
        tables = get_tables_in_schema(schema)
        if tables:
            valid_schemas.append(schema)

    return valid_schemas

# Función para obtener tablas dentro de un esquema
def get_tables_in_schema(schema):
    conn = duckdb.connect(DB_NAME)
    query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
    tables = conn.execute(query).fetchdf()
    conn.close()
    return tables['table_name'].tolist()

# Función para mostrar la tabla editable
def show_editable_table(table_name):
    conn = duckdb.connect(DB_NAME)
    df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
    conn.close()
    df.reset_index(inplace=True)
    edited_df = st.data_editor(df, use_container_width=True)
    return edited_df

# Función para guardar los cambios realizados
def save_changes(table_name, df):
    conn = duckdb.connect(DB_NAME)
    try:
        conn.execute(f"DELETE FROM {table_name}")
        for _, row in df.iterrows():
            insert_query = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({', '.join(['?' for _ in df.columns])})"
            conn.execute(insert_query, tuple(row))
        conn.commit()
        st.success("¡Cambios guardados correctamente!")
    except Exception as e:
        conn.rollback()
        st.error(f"Error al guardar los cambios: {e}")
    finally:
        conn.close()

# Página principal que inicializa y maneja el esquema y tabla
def app():
    st.title("Editor de Tablas con Selección de Esquemas")

    # Paso 1: Selección del esquema
    if 'selected_schema' not in st.session_state:
        st.session_state.selected_schema = None

    schemas = get_schemas_with_tables()

    if not schemas:
        st.error("No hay esquemas con tablas disponibles.")
        return

    # Asegurar que "main" esté primero en la lista, pero excluiremos la tabla 'Pages' más adelante
    if 'main' in schemas:
        schemas.remove('main')
        schemas = ['main'] + sorted(schemas)

    # Mostrar botones o radio buttons para cada esquema
    selected_schema = st.radio(
        "Selecciona un esquema",
        schemas,
        index=schemas.index(st.session_state.selected_schema) if st.session_state.selected_schema in schemas else 0,
        key="schema_radio_buttons"  # Usar un key único para evitar reordenamiento
    )

    # Si el esquema seleccionado cambia, actualizar el estado y forzar el rerun
    if selected_schema != st.session_state.selected_schema:
        st.session_state.selected_schema = selected_schema
        st.rerun()  # Forzar un rerun para actualizar la vista de las tablas

    # Paso 2: Selección de tablas dentro del esquema seleccionado
    tables_in_schema = get_tables_in_schema(st.session_state.selected_schema)
    if not tables_in_schema:
        st.error("No hay tablas en este esquema.")
        return

    # Excluir la tabla "Pages" del esquema "main" si está seleccionada
    if st.session_state.selected_schema == "main" and "Paginas" in tables_in_schema:
        tables_in_schema.remove("Paginas")

    selected_table = st.selectbox("Selecciona una tabla", tables_in_schema)

    # Paso 3: Mostrar la tabla editable
    if selected_table:
        table_name = f"{st.session_state.selected_schema}.{selected_table}"
        edited_data = show_editable_table(table_name)

        # Paso 4: Botón para guardar los cambios
        if st.button("Guardar cambios"):
            save_changes(table_name, edited_data)
