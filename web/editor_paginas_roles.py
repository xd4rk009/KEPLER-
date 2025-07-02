# ==========================================================
# editor_paginas_roles.py
# ==========================================================
# Vista para editar las tablas de Paginas y Roles.
# - La pestaña "Paginas" permite modificar la tabla de Paginas:
#   se pueden editar 'page_name', 'file_name' y 'description'.
# - La pestaña "Roles" permite modificar la tabla de Roles:
#   se asignan las vistas permitidas para cada rol (campo 'allowed_pages',
#   con los nombres de las vistas separados por comas).
# Esta vista es accesible para administradores y permite actualizar la
# configuración de la navegación en la aplicación.
# ==========================================================


import streamlit as st
import duckdb
import pandas as pd

#DB_NAME = "../data/Tronaduras_vs_Sismicidad.db"
DB_NAME = r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"



def load_pages():
    """Carga la tabla Paginas desde DuckDB, incluyendo la columna 'id'."""
    conn = duckdb.connect(DB_NAME)
    df = conn.execute("SELECT * FROM Paginas").fetchdf()
    conn.close()
    return df

def load_roles():
    """Carga la tabla Roles desde DuckDB."""
    conn = duckdb.connect(DB_NAME)
    df = conn.execute("SELECT * FROM Roles").fetchdf()
    conn.close()
    return df

def save_pages(updated_df):
    """Actualiza la tabla Paginas en DuckDB con los datos modificados."""
    conn = duckdb.connect(DB_NAME)
    
    for _, row in updated_df.iterrows():
        conn.execute(""" 
            INSERT INTO Paginas (page_name, file_name, description) 
            VALUES (?, ?, ?)
            ON CONFLICT(file_name) DO UPDATE
            SET page_name = EXCLUDED.page_name, 
                description = EXCLUDED.description
        """, [row["page_name"], row["file_name"], row["description"]])
    
    conn.close()


def save_roles(updated_df):
    """Actualiza la tabla Roles en DuckDB con los datos modificados."""
    conn = duckdb.connect(DB_NAME)
    
    # Actualizar los registros existentes y agregar los nuevos
    for _, row in updated_df.iterrows():
        conn.execute(""" 
            INSERT INTO Roles (role_name, allowed_pages) 
            VALUES (?, ?)
            ON CONFLICT(role_name) DO UPDATE
            SET allowed_pages = ?
        """, [row["role_name"], row["allowed_pages"], row["allowed_pages"]])
    
    conn.close()

def app():
    st.title("Asignación de Páginas a Roles")
    
    tab1, tab2 = st.tabs(["Páginas", "Roles"])
    
    with tab1:
        st.subheader("Editar Páginas")
        st.write("Modifica la tabla de Páginas: 'page_name', 'file_name' y 'description'.")
        df_pages = load_pages()
        edited_pages = st.data_editor(df_pages, use_container_width=True)
        
        if st.button("Guardar cambios en Páginas", key="save_pages"):
            save_pages(edited_pages)
            st.success("Tabla Páginas actualizada correctamente.")
            st.rerun()  # Recargar la página para reflejar los cambios
        
    with tab2:
        st.subheader("Editar Roles")
        st.write("Modifica los roles y asigna las vistas permitidas (separadas por comas) a cada rol.")
        df_roles = load_roles()
        edited_roles = st.data_editor(df_roles, use_container_width=True)
        
        if st.button("Guardar cambios en Roles", key="save_roles"):
            save_roles(edited_roles)
            st.success("Tabla Roles actualizada correctamente.")
            st.rerun()  # Recargar la página para reflejar los cambios
        
        # Formulario para agregar un nuevo rol
        st.subheader("Agregar Nuevo Rol")
        new_role_name = st.text_input("Nombre del Rol")
        
        # Cargar las páginas disponibles desde la base de datos
        conn = duckdb.connect(DB_NAME)
        all_pages = conn.execute("SELECT page_name FROM Paginas").fetchdf()["page_name"].tolist()
        conn.close()
        
        # Asegurarse de que "Home" esté en las opciones disponibles para evitar el error
        if "Home" not in all_pages:
            all_pages.append("Home")
        
        # Selección de páginas para el nuevo rol, incluyendo la página Home por defecto
        selected_pages = st.multiselect("Seleccionar Páginas Permitidas", all_pages, default=["Home"])
        
        if st.button("Agregar Rol"):
            if new_role_name and selected_pages:
                # Convertir las páginas seleccionadas en una cadena separada por comas
                allowed_pages = ",".join(selected_pages)
                conn = duckdb.connect(DB_NAME)
                conn.execute("""
                    INSERT INTO Roles (role_name, allowed_pages) 
                    VALUES (?, ?)
                """, [new_role_name, allowed_pages])
                conn.close()
                st.success(f"Rol '{new_role_name}' agregado exitosamente.")
                st.rerun()  # Recargar la página para mostrar el nuevo rol
            else:
                st.error("Por favor, completa todos los campos.")
        
        # Formulario para eliminar un rol
        st.subheader("Eliminar Rol")
        roles_to_delete = [role for role in df_roles['role_name'] if role.lower() != 'administrador']
        
        # Verificar si roles_to_delete tiene valores antes de mostrar el selectbox
        if roles_to_delete:
            role_to_delete = st.selectbox("Seleccionar Rol a Eliminar", roles_to_delete)
            if st.button("Eliminar Rol"):
                if role_to_delete:
                    conn = duckdb.connect(DB_NAME)
                    conn.execute("DELETE FROM Roles WHERE role_name = ?", [role_to_delete])
                    conn.close()
                    st.success(f"Rol '{role_to_delete}' eliminado exitosamente.")
                    st.rerun()  # Recargar la página para reflejar el cambio
        else:
            st.warning("No hay roles disponibles para eliminar.")
