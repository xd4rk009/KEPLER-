# editor_usuarios.py
# ==========================================================
# Página para editar Usuarios y Roles.
# Permite al administrador:
# - Modificar la tabla Usuarios (username, password y role).
# - Modificar la tabla Roles, que contiene los roles y las vistas permitidas 
#   (allowed_pages) para cada rol.
# La asignación de vistas se realiza en una única tabla (Roles).
# ==========================================================

import streamlit as st
import duckdb
import pandas as pd

#DB_NAME = "../data/Tronaduras_vs_Sismicidad.db"
DB_NAME = r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"


def load_users():
    """Carga la tabla Usuarios desde DuckDB."""
    conn = duckdb.connect(DB_NAME)
    df = conn.execute("SELECT * FROM Usuarios").fetchdf()
    conn.close()
    return df

def load_roles():
    """Carga la tabla Roles desde DuckDB."""
    conn = duckdb.connect(DB_NAME)
    df = conn.execute("SELECT * FROM Roles").fetchdf()
    conn.close()
    return df

def load_pages():
    """Carga las páginas disponibles desde DuckDB y asegura que 'Home' esté presente."""
    conn = duckdb.connect(DB_NAME)
    df = conn.execute("SELECT page_name FROM Paginas").fetchdf()
    conn.close()
    pages = df['page_name'].tolist()
    if 'Home' not in pages:
        pages.append('Home')  # Asegura que 'Home' siempre esté en la lista
    return pages

def save_users(updated_df):
    """Actualiza la tabla Usuarios en DuckDB con los datos modificados."""
    conn = duckdb.connect(DB_NAME)
    conn.execute("DROP TABLE IF EXISTS Usuarios")
    conn.register("temp_users", updated_df)
    conn.execute("CREATE TABLE Usuarios AS SELECT * FROM temp_users")
    conn.close()

def save_roles(updated_df):
    """Actualiza la tabla Roles en DuckDB con los datos modificados."""
    conn = duckdb.connect(DB_NAME)
    conn.execute("DROP TABLE IF EXISTS Roles")
    conn.register("temp_roles", updated_df)
    conn.execute("CREATE TABLE Roles AS SELECT * FROM temp_roles")
    conn.close()

def app():
    st.title("Editor de Usuarios y Roles")
    st.write("Esta página permite al administrador editar los usuarios y roles, asignando las vistas permitidas a cada rol.")
    
    tab1, tab2 = st.tabs(["Usuarios", "Roles"])
    
    with tab1:
        st.subheader("Editar Usuarios")
        st.write("Modifica la tabla Usuarios: 'username', 'password' y 'role'.")
        df_users = load_users()
        edited_users = st.data_editor(df_users, use_container_width=True)
        if st.button("Guardar cambios en Usuarios", key="save_users"):
            save_users(edited_users)
            st.success("Tabla Usuarios actualizada correctamente.")
        
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        # Menú desplegable para seleccionar la acción a realizar: agregar o eliminar usuario
        st.subheader("Administrar Usuarios")
        user_action = st.selectbox("Seleccione acción", ["Agregar Nuevo Usuario", "Eliminar un Usuario"])
        
        if user_action == "Agregar Nuevo Usuario":
            new_username = st.text_input("Nombre de Usuario", key="new_username")
            new_password = st.text_input("Contraseña", type="password", key="new_password")
            
            # Cargar roles actualizados
            roles = load_roles()
            role_names = [role['role_name'] for _, role in roles.iterrows()]
            new_role = st.selectbox("Rol", role_names, key="new_role")
            
            if st.button("Agregar Usuario", key="add_user"):
                if new_username and new_password:
                    conn = duckdb.connect(DB_NAME)
                    conn.execute("""
                        INSERT INTO Usuarios (username, password, role) 
                        VALUES (?, ?, ?)
                    """, [new_username, new_password, new_role])
                    conn.close()
                    st.success(f"Usuario '{new_username}' agregado exitosamente.")
                    st.rerun()  # Recargar la página para mostrar el nuevo usuario
                else:
                    st.error("Por favor, completa todos los campos.")
        else:
            # Opción para eliminar un usuario
            # Se utiliza el dataframe ya cargado para mostrar la lista actual de usuarios
            users_list = df_users["username"].tolist()
            user_to_delete = st.selectbox("Seleccione el Usuario a Eliminar", users_list, key="delete_user")
            if st.button("Eliminar Usuario", key="delete_user_button"):
                if user_to_delete:
                    conn = duckdb.connect(DB_NAME)
                    conn.execute("DELETE FROM Usuarios WHERE username = ?", [user_to_delete])
                    conn.close()
                    st.success(f"Usuario '{user_to_delete}' eliminado exitosamente.")
                    st.rerun()  # Recargar la página para reflejar el cambio

    with tab2:
        st.subheader("Editar Roles")
        st.write("Modifica los roles y asigna las vistas permitidas (separadas por comas) a cada rol.")
        df_roles = load_roles()
        edited_roles = st.data_editor(df_roles, use_container_width=True)
        if st.button("Guardar cambios en Roles", key="save_roles"):
            save_roles(edited_roles)
            st.success("Tabla Roles actualizada correctamente.")
        
        # Formulario para agregar un nuevo rol
        st.subheader("Agregar Nuevo Rol")
        new_role_name = st.text_input("Nombre del Rol", key="new_role_name")
        
        # Cargar las páginas disponibles desde la base de datos
        available_pages = load_pages()
        
        # Selección de páginas para el nuevo rol, incluyendo la página Home por defecto
        selected_pages = st.multiselect("Seleccionar Páginas Permitidas", available_pages, default=["Home"], key="selected_pages")
        
        if st.button("Agregar Rol", key="add_role"):
            if new_role_name and selected_pages:
                # Convertir las páginas seleccionadas en una cadena separada por comas
                allowed_pages = ",".join(selected_pages)
                if new_role_name.lower() != "administrador":  # No permitir agregar un rol "Administrador"
                    conn = duckdb.connect(DB_NAME)
                    conn.execute("""
                        INSERT INTO Roles (role_name, allowed_pages) 
                        VALUES (?, ?)
                    """, [new_role_name, allowed_pages])
                    conn.close()
                    st.success(f"Rol '{new_role_name}' agregado exitosamente.")
                    st.rerun()  # Recargar la página para mostrar el nuevo rol
                else:
                    st.error("No se puede agregar el rol 'Administrador'.")
            else:
                st.error("Por favor, completa todos los campos.")
        
        # Formulario para eliminar un rol
        st.subheader("Eliminar Rol")
        roles_to_delete = [role for role in df_roles['role_name'] if role.lower() != 'administrador']
        
        if roles_to_delete:
            role_to_delete = st.selectbox("Seleccionar Rol a Eliminar", roles_to_delete, key="role_to_delete")
            if st.button("Eliminar Rol", key="delete_role"):
                if role_to_delete:
                    conn = duckdb.connect(DB_NAME)
                    conn.execute("DELETE FROM Roles WHERE role_name = ?", [role_to_delete])
                    conn.close()
                    st.success(f"Rol '{role_to_delete}' eliminado exitosamente.")
                    st.rerun()  # Recargar la página para reflejar el cambio
        else:
            st.warning("No hay roles disponibles para eliminar.")
