import streamlit as st
import os
from datetime import datetime
import duckdb
import subprocess
import sys

# Configuración de rutas más robusta para diferentes entornos
WEB_DIR = "./"

# Detectar si estamos en un entorno local o en la nube
if os.path.exists(r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"):
    # Entorno local
    DB_NAME = r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"
else:
    # Entorno en la nube - usar ruta relativa
    DB_NAME = "./data/tronaduras_vs_sismicidad.db"
    # Crear directorio si no existe
    os.makedirs("./data", exist_ok=True)

# Directorio para archivos subidos
UPLOADED_DIR = "./uploaded_pages/"

# Código base para una nueva página
DEFAULT_CODE = """
import streamlit as st
from datetime import datetime

# Fecha y hora de creación de la página
creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.write("Página creada el:", creation_date)

# Inicia tu código de Streamlit a partir de aquí.
def app():
    st.title("Mi Nueva Página")
    st.write("Contenido de la nueva página creada.")

if __name__ == "__main__":
    app()
"""

def create_new_page(file_name, code_content):
    """Crear una nueva página con validación mejorada"""
    try:
        # Validar y limpiar el nombre del archivo
        if not isinstance(file_name, str) or not file_name.strip():
            st.error("El nombre del archivo debe ser una cadena de texto válida.")
            return
        
        file_name = str(file_name).strip()
        if not file_name.endswith(".py"):
            file_name += ".py"
        
        # Limpiar caracteres especiales
        file_name = file_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Crear ruta completa de forma segura
        file_path = os.path.join(str(WEB_DIR), file_name)
        
        if os.path.exists(file_path):
            st.error(f"El archivo '{file_name}' ya existe. Por favor, elige otro nombre.")
            return
        
        # Crear directorio si no existe
        os.makedirs(str(WEB_DIR), exist_ok=True)
        
        # Escribir archivo
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(str(code_content))
        
        st.success(f"Página '{file_name}' creada correctamente en el directorio '{WEB_DIR}'.")
        
    except Exception as e:
        st.error(f"Error al crear la página: {str(e)}")

def save_uploaded_file(uploaded_file, custom_name=None):
    """Guardar archivo subido con validación mejorada"""
    try:
        # Crear directorio si no existe
        os.makedirs(str(UPLOADED_DIR), exist_ok=True)
        
        if custom_name:
            if not isinstance(custom_name, str):
                st.error("El nombre personalizado debe ser una cadena de texto.")
                return None
            
            custom_name = str(custom_name).strip()
            if not custom_name.endswith(".py"):
                custom_name += ".py"
            file_name = custom_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        else:
            file_name = str(uploaded_file.name)
        
        # Crear ruta de forma segura
        file_path = os.path.join(str(UPLOADED_DIR), file_name)
        
        if os.path.exists(file_path):
            st.error(f"El archivo '{file_name}' ya existe. Por favor, elige otro nombre.")
            return None
        
        # Guardar archivo
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Archivo '{file_name}' subido correctamente.")
        return file_name
        
    except Exception as e:
        st.error(f"Error al guardar el archivo: {str(e)}")
        return None

def run_uploaded_page(file_name):
    """Lanzar página con mejor manejo de errores"""
    try:
        # Validar entrada
        if not isinstance(file_name, str) or not file_name.strip():
            st.error("Nombre de archivo inválido.")
            return
        
        file_name = str(file_name).strip()
        file_path = os.path.join(str(UPLOADED_DIR), file_name)
        
        if not os.path.exists(file_path):
            st.error(f"El archivo '{file_name}' no existe.")
            return
        
        st.subheader(f"Lanzar Aplicación: {file_name}")
        
        # Detectar entorno
        is_cloud = not os.path.exists(r"C:\Users")
        
        if is_cloud:
            st.warning("""
            **Nota importante**: Estás en un entorno en la nube (Streamlit Cloud).
            - La funcionalidad de abrir nuevas pestañas puede estar limitada.
            - Como alternativa, puedes descargar el archivo y ejecutarlo localmente.
            """)
        else:
            st.info(f"""
            Estás a punto de ejecutar el script **{file_name}** como una aplicación independiente.
            - Se abrirá en una **nueva pestaña** del navegador.
            - Se ejecutará en un puerto diferente (ej: 8502, 8503, ...).
            """)

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🚀 Lanzar '{file_name}'", key=f"launch_{file_name}"):
                if is_cloud:
                    st.warning("Esta funcionalidad no está disponible en Streamlit Cloud. Descarga el archivo para ejecutarlo localmente.")
                else:
                    try:
                        python_executable = sys.executable
                        command = [str(python_executable), "-m", "streamlit", "run", str(file_path)]
                        
                        subprocess.Popen(command)
                        
                        st.success(f"✅ Se ha enviado la orden para iniciar '{file_name}'.")
                        st.info("Revisa tu navegador, se debería haber abierto una nueva pestaña.")
                        
                    except Exception as e:
                        st.error(f"Error al lanzar la aplicación: {str(e)}")
        
        with col2:
            # Opción para descargar el archivo
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                
                st.download_button(
                    label=f"📥 Descargar {file_name}",
                    data=file_content,
                    file_name=file_name,
                    mime="text/plain",
                    key=f"download_{file_name}"
                )
            except Exception as e:
                st.error(f"Error al preparar la descarga: {str(e)}")
                
    except Exception as e:
        st.error(f"Error general en run_uploaded_page: {str(e)}")

def delete_page(file_name, username, password, is_uploaded=False):
    """Eliminar página con validación mejorada"""
    try:
        # Validar entradas
        if not all(isinstance(x, str) and x.strip() for x in [file_name, username, password]):
            st.error("Todos los campos deben ser cadenas de texto válidas.")
            return
        
        # Conectar a la base de datos
        if not os.path.exists(str(DB_NAME)):
            st.error("Base de datos no encontrada.")
            return
        
        conn = duckdb.connect(str(DB_NAME))
        query = "SELECT password, role FROM Usuarios WHERE username = ?"
        df = conn.execute(query, [str(username)]).fetchdf()
        conn.close()
        
        if not df.empty and df['role'].iloc[0] == 'Administrador' and df['password'].iloc[0] == str(password):
            directory = str(UPLOADED_DIR) if is_uploaded else str(WEB_DIR)
            file_path = os.path.join(directory, str(file_name))
            
            if os.path.exists(file_path):
                os.remove(file_path)
                page_type = "subida" if is_uploaded else "creada"
                st.success(f"Página {page_type} '{file_name}' eliminada exitosamente.")
            else:
                st.error(f"La página '{file_name}' no existe.")
        else:
            st.error("Credenciales incorrectas o no tienes permisos de administrador.")
            
    except Exception as e:
        st.error(f"Error al eliminar la página: {str(e)}")

def edit_page(file_name, is_uploaded=False):
    """Editar página con validación mejorada"""
    try:
        if not isinstance(file_name, str) or not file_name.strip():
            st.error("Nombre de archivo inválido.")
            return
        
        directory = str(UPLOADED_DIR) if is_uploaded else str(WEB_DIR)
        file_path = os.path.join(directory, str(file_name))
        
        if not os.path.exists(file_path):
            st.error(f"La página '{file_name}' no existe.")
            return
        
        with open(file_path, "r", encoding="utf-8") as file:
            current_code = file.read()
        
        page_type = "Subida" if is_uploaded else "Creada"
        st.subheader(f"Editar Página {page_type}: {file_name}")
        updated_code = st.text_area(
            "Modifica el código de la página", 
            current_code, 
            height=600, 
            key=f"edit_area_{file_name}"
        )
        
        if st.button("Guardar Cambios", key=f"guardar_edit_{file_name}"):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(str(updated_code))
            st.success(f"Página {page_type.lower()} '{file_name}' actualizada correctamente.")
            
    except Exception as e:
        st.error(f"Error al editar la página: {str(e)}")

def get_all_pages():
    """Obtener todas las páginas con manejo de errores mejorado"""
    try:
        created_pages = []
        if os.path.exists(str(WEB_DIR)):
            files = os.listdir(str(WEB_DIR))
            created_pages = [f for f in files if isinstance(f, str) and f.endswith(".py") and f != os.path.basename(__file__)]
        
        uploaded_pages = []
        if os.path.exists(str(UPLOADED_DIR)):
            files = os.listdir(str(UPLOADED_DIR))
            uploaded_pages = [f for f in files if isinstance(f, str) and f.endswith(".py")]
        
        return created_pages, uploaded_pages
        
    except Exception as e:
        st.error(f"Error al obtener la lista de páginas: {str(e)}")
        return [], []

def app():
    st.set_page_config(page_title="Gestión de Páginas", page_icon="📁", layout="wide")
    st.title("📁 Gestión de Páginas o Vistas")
    
    # Mostrar información del entorno
    is_cloud = not os.path.exists(r"C:\Users")
    env_info = "☁️ Streamlit Cloud" if is_cloud else "💻 Entorno Local"
    st.sidebar.info(f"Entorno: {env_info}")
    
    try:
        created_pages, uploaded_pages = get_all_pages()
        
        col1, col2 = st.columns(2)
        col1.metric("Páginas Creadas", len(created_pages))
        col2.metric("Páginas Subidas", len(uploaded_pages))
        
        tabs = st.tabs(["📝 Crear Página", "📤 Subir Archivo", "🚀 Ejecutar Páginas", "✏️ Modificar Página", "🗑️ Eliminar Página"])
        
        with tabs[0]:  # Crear
            st.subheader("📝 Crear una nueva página")
            new_file_name = st.text_input("Nombre del archivo (sin extensión)", key="crear_nombre")
            code_content = st.text_area("Escribe el código de la nueva página", DEFAULT_CODE, height=400, key="crear_codigo")
            
            if st.button("Crear Página", key="crear_boton"):
                if new_file_name and new_file_name.strip():
                    create_new_page(new_file_name, code_content)
                    st.rerun()
                else:
                    st.error("Por favor, ingresa un nombre de archivo válido.")

        with tabs[1]:  # Subir
            st.subheader("📤 Subir archivo Python")
            uploaded_file = st.file_uploader("Selecciona un archivo Python", type=['py'])
            
            if uploaded_file is not None:
                st.info(f"Archivo seleccionado: {uploaded_file.name}")
                custom_name = st.text_input("Nombre personalizado (opcional)", placeholder="Deja vacío para usar el nombre original")
                
                with st.expander("Vista previa del código"):
                    try:
                        code_preview = uploaded_file.read().decode('utf-8')
                        st.code(code_preview, language='python')
                        uploaded_file.seek(0)
                    except Exception as e:
                        st.error(f"Error al leer el archivo: {str(e)}")

                if st.button("Guardar Archivo", key="subir_boton"):
                    saved_name = save_uploaded_file(uploaded_file, custom_name)
                    if saved_name:
                        st.rerun()

        with tabs[2]:  # Ejecutar
            st.subheader("🚀 Ejecutar Páginas Subidas")
            
            if is_cloud:
                st.warning("⚠️ Funcionalidad limitada en Streamlit Cloud. Usa la opción de descarga para ejecutar localmente.")
            
            if uploaded_pages:
                selected_uploaded = st.selectbox("Selecciona una página subida para lanzar", uploaded_pages, key="exec_uploaded_select")
                if selected_uploaded:
                    run_uploaded_page(selected_uploaded)
            else:
                st.info("No hay páginas subidas disponibles para ejecutar.")

        with tabs[3]:  # Modificar
            st.subheader("✏️ Modificar una página existente")
            edit_tabs = st.tabs(["Editar Creadas", "Editar Subidas"])
            
            with edit_tabs[0]:
                if created_pages:
                    page_to_edit_c = st.selectbox("Selecciona página creada para editar", created_pages, key="editar_created_select")
                    if page_to_edit_c:
                        edit_page(page_to_edit_c, is_uploaded=False)
                else:
                    st.info("No hay páginas creadas para editar.")
            
            with edit_tabs[1]:
                if uploaded_pages:
                    page_to_edit_u = st.selectbox("Selecciona página subida para editar", uploaded_pages, key="editar_uploaded_select")
                    if page_to_edit_u:
                        edit_page(page_to_edit_u, is_uploaded=True)
                else:
                    st.info("No hay páginas subidas para editar.")

        with tabs[4]:  # Eliminar
            st.subheader("🗑️ Eliminar una página")
            delete_tabs = st.tabs(["Eliminar Creadas", "Eliminar Subidas"])
            
            def delete_ui(page_list, is_uploaded):
                if page_list:
                    page_type = "subida" if is_uploaded else "creada"
                    key_prefix = f"del_{page_type}"
                    
                    page_to_delete = st.selectbox(f"Selecciona una página {page_type} para eliminar", page_list, key=f"{key_prefix}_select")
                    
                    username = st.text_input("Nombre de usuario", key=f"{key_prefix}_user")
                    password = st.text_input("Contraseña", type="password", key=f"{key_prefix}_pass")
                    
                    if st.button(f"Eliminar Página {page_type.capitalize()}", key=f"{key_prefix}_btn"):
                        if username and password and page_to_delete:
                            delete_page(page_to_delete, username, password, is_uploaded=is_uploaded)
                            st.rerun()
                        else:
                            st.error("Por favor, completa todos los campos.")
                else:
                    page_type = "subidas" if is_uploaded else "creadas"
                    st.info(f"No hay páginas {page_type} para eliminar.")

            with delete_tabs[0]:
                delete_ui(created_pages, is_uploaded=False)
            with delete_tabs[1]:
                delete_ui(uploaded_pages, is_uploaded=True)
                
    except Exception as e:
        st.error(f"Error general en la aplicación: {str(e)}")
        st.info("Si el problema persiste, verifica que todos los directorios y archivos necesarios existan.")

if __name__ == "__main__":
    app()
