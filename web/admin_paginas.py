
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

def safe_str_convert(value, default=""):
    """Convierte cualquier valor a string de forma segura"""
    if value is None:
        return default
    return str(value).strip()

def safe_path_join(*paths):
    """Une rutas de forma segura, convirtiendo todos los argumentos a string"""
    try:
        # Convertir todos los argumentos a string y filtrar valores vacíos
        safe_paths = [safe_str_convert(p) for p in paths if p is not None]
        safe_paths = [p for p in safe_paths if p]  # Filtrar strings vacíos
        
        if not safe_paths:
            return "./"
            
        return os.path.join(*safe_paths)
    except Exception as e:
        st.error(f"Error al construir ruta: {e}")
        return "./"

def create_new_page(file_name, code_content):
    """Crear una nueva página con validación mejorada"""
    try:
        # Validar y limpiar el nombre del archivo
        file_name = safe_str_convert(file_name)
        if not file_name:
            st.error("El nombre del archivo no puede estar vacío.")
            return
        
        if not file_name.endswith(".py"):
            file_name += ".py"
        
        # Limpiar caracteres especiales
        file_name = file_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Crear ruta completa de forma segura
        file_path = safe_path_join(WEB_DIR, file_name)
        
        if os.path.exists(file_path):
            st.error(f"El archivo '{file_name}' ya existe. Por favor, elige otro nombre.")
            return
        
        # Crear directorio si no existe
        dir_path = safe_str_convert(WEB_DIR)
        os.makedirs(dir_path, exist_ok=True)
        
        # Escribir archivo
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(safe_str_convert(code_content))
        
        st.success(f"Página '{file_name}' creada correctamente.")
        
    except Exception as e:
        st.error(f"Error al crear la página: {str(e)}")

def save_uploaded_file(uploaded_file, custom_name=None):
    """Guardar archivo subido con validación mejorada"""
    try:
        # Crear directorio si no existe
        dir_path = safe_str_convert(UPLOADED_DIR)
        os.makedirs(dir_path, exist_ok=True)
        
        if custom_name:
            custom_name = safe_str_convert(custom_name)
            if not custom_name:
                st.error("El nombre personalizado no puede estar vacío.")
                return None
            
            if not custom_name.endswith(".py"):
                custom_name += ".py"
            file_name = custom_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        else:
            file_name = safe_str_convert(uploaded_file.name)
        
        # Crear ruta de forma segura
        file_path = safe_path_join(UPLOADED_DIR, file_name)
        
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
        # Validar entrada de forma más robusta
        if file_name is None:
            st.error("No se ha seleccionado ningún archivo.")
            return
            
        file_name = safe_str_convert(file_name)
        if not file_name:
            st.error("Nombre de archivo inválido.")
            return
        
        file_path = safe_path_join(UPLOADED_DIR, file_name)
        
        if not os.path.exists(file_path):
            st.error(f"El archivo '{file_name}' no existe.")
            return
        
        st.subheader(f"Lanzar Aplicación: {file_name}")
        
        # Detectar entorno
        is_cloud = not os.path.exists(r"C:\Users")
        
        if is_cloud:
            st.warning("""
            **⚠️ Funcionalidad limitada en Streamlit Cloud**
            - La apertura de nuevas pestañas no está disponible
            - Usa el botón de descarga para ejecutar localmente
            """)
        else:
            st.info(f"""
            📋 **Instrucciones:**
            - Se abrirá en una nueva pestaña del navegador
            - Se ejecutará en un puerto diferente (8502, 8503, etc.)
            """)

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🚀 Lanzar '{file_name}'", key=f"launch_{file_name}", disabled=is_cloud):
                if not is_cloud:
                    try:
                        python_executable = safe_str_convert(sys.executable)
                        command = [python_executable, "-m", "streamlit", "run", file_path]
                        
                        subprocess.Popen(command)
                        
                        st.success(f"✅ Aplicación '{file_name}' iniciada.")
                        st.info("🌐 Revisa tu navegador para la nueva pestaña.")
                        
                    except Exception as e:
                        st.error(f"❌ Error al lanzar la aplicación: {str(e)}")
        
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
        st.error(f"Error general: {str(e)}")

def delete_page(file_name, username, password, is_uploaded=False):
    """Eliminar página con validación mejorada"""
    try:
        # Validar entradas de forma más robusta
        file_name = safe_str_convert(file_name)
        username = safe_str_convert(username)
        password = safe_str_convert(password)
        
        if not all([file_name, username, password]):
            st.error("Todos los campos son obligatorios.")
            return
        
        # Conectar a la base de datos
        db_path = safe_str_convert(DB_NAME)
        if not os.path.exists(db_path):
            st.error("Base de datos no encontrada.")
            return
        
        conn = duckdb.connect(db_path)
        query = "SELECT password, role FROM Usuarios WHERE username = ?"
        df = conn.execute(query, [username]).fetchdf()
        conn.close()
        
        if not df.empty and df['role'].iloc[0] == 'Administrador' and df['password'].iloc[0] == password:
            directory = safe_str_convert(UPLOADED_DIR if is_uploaded else WEB_DIR)
            file_path = safe_path_join(directory, file_name)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                page_type = "subida" if is_uploaded else "creada"
                st.success(f"✅ Página {page_type} '{file_name}' eliminada.")
            else:
                st.error(f"❌ La página '{file_name}' no existe.")
        else:
            st.error("❌ Credenciales incorrectas o permisos insuficientes.")
            
    except Exception as e:
        st.error(f"Error al eliminar la página: {str(e)}")

def edit_page(file_name, is_uploaded=False):
    """Editar página con validación mejorada"""
    try:
        file_name = safe_str_convert(file_name)
        if not file_name:
            st.error("Nombre de archivo inválido.")
            return
        
        directory = safe_str_convert(UPLOADED_DIR if is_uploaded else WEB_DIR)
        file_path = safe_path_join(directory, file_name)
        
        if not os.path.exists(file_path):
            st.error(f"La página '{file_name}' no existe.")
            return
        
        with open(file_path, "r", encoding="utf-8") as file:
            current_code = file.read()
        
        page_type = "Subida" if is_uploaded else "Creada"
        st.subheader(f"✏️ Editar Página {page_type}: {file_name}")
        
        updated_code = st.text_area(
            "Modifica el código de la página", 
            current_code, 
            height=600, 
            key=f"edit_area_{file_name}"
        )
        
        if st.button("💾 Guardar Cambios", key=f"guardar_edit_{file_name}"):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(safe_str_convert(updated_code))
            st.success(f"✅ Página '{file_name}' actualizada correctamente.")
            
    except Exception as e:
        st.error(f"Error al editar la página: {str(e)}")

def get_all_pages():
    """Obtener todas las páginas con manejo de errores mejorado"""
    try:
        created_pages = []
        web_dir = safe_str_convert(WEB_DIR)
        if os.path.exists(web_dir):
            files = os.listdir(web_dir)
            current_file = os.path.basename(__file__)
            created_pages = [
                f for f in files 
                if isinstance(f, str) and f.endswith(".py") and f != current_file
            ]
        
        uploaded_pages = []
        uploaded_dir = safe_str_convert(UPLOADED_DIR)
        if os.path.exists(uploaded_dir):
            files = os.listdir(uploaded_dir)
            uploaded_pages = [
                f for f in files 
                if isinstance(f, str) and f.endswith(".py")
            ]
        
        return created_pages, uploaded_pages
        
    except Exception as e:
        st.error(f"Error al obtener la lista de páginas: {str(e)}")
        return [], []

def safe_selectbox(label, options, **kwargs):
    """Selectbox seguro que siempre devuelve string"""
    try:
        if not options:
            return None
        
        # Asegurar que todas las opciones sean strings
        safe_options = [safe_str_convert(opt) for opt in options if opt is not None]
        safe_options = [opt for opt in safe_options if opt]  # Filtrar strings vacíos
        
        if not safe_options:
            return None
        
        # Usar selectbox normal pero asegurar que el resultado sea string
        result = st.selectbox(label, safe_options, **kwargs)
        return safe_str_convert(result) if result is not None else None
        
    except Exception as e:
        st.error(f"Error en selectbox '{label}': {str(e)}")
        return None

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
        col1.metric("📝 Páginas Creadas", len(created_pages))
        col2.metric("📤 Páginas Subidas", len(uploaded_pages))
        
        tabs = st.tabs(["📝 Crear", "📤 Subir", "🚀 Ejecutar", "✏️ Editar", "🗑️ Eliminar"])
        
        with tabs[0]:  # Crear
            st.subheader("📝 Crear una nueva página")
            new_file_name = st.text_input("Nombre del archivo (sin extensión)", key="crear_nombre")
            code_content = st.text_area("Escribe el código de la nueva página", DEFAULT_CODE, height=400, key="crear_codigo")
            
            if st.button("✨ Crear Página", key="crear_boton"):
                file_name_safe = safe_str_convert(new_file_name)
                if file_name_safe:
                    create_new_page(file_name_safe, code_content)
                    st.rerun()
                else:
                    st.error("❌ Por favor, ingresa un nombre válido.")

        with tabs[1]:  # Subir
            st.subheader("📤 Subir archivo Python")
            uploaded_file = st.file_uploader("Selecciona un archivo Python", type=['py'])
            
            if uploaded_file is not None:
                st.info(f"📄 Archivo seleccionado: {uploaded_file.name}")
                custom_name = st.text_input("Nombre personalizado (opcional)", placeholder="Deja vacío para usar el nombre original")
                
                with st.expander("👀 Vista previa del código"):
                    try:
                        code_preview = uploaded_file.read().decode('utf-8')
                        st.code(code_preview, language='python')
                        uploaded_file.seek(0)
                    except Exception as e:
                        st.error(f"Error al leer el archivo: {str(e)}")

                if st.button("💾 Guardar Archivo", key="subir_boton"):
                    saved_name = save_uploaded_file(uploaded_file, custom_name)
                    if saved_name:
                        st.rerun()

        with tabs[2]:  # Ejecutar
            st.subheader("🚀 Ejecutar Páginas Subidas")
            
            if is_cloud:
                st.warning("⚠️ Funcionalidad limitada en Streamlit Cloud. Usa la opción de descarga.")
            
            if uploaded_pages:
                selected_uploaded = safe_selectbox(
                    "Selecciona una página subida para lanzar", 
                    uploaded_pages, 
                    key="exec_uploaded_select"
                )
                if selected_uploaded:
                    run_uploaded_page(selected_uploaded)
            else:
                st.info("ℹ️ No hay páginas subidas disponibles.")

        with tabs[3]:  # Modificar
            st.subheader("✏️ Modificar páginas")
            edit_tabs = st.tabs(["Editar Creadas", "Editar Subidas"])
            
            with edit_tabs[0]:
                if created_pages:
                    page_to_edit_c = safe_selectbox(
                        "Selecciona página creada para editar", 
                        created_pages, 
                        key="editar_created_select"
                    )
                    if page_to_edit_c:
                        edit_page(page_to_edit_c, is_uploaded=False)
                else:
                    st.info("ℹ️ No hay páginas creadas para editar.")
            
            with edit_tabs[1]:
                if uploaded_pages:
                    page_to_edit_u = safe_selectbox(
                        "Selecciona página subida para editar", 
                        uploaded_pages, 
                        key="editar_uploaded_select"
                    )
                    if page_to_edit_u:
                        edit_page(page_to_edit_u, is_uploaded=True)
                else:
                    st.info("ℹ️ No hay páginas subidas para editar.")

        with tabs[4]:  # Eliminar
            st.subheader("🗑️ Eliminar páginas")
            st.warning("⚠️ Esta acción no se puede deshacer.")
            
            delete_tabs = st.tabs(["Eliminar Creadas", "Eliminar Subidas"])
            
            def delete_ui(page_list, is_uploaded):
                if page_list:
                    page_type = "subida" if is_uploaded else "creada"
                    key_prefix = f"del_{page_type}"
                    
                    page_to_delete = safe_selectbox(
                        f"Selecciona una página {page_type} para eliminar", 
                        page_list, 
                        key=f"{key_prefix}_select"
                    )
                    
                    username = st.text_input("👤 Nombre de usuario", key=f"{key_prefix}_user")
                    password = st.text_input("🔒 Contraseña", type="password", key=f"{key_prefix}_pass")
                    
                    if st.button(f"🗑️ Eliminar Página {page_type.capitalize()}", key=f"{key_prefix}_btn", type="primary"):
                        username_safe = safe_str_convert(username)
                        password_safe = safe_str_convert(password)
                        
                        if username_safe and password_safe and page_to_delete:
                            delete_page(page_to_delete, username_safe, password_safe, is_uploaded)
                            st.rerun()
                        else:
                            st.error("❌ Por favor, completa todos los campos.")
                else:
                    page_type = "subidas" if is_uploaded else "creadas"
                    st.info(f"ℹ️ No hay páginas {page_type} para eliminar.")

            with delete_tabs[0]:
                delete_ui(created_pages, is_uploaded=False)
            with delete_tabs[1]:
                delete_ui(uploaded_pages, is_uploaded=True)
                
    except Exception as e:
        st.error(f"❌ Error general en la aplicación: {str(e)}")
        st.info("💡 Si el problema persiste, verifica la configuración del entorno.")

if __name__ == "__main__":
    app()
