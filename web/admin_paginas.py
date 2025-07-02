import streamlit as st
import os
from datetime import datetime
import duckdb
import subprocess # <--- AÑADIDO
import sys        # <--- AÑADIDO

WEB_DIR = "./"
# La ruta absoluta puede causar problemas si mueves el proyecto. 
# Es mejor usar rutas relativas, pero si es necesario, asegúrate que la ruta sea correcta.
DB_NAME = r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"
#DB_NAME = "../data/Tronaduras_vs_Sismicidad.db"

# Directorio para archivos subidos
# Usas una ruta relativa, asegúrate de correr el script desde el directorio correcto.
# Por ejemplo, desde "C:\Users\Sergio Arias\Desktop\Python\web\"
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
"""

# Función para crear una nueva página (sin cambios)
def create_new_page(file_name, code_content):
    if not file_name.endswith(".py"):
        file_name += ".py"
    file_name = file_name.replace(" ", "_")
    if os.path.exists(os.path.join(WEB_DIR, file_name)):
        st.error(f"El archivo '{file_name}' ya existe. Por favor, elige otro nombre.")
        return
    os.makedirs(WEB_DIR, exist_ok=True)
    with open(os.path.join(WEB_DIR, file_name), "w", encoding="utf-8") as file:
        file.write(code_content)
    st.success(f"Página '{file_name}' creada correctamente en el directorio '{WEB_DIR}'.")

# Función para guardar archivo subido (sin cambios)
def save_uploaded_file(uploaded_file, custom_name=None):
    os.makedirs(UPLOADED_DIR, exist_ok=True)
    if custom_name:
        if not custom_name.endswith(".py"):
            custom_name += ".py"
        file_name = custom_name.replace(" ", "_")
    else:
        file_name = uploaded_file.name
    file_path = os.path.join(UPLOADED_DIR, file_name)
    if os.path.exists(file_path):
        st.error(f"El archivo '{file_name}' ya existe. Por favor, elige otro nombre.")
        return None
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Archivo '{file_name}' subido correctamente.")
        return file_name
    except Exception as e:
        st.error(f"Error al guardar el archivo: {str(e)}")
        return None

# ######################################################################
# ### FUNCIÓN MODIFICADA PARA LANZAR EN UNA NUEVA VENTANA ###
# ######################################################################

def run_uploaded_page(file_name):
    """
    Lanza la página seleccionada como un nuevo proceso de Streamlit,
    abriendo una nueva pestaña en el navegador.
    """
    file_path = os.path.join(UPLOADED_DIR, file_name)
    if not os.path.exists(file_path):
        st.error(f"El archivo '{file_name}' no existe.")
        return
    
    st.subheader(f"Lanzar Aplicación: {file_name}")
    st.info(f"""
    Estás a punto de ejecutar el script **{file_name}** como una aplicación independiente.
    - Se abrirá en una **nueva pestaña** del navegador.
    - Se ejecutará en un puerto diferente (ej: 8502, 8503, ...).
    """)

    if st.button(f"🚀 Lanzar '{file_name}' en una nueva ventana", key=f"launch_{file_name}"):
        try:
            # Usar sys.executable asegura que se use el mismo intérprete de Python
            # que está corriendo el panel de administración.
            python_executable = sys.executable
            command = [python_executable, "-m", "streamlit", "run", file_path]
            
            # Popen ejecuta el comando en un nuevo proceso sin bloquear la aplicación principal.
            subprocess.Popen(command)
            
            st.success(f"✅ Se ha enviado la orden para iniciar '{file_name}'.")
            st.warning("Revisa tu navegador, se debería haber abierto una nueva pestaña. "
                       "Para detener esa aplicación, cierra la nueva ventana de terminal que se pudo haber abierto.")

        except FileNotFoundError:
            st.error(
                "Error: No se pudo encontrar 'streamlit'. Asegúrate de que Streamlit "
                "esté instalado en tu entorno de Python. Intenta ejecutar "
                "`pip install streamlit` en tu terminal."
            )
        except Exception as e:
            st.error(f"Ocurrió un error al intentar lanzar el proceso: {e}")
            with st.expander("Ver detalles del error"):
                st.code(str(e))

# ######################################################################
# ### EL RESTO DEL CÓDIGO PERMANECE IGUAL ###
# ######################################################################

def clean_code_for_execution(code_content):
    lines = code_content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if (stripped_line.startswith('st.set_page_config(') or 
            stripped_line.startswith('import streamlit as st') or
            stripped_line.startswith('from streamlit') or
            'webbrowser.open' in line or
            'subprocess.run' in line or
            'os.system' in line):
            cleaned_lines.append(f"# {line}  # Comentado para evitar conflictos")
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def validate_code_syntax(code_content):
    try:
        compile(code_content, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Error de sintaxis en línea {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error de compilación: {str(e)}"

def delete_page(file_name, username, password, is_uploaded=False):
    try:
        conn = duckdb.connect(DB_NAME)
        query = "SELECT password, role FROM Usuarios WHERE username = ?"
        df = conn.execute(query, [username]).fetchdf()
        conn.close()
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return

    if not df.empty and df['role'].iloc[0] == 'Administrador' and df['password'].iloc[0] == password:
        directory = UPLOADED_DIR if is_uploaded else WEB_DIR
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            page_type = "subida" if is_uploaded else "creada"
            st.success(f"Página {page_type} '{file_name}' eliminada exitosamente.")
        else:
            st.error(f"La página '{file_name}' no existe.")
    else:
        st.error("Credenciales incorrectas o no tienes permisos de administrador.")

def edit_page(file_name, is_uploaded=False):
    directory = UPLOADED_DIR if is_uploaded else WEB_DIR
    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        st.error(f"La página '{file_name}' no existe.")
        return
    
    with open(file_path, "r", encoding="utf-8") as file:
        current_code = file.read()
    
    page_type = "Subida" if is_uploaded else "Creada"
    st.subheader(f"Editar Página {page_type}: {file_name}")
    updated_code = st.text_area("Modifica el código de la página", current_code, height=600, key=f"edit_area_{file_name}")
    
    if st.button("Guardar Cambios", key=f"guardar_edit_{file_name}"):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(updated_code)
        st.success(f"Página {page_type.lower()} '{file_name}' actualizada correctamente.")

def get_all_pages():
    created_pages = []
    if os.path.exists(WEB_DIR):
        created_pages = [f for f in os.listdir(WEB_DIR) if f.endswith(".py") and f != os.path.basename(__file__)]
    
    uploaded_pages = []
    if os.path.exists(UPLOADED_DIR):
        uploaded_pages = [f for f in os.listdir(UPLOADED_DIR) if f.endswith(".py")]
    
    return created_pages, uploaded_pages

def app():
    st.title("Gestión de Páginas o Vistas")
    
    created_pages, uploaded_pages = get_all_pages()
    
    col1, col2 = st.columns(2)
    col1.metric("Páginas Creadas", len(created_pages))
    col2.metric("Páginas Subidas", len(uploaded_pages))
    
    tabs = st.tabs(["Crear Página", "Subir Archivo", "Ejecutar Páginas", "Modificar Página", "Eliminar Página"])
    
    with tabs[0]: # Crear
        st.subheader("Crear una nueva página")
        new_file_name = st.text_input("Nombre del archivo (sin extensión)", key="crear_nombre")
        code_content = st.text_area("Escribe el código de la nueva página", DEFAULT_CODE, height=400, key="crear_codigo")
        if st.button("Crear Página", key="crear_boton"):
            if new_file_name:
                create_new_page(new_file_name, code_content)
                st.rerun()
            else:
                st.error("Por favor, ingresa un nombre de archivo válido.")

    with tabs[1]: # Subir
        st.subheader("Subir archivo Python")
        uploaded_file = st.file_uploader("Selecciona un archivo Python", type=['py'])
        
        if uploaded_file is not None:
            st.info(f"Archivo seleccionado: {uploaded_file.name}")
            custom_name = st.text_input("Nombre personalizado (opcional)", placeholder="Deja vacío para usar el nombre original")
            
            with st.expander("Vista previa del código"):
                code_preview = uploaded_file.read().decode('utf-8')
                st.code(code_preview, language='python')
                uploaded_file.seek(0)

            if st.button("Guardar Archivo", key="subir_boton"):
                saved_name = save_uploaded_file(uploaded_file, custom_name)
                if saved_name:
                    st.rerun()

    with tabs[2]: # Ejecutar
        st.subheader("Ejecutar Páginas Subidas")
        st.warning("La ejecución de páginas creadas directamente no está implementada. Solo se pueden lanzar las páginas subidas.")
        
        if uploaded_pages:
            selected_uploaded = st.selectbox("Selecciona una página subida para lanzar", uploaded_pages, key="exec_uploaded_select")
            # La lógica de ejecución ahora está encapsulada en la función
            run_uploaded_page(selected_uploaded)
        else:
            st.info("No hay páginas subidas disponibles para ejecutar.")

    with tabs[3]: # Modificar
        st.subheader("Modificar una página existente")
        edit_tabs = st.tabs(["Editar Creadas", "Editar Subidas"])
        
        with edit_tabs[0]:
            if created_pages:
                page_to_edit_c = st.selectbox("Selecciona página creada para editar", created_pages, key="editar_created_select")
                edit_page(page_to_edit_c, is_uploaded=False)
            else:
                st.info("No hay páginas creadas para editar.")
        
        with edit_tabs[1]:
            if uploaded_pages:
                page_to_edit_u = st.selectbox("Selecciona página subida para editar", uploaded_pages, key="editar_uploaded_select")
                edit_page(page_to_edit_u, is_uploaded=True)
            else:
                st.info("No hay páginas subidas para editar.")

    with tabs[4]: # Eliminar
        st.subheader("Eliminar una página")
        delete_tabs = st.tabs(["Eliminar Creadas", "Eliminar Subidas"])
        
        # Implementación unificada para eliminar
        def delete_ui(page_list, is_uploaded):
            if page_list:
                page_type = "subida" if is_uploaded else "creada"
                key_prefix = f"del_{page_type}"
                
                page_to_delete = st.selectbox(f"Selecciona una página {page_type} para eliminar", page_list, key=f"{key_prefix}_select")
                
                username = st.text_input("Nombre de usuario", key=f"{key_prefix}_user")
                password = st.text_input("Contraseña", type="password", key=f"{key_prefix}_pass")
                
                if st.button(f"Eliminar Página {page_type.capitalize()}", key=f"{key_prefix}_btn"):
                    if username and password:
                        delete_page(page_to_delete, username, password, is_uploaded=is_uploaded)
                        st.rerun()
                    else:
                        st.error("Por favor, ingresa el nombre de usuario y la contraseña.")
            else:
                st.info(f"No hay páginas de tipo '{page_type}' para eliminar.")

        with delete_tabs[0]:
            delete_ui(created_pages, is_uploaded=False)
        with delete_tabs[1]:
            delete_ui(uploaded_pages, is_uploaded=True)

# Llama a la aplicación principal
if __name__ == "__main__":
    app()
