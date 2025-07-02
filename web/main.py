import streamlit as st
import duckdb
import pandas as pd
import os
import streamlit.components.v1 as components

# Configuración de la página
st.set_page_config(layout="wide")



# Función para agregar el fondo interactivo con partículas y formulario de login
def particles_background_and_login_form():
    
    particles_html = """
    <style>
    /* Estilo para el contenedor */
    .container {
        position: relative;
        width: 100vw;
        height: 100vh;
    }

    /* Partículas */
    #particles-js {
        position: absolute;
        top: 0;
        left: 0;
        width: 200%;
        height: 200%;
        transform: translate(-10%, -10%);
    }

    /* Estilo del formulario */
    .form-container {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -40%);
        z-index: 10;  /* El formulario estará encima de las partículas */
        background-color: rgba(255, 255, 255, 0.8);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    h1 {
        text-align: center;
        margin-top: -40px;
    }

    /* Estilo de los inputs */
    input {
        width: 100%;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #ccc;
        border-radius: 5px;
        font-size: 16px;
    }

    button {
        width: 100%;
        padding: 10px;
        background-color: #28a745;
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }

    button:hover {
        background-color: #218838;
    }

    /* Mensaje de error */
    #error-message {
        color: red;
        text-align: center;
        display: none;
    }

    </style>

    <div class="container">
        <!-- Contenedor de partículas -->
        <div id="particles-js"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <script>
    particlesJS("particles-js", {
    
        "particles": {
            "number": {
                "value": 200,
                "density": {
                    "enable": true,
                    "value_area": 800
                }
            },
            "color": {
                "value": "#0000ff"
            },
            "shape": {
                "type": "circle",
                "stroke": {
                    "width": 1,
                    "color": "#000000"
                }
            },
            "opacity": {
                "value": 0.9
            },
            "size": {
                "value": 5,
                "random": true
            },
            "line_linked": {
                "enable": true,
                "distance": 200,
                "color": "#0000ff",
                "opacity": 0.4,
                "width": 1
            },
            "move": {
                "enable": true,
                "speed": 6,
                "direction": "none",
                "out_mode": "out"
            }
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": {
                    "enable": true,
                    "mode": "repulse"
                },
                "onclick": {
                    "enable": true,
                    "mode": "push"
                }
            },
            "modes": {
                "repulse": {
                    "distance": 100
                }
            }
        },
        "retina_detect": true
    });

    // Evita el desplazamiento de la página
    document.body.style.overflow = "hidden";

    </script>
    """
    # Renderiza el fondo con partículas y formulario en Streamlit
    components.html(particles_html, height=1000, width=1900, scrolling=False)


  
# Ruta a la base de datos
DB_NAME = "../data/Tronaduras_vs_Sismicidad.db"


#╭───────────────────────────────────────────────────────────────────────────╮
#│ Función para inicializar la tabla de roles.                               │
#│ Crea la tabla Roles si no existe e inserta roles por defecto.             │                 
#╰───────────────────────────────────────────────────────────────────────────╯
def init_roles_table():
    """ Crea la tabla Roles si no existe e inserta roles por defecto. """
    conn = duckdb.connect(DB_NAME)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Roles (
        role_name VARCHAR PRIMARY KEY,
        allowed_pages VARCHAR
    )
    """
    conn.execute(create_table_query)
    
    df = conn.execute("SELECT COUNT(*) as count FROM Roles WHERE role_name = 'Administrador'").fetchdf()
    if df['count'].iloc[0] == 0:
        insert_query = """
        INSERT INTO Roles (role_name, allowed_pages) VALUES
        ('Administrador', 'todo'),
        ('Usuario', 'tronaduras')
        """
        conn.execute(insert_query)
    conn.close()


#╭───────────────────────────────────────────────────────────────────────────╮
#│ Función para inicializar la tabla de usuarios.                            │
#│ Crea la tabla Usuarios si no existe e inserta usuarios por defecto.       │                
#╰───────────────────────────────────────────────────────────────────────────╯
def init_users_table():
    """ Crea la tabla Usuarios si no existe e inserta usuarios por defecto. """
    conn = duckdb.connect(DB_NAME)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Usuarios (
        username VARCHAR PRIMARY KEY,
        password VARCHAR,
        role VARCHAR
    )
    """
    conn.execute(create_table_query)
    
    df = conn.execute("SELECT COUNT(*) as count FROM Usuarios").fetchdf()
    if df['count'].iloc[0] == 0:
        insert_query = """
        INSERT INTO Usuarios (username, password, role) VALUES
        ('admin', 'admin', 'Administrador'),
        ('usuario', 'usuario', 'Usuario')
        """
        conn.execute(insert_query)
    conn.close()



#╭───────────────────────────────────────────────────────────────────────────╮
#│ Función para inicializar la tabla de páginas.                             │
#│ Crea o actualiza la tabla Paginas con los nombres de los archivos .py     │                  
#╰───────────────────────────────────────────────────────────────────────────╯
def init_pages_table():
    """ Crea o actualiza la tabla Paginas. """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pages = []
    for file in os.listdir(base_dir):
        if file.endswith(".py") and file.lower() != "main.py":
            page_name = file[:-3].lower()
            pages.append((page_name, file, ""))
    
    conn = duckdb.connect(DB_NAME)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Paginas (
        file_name VARCHAR NOT NULL PRIMARY KEY,
        page_name VARCHAR NOT NULL,
        description VARCHAR
    )
    """
    conn.execute(create_table_query)
    
    existing_pages = conn.execute("SELECT file_name FROM Paginas").fetchdf()
    existing_page_names = set(existing_pages['file_name'].tolist())
    new_page_names = set(page[1] for page in pages)
    
    pages_to_remove = existing_page_names - new_page_names
    if pages_to_remove:
        conn.execute(f"DELETE FROM Paginas WHERE file_name IN ({','.join(['?' for _ in pages_to_remove])})", list(pages_to_remove))
    
    pages_to_add = new_page_names - existing_page_names
    for page_name, file_name, description in pages:
        if file_name in pages_to_add:
            conn.execute("INSERT INTO Paginas (page_name, file_name, description) VALUES (?, ?, ?)", [page_name, file_name, description])
    
    conn.close()



#╭───────────────────────────────────────────────────────────────────────────╮
#│ Función para obtener un usuario de la base de datos.                      │
#│ Devuelve un diccionario con el nombre de usuario y el rol.                │                  
#╰───────────────────────────────────────────────────────────────────────────╯
def get_user(username, password):
    """
    Consulta la tabla Usuarios y retorna el usuario (diccionario)
    si las credenciales son válidas.
    """
    conn = duckdb.connect(DB_NAME)
    query = f"""
    SELECT username, role FROM Usuarios
    WHERE username = '{username}' AND password = '{password}'
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    if not df.empty:
        return df.iloc[0].to_dict()
    return None



#╭───────────────────────────────────────────────────────────────────────────╮
#│ Función para obtener las páginas permitidas para un rol.                  │
#│ Devuelve una lista de nombres de páginas (en minúsculas).                 │
#╰───────────────────────────────────────────────────────────────────────────╯
def get_allowed_pages_for_role(role):
    """
    Consulta la tabla Roles para obtener las páginas permitidas
    para el rol indicado. Devuelve una lista de nombres de páginas (en minúsculas).
    """
    conn = duckdb.connect(DB_NAME)
    query = f"""
    SELECT allowed_pages FROM Roles
    WHERE role_name = '{role}'
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    if not df.empty:
        allowed = df['allowed_pages'].iloc[0]
        if allowed == "todo":
            return []  # Administrador tiene acceso a todas las páginas
        return [p.strip().lower() for p in allowed.split(",") if p.strip() != ""]
    return []



#╭───────────────────────────────────────────────────────────────────────────╮
#│ Formulario de login con campos de usuario y contraseña. Al presionar el   │
#│ botón "Ingresar", se verifica si las credenciales son correctas y se      │
#│ almacena el usuario en la sesión.                                         │
#╰───────────────────────────────────────────────────────────────────────────╯
def login():
    # Renderiza el fondo de partículas (se mantiene igual)
    particles_background_and_login_form()

    st.markdown(
    """
    <style>
    /* 
      .block-container:
      - Define el contenedor principal de la app.
      - 'position: relative' permite que los elementos hijos con posición absoluta se ubiquen respecto a él.
      - 'height: 100vh' hace que el contenedor ocupe el 100% de la altura de la ventana (viewport).
      - Alternativas para 'height': 100%, 500px, etc.
    */
    .block-container {
        position: relative;
        height: 100vh;
    }
    /* 
      .stForm:
      - Es el contenedor del formulario.
      - 'position: absolute' permite posicionarlo de forma personalizada.
      - 'top: 30%' ubica el formulario al 30% de la altura del contenedor (ajustable: 20%, 50%, etc.).
      - 'left: 50%' centra el formulario horizontalmente.
      - 'transform: translate(-50%, -50%)' corrige la posición para que quede centrado.
      - 'background-color: rgba(255, 255, 255, 0.8)' define un fondo blanco con 80% de opacidad.
           Alternativas: rgba(0, 0, 0, 0.5) para un fondo negro semitransparente.
      - 'padding: 30px' establece el espaciado interno; se puede ajustar a 20px, 1em, etc.
      - 'border-radius: 15px' redondea las esquinas; 0px para sin redondear o 50% para formas circulares.
      - 'box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3)' agrega una sombra; el formato es: [offset-x] [offset-y] [blur-radius] [color].
      - 'width: 500px' fija el ancho del formulario; se puede usar 100% para ocupar todo el contenedor.
      - 'z-index: 10' asegura que el formulario esté por encima de otros elementos.
    */
    .stForm {
        position: absolute;
        top: 30%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: rgba(255, 255, 255, 0.8);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        width: 500px;
        z-index: 10;
    }
    /* 
      Botón de envío del formulario:
      - Se usa el selector que apunta al botón dentro del div identificado con data-testid="stFormSubmitButton".
      - 'background-color: #007bff' establece un color azul (alternativas: #28a745 para verde, #dc3545 para rojo, #ffc107 para amarillo).
      - 'border: none' remueve el borde; también se puede definir por ejemplo '1px solid #000' para un borde negro.
      - 'color: white' fija el color del texto a blanco; puedes usar 'black', '#fff', etc.
      - 'padding: 10px 24px' define el espacio interno vertical (10px) y horizontal (24px); otras opciones: 8px 16px, 12px 20px.
      - 'font-size: 16px' define el tamaño de la fuente; alternativas: 14px, 18px, etc.
      - 'border-radius: 5px' redondea las esquinas del botón; 0px para sin redondeo o 50% para botón circular.
      - 'width: 100%' hace que el botón ocupe el ancho completo del contenedor.
      - Se usa '!important' para forzar el estilo sobre cualquier otro definido.
    */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #007bff !important;
        border: none !important;
        color: white !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
        border-radius: 5px !important;
        width: 100% !important;
    }
    /* 
      Labels de los inputs dentro del formulario:
      - Se apunta a los elementos <label> que están dentro del contenedor con data-testid="stForm".
      - 'color: #ff5722' define el color del texto, en este caso un naranja. Alternativas: #000000 (negro), #ffffff (blanco), o valores RGB (ej., rgb(255,87,34)).
      - 'font-size: 18px' establece el tamaño de la fuente; se puede modificar a 14px, 16px, 20px, etc.
      - 'font-style: italic' aplica cursiva; las opciones son 'normal', 'italic' o 'oblique'.
      - 'font-weight: bold' define un peso fuerte; también se pueden usar valores numéricos (100-900) o 'normal'.
      - 'border: 1px solid #ccc' añade un borde alrededor del label; el formato es: [ancho] [estilo] [color].
      - 'padding: 4px 8px' agrega espacio interno; por ejemplo, 2px 4px o 6px 12px.
      - 'border-radius: 5px' redondea los bordes; puede ajustarse a 0px para bordes cuadrados o mayor para más redondeo.
    */
    [data-testid="stForm"] label {
        color:  #333333 !important;          /* Color del texto */
        font-size: 20px !important;          /* Tamaño de la fuente */
        font-style: italic !important;       /* Estilo: normal, italic o oblique */
        font-weight: 900 !important;        /* Grosor: normal, bold o numérico (100-900) */
        border: 0px solid #ccc !important;   /* Borde: 1px solid #ccc */
        padding: 4px 8px !important;         /* Espaciado interno: 4px arriba/abajo, 8px izquierda/derecha */
        border-radius: 5px !important;       /* Bordes redondeados: 5px */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    # Título principal de la aplicación, posicionado usando margen negativo para desplazarlo
    st.markdown(
        """<h1 style="text-align:center; margin-top: -1000px; color: #000000;">Kepler Brain - Sismicidad</h1>""",
        unsafe_allow_html=True
    )

    with st.form(key='login_form'):
        # Input para el nombre de usuario
        username = st.text_input("Usuario:")
        # Input para la contraseña, el tipo "password" oculta el texto ingresado
        password = st.text_input("Contraseña:", type="password")
        # Botón de envío del formulario
        submit = st.form_submit_button("Ingresar")
                
    # Procesamiento del formulario de login en el servidor
    if submit:
        user = get_user(username, password)
        if user is not None:
            st.session_state.user = user
            st.rerun()  # Recarga la app para mostrar la navegación
        else:
            st.error("Credenciales inválidas")





#╭───────────────────────────────────────────────────────────────────────────╮
#│ Función de logout                                                         │
#│ Cierra la sesión y redirige al login.                                     │
#╰───────────────────────────────────────────────────────────────────────────╯
def logout():
    del st.session_state.user
    st.rerun()

# Inicializar las tablas (Usuarios, Roles y Paginas)
init_roles_table()
init_users_table()
init_pages_table()

# Proceso de autenticación y navegación
if "user" not in st.session_state:
    login()

else:
    user = st.session_state.user
    st.sidebar.title("Navegación")
    
    conn = duckdb.connect(DB_NAME)
    paginas_df = conn.execute("SELECT * FROM Paginas").fetchdf()
    conn.close()
    
    allowed_pages_list = get_allowed_pages_for_role(user["role"])
    
    # Construir el menú de navegación (incluye "Home")
    pages = {"Home": "main.py"}
    for _, row in paginas_df.iterrows():
        page_name = row["page_name"].lower()
        file_name = row["file_name"]
        if page_name in allowed_pages_list or user["role"] == "Administrador":
            pages[page_name.capitalize()] = file_name
    
    selection = st.sidebar.radio("Ir a", list(pages.keys()))
    
    if st.sidebar.button("Cerrar sesión"):
        logout()

    st.title("Kepler-Sismicidad")
    st.write(f"Bienvenido, **{user['username']}**. Rol: **{user['role']}**")
    
    if selection == "Home":
        st.write("Esta es la página principal de Kepler-Sismicidad. Seleccione una vista en la barra lateral para modificar los datos.")
    else:
        # Importación dinámica de la vista seleccionada (cada módulo debe tener una función app())
        module_name = pages[selection].replace(".py", "")
        try:
            mod = __import__(module_name)
            mod.app()
        except Exception as e:
            st.error(f"Error al cargar la página {selection}: {e}")
