import streamlit as st
import duckdb
import pandas as pd

#DB_NAME = "../data/Tronaduras_vs_Sismicidad.db"
DB_NAME = r"C:\Users\Sergio Arias\Desktop\Python\data\tronaduras_vs_sismicidad.db"


# Opciones de imputación disponibles
IMPUTATION_OPTIONS = [    
    "interpolate", 
    "ffill", 
    "bfill",
    "median", 
    "mode", 
    "constant", 
    "datetime",
    "drop" 
]

IMPUTATION_DESCRIPTIONS = {
    "interpolate": "Interpola los valores faltantes (lineal).",
    "ffill": "Rellena con el valor anterior válido (forward fill).",
    "bfill": "Rellena con el siguiente valor válido (backward fill).",
    "median": "Imputa con la mediana de la columna.",
    "mode": "Imputa con la moda (valor más frecuente).",
    "constant": "Imputa con un valor constante definido en 'Defecto'.",
    "datetime": "Convierte la columna a tipo fecha, si falla usa 'Defecto'.",
    "drop": "Elimina la columna si tiene valores NaN."
}


# Si el espaciado no se ha definido aún, asignar un valor por defecto
if "ROW_SPACING" not in st.session_state:
    st.session_state.ROW_SPACING = 8  # Modificable por el usuario

# Función para cargar variables
def load_variables():
    """Carga las variables del conjunto Tronaduras desde la tabla Raw_Data.Variables_Description."""
    conn = duckdb.connect(DB_NAME)
    query = """
        SELECT "N°", Variable, Descripción, Imputacion, Defecto 
        FROM Raw_Data.Variables_Description
        WHERE Conjunto = 'Tronaduras'
        ORDER BY "N°"
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    return df

# Guardar cambios en la base de datos
def save_changes(changes):
    """Guarda los cambios en la tabla Raw_Data.Variables_Description."""
    conn = duckdb.connect(DB_NAME)
    for change in changes:
        numero = change["N°"]
        new_imputacion = change["new_imputacion"]
        new_defecto = change["new_defecto"]
        update_query = """
            UPDATE Raw_Data.Variables_Description
            SET Imputacion = ?, Defecto = ?
            WHERE "N°" = ?
        """
        conn.execute(update_query, [new_imputacion, new_defecto, numero])
    conn.close()

# Aplicación principal
def app():
    st.title("Modificar Imputación - Tronaduras")
    st.write("Modifica las columnas **Imputación** y **Defecto** para las variables del conjunto Tronaduras.")

    # Línea separadora después de los encabezados
    st.markdown('<hr style="margin-top: 20px; margin-bottom: 10px; border: 1px solid #ddd;">', unsafe_allow_html=True)
    head_cols = st.columns([2, 2, 1])
    # Coloca el texto explicativo en la columna derecha
    with head_cols[0]:
        st.markdown("""
            <div class="explanation-text">
                <b>Tipos de imputación disponibles:</b><br>
                🔄 <b>drop</b>: Elimina la columna si tiene valores NaN.<br>
                🔢 <b>median</b>: Imputa con la mediana de la columna.<br>
                🔀 <b>mode</b>: Imputa con la moda (valor más frecuente).<br>
                🔑 <b>constant</b>: Imputa con un valor constante definido en 'Defecto'.<br>
                📅 <b>datetime</b>: Convierte la columna a tipo fecha, si falla usa 'Defecto'.<br>
                🔄 <b>interpolate</b>: Interpola los valores faltantes (lineal).<br>
                ⬆️ <b>ffill</b>: Rellena con el valor anterior válido (forward fill).<br>
                ⬇️ <b>bfill</b>: Rellena con el siguiente valor válido (backward fill).<br>
            </div>
        """, unsafe_allow_html=True)

    # Línea separadora después de los encabezados
    st.markdown('<hr style="margin-top: 5px; margin-bottom: 50px; border: 1px solid #ddd;">', unsafe_allow_html=True)

    # Cargar las variables correspondientes
    df = load_variables()

    # Lista para guardar los cambios
    changes = []

    # Crear un formulario para modificar variables
    with st.form("form_modificar_variables"):
        # Encabezados de la tabla
        cols = st.columns([1, 3, 4, 2, 2])  
        with cols[0]: st.markdown("**N°**")
        with cols[1]: st.markdown("**Variable**")
        with cols[2]: st.markdown("**Descripción**")
        with cols[3]: st.markdown("**Imputación**")
        with cols[4]: st.markdown("**Defecto**")

        # Línea separadora después de los encabezados
        st.markdown('<hr style="margin-top: 5px; margin-bottom: 10px; border: 1px solid #ddd;">', unsafe_allow_html=True)

        # Mostrar cada variable en una fila horizontal
        for idx, row in df.iterrows():
            cols = st.columns([1, 2, 4, 2, 2])  

            with cols[0]: 
                st.markdown(f"<p class='row-separator'>{row['N°']}</p>", unsafe_allow_html=True)

            with cols[1]: 
                st.markdown(f"<p class='row-separator'>{row['Variable']}</p>", unsafe_allow_html=True)

            with cols[2]: 
                st.markdown(f"<p class='row-separator'>{row['Descripción']}</p>", unsafe_allow_html=True)

            # Columna Imputación (editable con selectbox)
            try:
                default_index = IMPUTATION_OPTIONS.index(row['Imputacion'])
            except ValueError:
                default_index = 0
            with cols[3]:
                st.selectbox(
                    "Imputación",
                    options=IMPUTATION_OPTIONS,
                    index=default_index,
                    key=f"imputacion_{idx}",
                    label_visibility="collapsed"
                )


            # Columna Defecto (editable con text_input)
            with cols[4]:
                st.text_input(
                    "Defecto",
                    value=row['Defecto'] if pd.notna(row['Defecto']) else "",
                    key=f"defecto_{idx}",
                    label_visibility="collapsed"
                )

            # Línea separadora entre filas
            st.markdown('<hr class="row-separator">', unsafe_allow_html=True)

            # Guardar los cambios
            changes.append({
                "N°": row["N°"],
                "new_imputacion": IMPUTATION_OPTIONS[default_index],
                "new_defecto": row["Defecto"]
            })

        # Botón de guardado
        submitted = st.form_submit_button("Guardar cambios")
        if submitted:
            save_changes(changes)
            st.success("Los cambios se han guardado correctamente.")
    

    # Línea separadora después de los encabezados
    st.markdown('<hr style="margin-top: 50px; margin-bottom: 10px; border: 0px solid #ddd;">', unsafe_allow_html=True)

    # Slider al final del formulario, dentro de un contenedor con 3 columnas
    # st.write("### Ajuste de Espaciado")
    foot_cols = st.columns([2, 2, 1])

    # Colocar el slider en la columna central
    with foot_cols[2]:  # Colocamos el slider en la columna central
        new_spacing = st.slider("Ajusta la separación entre filas", 5, 30, st.session_state.ROW_SPACING, key="spacing_slider", help="Usa este control para ajustar la distancia entre filas.")
    
    # Solo actualizar el valor de separación si ha cambiado
    if new_spacing != st.session_state.ROW_SPACING:
        st.session_state.ROW_SPACING = new_spacing

    # Aplicar el espaciado actualizado entre las filas
    st.markdown(f"""
        <style>
            .row-separator {{
                margin-top: {st.session_state.ROW_SPACING}px !important;
                margin-bottom: {st.session_state.ROW_SPACING}px !important;
            }}
        </style>
    """, unsafe_allow_html=True)

