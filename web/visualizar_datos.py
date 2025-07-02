import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from datetime import datetime

# ──────────────────────────────────────────────
# Configuración de rutas e imports
# ──────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from classes.DuckDB_Helper_v02 import DuckDBHelper 
from classes.Tronaduras_File_Reader_v03 import TronadurasFileReader

# ──────────────────────────────────────────────
# Parámetros de conexión
# ──────────────────────────────────────────────
DB_NAME = os.path.join(project_root, "data", "Tronaduras_vs_Sismicidad.db")
SCHEMA_NAME = "Processed_Data"

def initialize_session_state():
    """Inicializa las variables necesarias en session_state"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df_sismicidad = None
        st.session_state.df_tronaduras = None
        st.session_state.df_Tabla_Unificada = None

@st.cache_data(show_spinner=False)
def load_data_from_db():
    """Función caché para cargar datos desde la base de datos"""
    try:
        db_helper = DuckDBHelper(DB_NAME)
        
        datos = {
            'sismicidad': db_helper.select_df(table="Sismicidad", schema=SCHEMA_NAME),
            'tronaduras': db_helper.select_df(table="Tronaduras", schema=SCHEMA_NAME),
            'unificada': db_helper.select_df(table="Tabla_Unificada", schema=SCHEMA_NAME)
        }
        
        db_helper.close_connection()
        return datos
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

def show_plots_section():
    """Muestra la sección de gráficos con los datos cargados"""
    st.markdown("### Gráficos")
    
    # Checkboxes para selección de gráficos (solo se muestran si están en graficos_habilitados)
    show_mo_chart = st.checkbox("Gráfico de Momento Sísmico (Mo) vs Tiempo", value=False)
    show_energy_chart = st.checkbox("Gráfico de Energía vs Tiempo", value=False)
    show_magnitude_chart = st.checkbox("Gráfico de Magnitud vs Tiempo", value=False)
    show_histogram_magnitude = st.checkbox("Histograma de Magnitudes", value=False)
    show_scatter_correlation = st.checkbox("Correlación: Mo vs Energía", value=False)
    show_correlation_heatmap = st.checkbox("Matriz de Correlación", value=False, disabled=True)
    show_scatter_matrix = st.checkbox("Matriz de Dispersión", value=False, disabled=True)
    show_boxplots = st.checkbox("Boxplots", value=False, disabled=True)
    show_kg_vs_Mo_cumulative = st.checkbox("Gráfico de Mo_cumulative vs Explosivos", value=False, disabled=False)
    show_tronadura_3d = st.checkbox("Gráfico 3D de ubicación de tronaduras", value=False, disabled=False)

    
    st.markdown("---")

    metric_options = {
        'Mo': 'Mo',
        'Momento Sísmico Acumulado (Mo_cumulative)': 'Mo_cumulative', 
        'Energía (Energy [J])': 'Energy [J]',
        'Energía Acumulativa (Energy_cumulative)': 'Energy_cumulative'
    }
    selected_metric = st.radio(
        "Seleccione la métrica a visualizar:",
        list(metric_options.keys()),
        index=0
    )
    selected_metric_col = metric_options[selected_metric]

    show_shot_grouped_mo = st.checkbox("Mo vs Tiempo por N° Disparo", value=False)  # Nueva opción
    show_all_shots_combined = st.checkbox("Mo vs Tiempo todos disparos juntos", value=False)  # Nueva opción


    # ──────────────────────────────────────────────
    # Generación de gráficos
    # ──────────────────────────────────────────────
    try:
        df_sismicidad = st.session_state.df_sismicidad
        df_tronaduras = st.session_state.df_tronaduras
        df_Tabla_Unificada = st.session_state.df_Tabla_Unificada


        #╭──────────────────────────────────────────────────────╮
        #| Gráfico de Momento Sísmico (Mo) vs Tiempo            |
        #╰──────────────────────────────────────────────────────╯
        if show_mo_chart:
            with st.spinner("Generando gráfico de Momento Sísmico..."):
                fig_mo = go.Figure()
                fig_mo.add_trace(go.Scatter(
                    x=df_sismicidad['Sismicidad - Date Time'],
                    y=df_sismicidad['Mo'],
                    mode='lines',
                    name='Mo'
                ))
                fig_mo.update_layout(
                    title='Momento Sísmico (Mo) vs Tiempo',
                    xaxis_title='Fecha',
                    yaxis_title='Mo'
                )
                st.plotly_chart(fig_mo, use_container_width=True)

        #╭──────────────────────────────────────────────────────╮
        #| Gráfico de Energía vs Tiempo                         |
        #╰──────────────────────────────────────────────────────╯
        if show_energy_chart:
            with st.spinner("Generando gráfico de Energía..."):
                fig_energy = px.line(
                    df_sismicidad,
                    x='Sismicidad - Date Time',
                    y='Energy [J]',
                    title='Energía vs Tiempo'
                )
                st.plotly_chart(fig_energy, use_container_width=True)

        #╭──────────────────────────────────────────────────────╮
        #| Gráfico de Magnitud vs Tiempo                         |
        #╰──────────────────────────────────────────────────────╯
        if show_magnitude_chart:
            with st.spinner("Generando gráfico de Magnitud..."):
                fig_magnitude = px.line(
                    df_sismicidad,
                    x='Sismicidad - Date Time',
                    y='Local Magnitude',
                    title='Magnitud vs Tiempo'
                )
                st.plotly_chart(fig_magnitude, use_container_width=True)

        #╭──────────────────────────────────────────────────────╮
        #| Gráfico de Distribución de Magnitudes                |
        #╰──────────────────────────────────────────────────────╯
        if show_histogram_magnitude:
            with st.spinner("Generando histograma..."):
                fig_hist = px.histogram(
                    df_sismicidad,
                    x='Local Magnitude',
                    nbins=20,
                    title='Distribución de Magnitudes'
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        #╭──────────────────────────────────────────────────────╮
        #| Gráfico de Mo vs. Energía                            |
        #╰──────────────────────────────────────────────────────╯
        if show_scatter_correlation:
            with st.spinner("Generando correlación..."):
                fig_scatter = px.scatter(
                    df_sismicidad,
                    x='Mo',
                    y='Energy [J]',
                    title='Mo vs Energía',
                    trendline="ols"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        #╭──────────────────────────────────────────────────────╮
        #| Matriz de Correlación                                |
        #╰──────────────────────────────────────────────────────╯
        if show_correlation_heatmap:
            with st.spinner("Generando matriz de correlación..."):
                corr_matrix = df_sismicidad.corr(numeric_only=True)
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Correlación",
                    height=1000  # Aumenta la altura del gráfico
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

        #╭──────────────────────────────────────────────────────╮
        #| Matriz de Dispersión                                 |
        #╰──────────────────────────────────────────────────────╯
        if show_scatter_matrix:
            with st.spinner("Generando matriz de dispersión..."):
                num_df = df_sismicidad.select_dtypes(include=np.number)
                fig_matrix = px.scatter_matrix(
                    num_df,
                    title="Matriz de Dispersión"
                )
                st.plotly_chart(fig_matrix, use_container_width=True)

        #╭──────────────────────────────────────────────────────╮
        #| Gráfico de Distribución de Magmitudes Mo - Boxplot   |
        #╰──────────────────────────────────────────────────────╯
        if show_boxplots:
            with st.spinner("Generando boxplots..."):
                col1, col2 = st.columns(2)
                with col1:
                    fig_box1 = px.box(
                        df_sismicidad,
                        y="Local Magnitude",
                        title="Distribución de Magnitudes"
                    )
                    st.plotly_chart(fig_box1, use_container_width=True)
                with col2:
                    fig_box2 = px.box(
                        df_sismicidad,
                        y="Mo",
                        title="Distribución de Mo"
                    )
                    st.plotly_chart(fig_box2, use_container_width=True)


        #╭──────────────────────────────────────────────────────╮
        #| Gráficos agrupados por N° Disparo - Separados        |
        #╰──────────────────────────────────────────────────────╯
        if show_shot_grouped_mo:
            with st.spinner(f"Generando gráficos por N° Disparo para {selected_metric}..."):
                grouped = df_sismicidad.groupby("N° Disparo")
                
                num_cols = 6
                num_shots = len(grouped)
                
                num_rows = (num_shots // num_cols) + (1 if num_shots % num_cols != 0 else 0)
                cols = st.columns(num_cols)
                
                for idx, (shot_id, group) in enumerate(grouped):
                    col = cols[idx % num_cols]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=group['Sismicidad - Date Time'],
                        y=group[selected_metric_col],
                        mode='lines',
                        name=f'Disparo #{shot_id}'
                    ))
                    fig.update_layout(
                        title=f"{selected_metric} vs Tiempo - Disparo #{shot_id}",
                        xaxis_title='Fecha',
                        yaxis_title=selected_metric
                    )
                    
                    with col:
                        st.plotly_chart(fig, use_container_width=True)


        #╭──────────────────────────────────────────────────────╮
        #| Gráficos agrupados por N° Disparo - Combinados       |
        #╰──────────────────────────────────────────────────────╯
        if show_all_shots_combined:
            with st.spinner(f"Generando gráfico combinado para {selected_metric}..."):
                # Filtramos solo las filas con "N° Disparo" numérico
                filtered_df = df_sismicidad[pd.to_numeric(df_sismicidad["N° Disparo"], errors='coerce').notna()]
                
                fig_combined = px.line(
                    filtered_df,
                    x='Sismicidad - Date Time',
                    y=selected_metric_col,
                    color='N° Disparo',
                    title=f"{selected_metric} vs Tiempo - Todos los disparos juntos",
                    labels={'N° Disparo': 'Número de Disparo'}
                )
                
                fig_combined.update_traces(
                    line=dict(width=2),
                    opacity=0.8
                )
                
                fig_combined.update_layout(
                    xaxis_title='Fecha',
                    yaxis_title=selected_metric,
                    legend_title='Número de Disparo',
                    hovermode="x unified",  # Muestra todas las líneas en el hover
                )
                
                st.plotly_chart(fig_combined, use_container_width=True)


        #╭──────────────────────────────────────────────────────╮
        #| Gráfico de Mo_cumulative vs Explosivos Tronadura     |
        #╰──────────────────────────────────────────────────────╯
        if show_kg_vs_Mo_cumulative:
            with st.spinner("Generando gráfico de relación Mo Acumulado vs Explosivos..."):
                # Filtrar solo filas con valores numéricos en las columnas requeridas
                filtered_df = df_Tabla_Unificada[
                    pd.to_numeric(df_Tabla_Unificada["Mo_cumulative"], errors='coerce').notna() &
                    pd.to_numeric(df_Tabla_Unificada["Tronadura_Kg. de explosivos tronadura"], errors='coerce').notna()
                ]
                
                if not filtered_df.empty:
                    # Ordenar por "Tronadura_Kg. de explosivos tronadura" en orden ascendente
                    sorted_df = filtered_df.sort_values(
                        by="Tronadura_Kg. de explosivos tronadura",
                        ascending=True
                    )
                    
                    # Crear el gráfico de dispersión con tendencia
                    fig_tronadura = px.scatter(
                        sorted_df,
                        x="Mo_cumulative",
                        y="Tronadura_Kg. de explosivos tronadura",
                        #trendline="ols",
                        # Se corrigió: eliminamos `formular` y `kwargs`
                        title="Relación entre Momento Sísmico Acumulado (Mo) y Explosivos de Tronadura"
                    )
                    
                    # Mejorar la apariencia del gráfico
                    fig_tronadura.update_layout(
                        plot_bgcolor="white",
                        xaxis_gridcolor="lightgrey",
                        yaxis_gridcolor="lightgrey",
                        xaxis_title="Momento Sísmico Acumulado (Mo)",
                        yaxis_title="Explosivos de Tronadura (kg)",
                        #trendline_color_override="red"
                    )
                    
                    # Mostrar el gráfico
                    st.plotly_chart(fig_tronadura, use_container_width=True)
                else:
                    st.warning("No hay datos suficientes para generar este gráfico.")


        #╭──────────────────────────────────────────────────────╮
        #| Gráfico 3D de ubicación de tronaduras                |
        #╰──────────────────────────────────────────────────────╯
         # Nueva sección para gráfico 3D de tronaduras
        if show_tronadura_3d:
            with st.spinner("Generando gráfico 3D de ubicación de tronaduras..."):
                # Convertir columnas a numéricas, omitiendo valores no numéricos
                df_tronaduras['Coordenadas_Norte (m)'] = pd.to_numeric(df_tronaduras['Coordenadas_Norte (m)'], errors='coerce')
                df_tronaduras['Coordenadas_Este (m)'] = pd.to_numeric(df_tronaduras['Coordenadas_Este (m)'], errors='coerce')
                df_tronaduras['Coordenadas_Cota (m)'] = pd.to_numeric(df_tronaduras['Coordenadas_Cota (m)'], errors='coerce')
                
                # Agrupar por 'N° Disparo'
                grouped = df_tronaduras.groupby("N° Disparo")
                
                # Crear el gráfico 3D de dispersión
                fig_3d = px.scatter_3d(
                    df_tronaduras,
                    x='Coordenadas_Este (m)',
                    y='Coordenadas_Norte (m)',
                    z='Coordenadas_Cota (m)',
                    color='N° Disparo',
                    title='Ubicación Georreferenciada de Tronaduras',
                    labels={
                        'Coordenadas_Este (m)': 'Este (metros)',
                        'Coordenadas_Norte (m)': 'Norte (metros)',
                        'Coordenadas_Cota (m)': 'Cota (metros)',
                        'N° Disparo': 'Número de Disparo'
                    }
                )
                
                fig_3d.update_traces(
                    marker=dict(
                        size=5,
                        opacity=0.8,
                    )
                )
                
                fig_3d.update_layout(
                    height=800,
                    margin=dict(r=20, l=20, t=40, b=20),
                    scene=dict(
                        xaxis_visible=True,
                        yaxis_visible=True,
                        zaxis_visible=True
                    )
                )
                
                # Mostrar el gráfico
                st.plotly_chart(fig_3d, use_container_width=True)



    except KeyError as e:
        st.error(f"Error al generar gráfico: Columna no encontrada - {str(e)}")
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")



def save_changes(table_name, edited_df):
    # Conexión a la base de datos
    conn = duckdb.connect(DB_NAME)
    try:
        # Borramos los datos existentes
        conn.execute(f"DELETE FROM {table_name}")
        
        # Reinsertamos los datos editados
        # Conversión del DataFrame a SQL
        conn.execute("BEGIN;")
        edited_df.to_sql(table_name, conn, if_exists='append', index=False)
        conn.execute("COMMIT;")
        
        st.success("Cambios guardados correctamente en la base de datos.")
    except Exception as e:
        conn.execute("ROLLBACK;")
        st.error(f"Error al guardar los cambios: {str(e)}")
    finally:
        conn.close()



def app():
    initialize_session_state()
    
    # ──────────────────────────────────────────────
    # 1. Sección de carga de datos
    # ──────────────────────────────────────────────
    st.title("Análisis de Sismicidad y Tronaduras")
    
    datos = None

    col1, col2, col3, col4 = st.columns([1,1,1,5])
    with col1:
        if st.button("🎯 Cargar Datos"):
            with st.spinner("Cargando datos desde la base de datos..."):
                datos = load_data_from_db()
    with col2:           
        if st.button("❌ Limpiar Datos"):
            st.session_state.data_loaded = False

    cols = st.columns([6,1,1])
    if datos:
        st.session_state.df_sismicidad = datos['sismicidad']
        st.session_state.df_tronaduras = datos['tronaduras']
        st.session_state.df_Tabla_Unificada = datos['unificada']
        st.session_state.data_loaded = True
        with cols[0]:
            st.success("✅ Datos cargados correctamente!")
            with st.expander("Ver datos de Sismicidad"):
                st.dataframe(st.session_state.df_sismicidad, height=250)
            
            with st.expander("Ver datos de Tronaduras"):
                st.dataframe(st.session_state.df_tronaduras, height=250)
            
            with st.expander("Ver datos de Tabla Unificada"):
                st.dataframe(st.session_state.df_Tabla_Unificada, height=250)

    st.markdown("---")

    # ──────────────────────────────────────────────
    # 2. Sección de visualización
    # ──────────────────────────────────────────────
    if st.session_state.data_loaded:
        show_plots_section()
    else:
        st.warning("⚠️ Por favor, carga los datos primero usando el botón 'Cargar Datos'")


    # ──────────────────────────────────────────────
    # 3. Sección de visualización/editación de datos
    # ──────────────────────────────────────────────
    st.markdown("### Datos de Tronaduras (TablaEditable)")

    # Menú desplegable para seleccionar la tabla
    with st.expander("Seleccionar tabla para editar", expanded=True):
        table_options = ["Tronaduras", "Sismicidad", "Tabla Unificada"]
        selected_table = st.selectbox("Seleccione la tabla que desea editar:", table_options, index=0)

    # Muestra la tabla seleccionada
    if st.session_state.data_loaded:
        if selected_table == "Tronaduras":
            current_df = st.session_state.df_tronaduras
        elif selected_table == "Sismicidad":
            current_df = st.session_state.df_sismicidad
        else:
            current_df = st.session_state.df_Tabla_Unificada
    
        with st.expander("Tabla editable"):
            # Utilizamos st.experimental_data_editor para permitir la edición
            edited_df = st.data_editor(
                current_df,
                key=f"editor_{selected_table}",
                num_rows='fixed',
                column_config=None
            )
        
        # Botón para guardar cambios
        col_save = st.columns([3, 1])
        with col_save[1]:
            if st.button("💾 Guardar cambios"):
                save_changes(selected_table, edited_df)



if __name__ == "__main__":
    app()