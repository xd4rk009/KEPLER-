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
# Extracción de parámetros de conexión a la base de datos
# ──────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Parámetros de conexión (extraídos automáticamente, no se muestran en la interfaz)
DB_NAME = os.path.join(project_root, "data", "Tronaduras_vs_Sismicidad.db")
SCHEMA_NAME = "Raw_Data"
CONJUNTO_ESPECIFICO = "Tronaduras"

# Importar las clases necesarias
from classes.DuckDB_Helper_v02 import DuckDBHelper 
from classes.Tronaduras_File_Reader_v03 import TronadurasFileReader

def app():

    st.title("Procesamiento e Imputación de Datos")
    st.markdown("Esta vista permite parametrizar y ejecutar el preprocesamiento, imputación y análisis de datos de **Tronaduras** y **Sismicidad**.")
    
    # ──────────────────────────────────────────────
    # Parámetros del Proceso
    # ──────────────────────────────────────────────
    st.header("Parámetros del Proceso")
    
    st.markdown("---")


    # Parámetros para cálculo de 'Mo' mostrados en columnas
    st.markdown("#### Parámetros para Cálculo de 'Mo'")
    cols = st.columns(6)
    with cols[0]:
        const_mo = st.number_input("Constante (desplazamiento)", value=6.1, step=0.1)
    with cols[1]:
        factor_mo = st.number_input("Factor Exponencial (3/2 = 1.5)", value=1.5, step=0.1)
    
    st.markdown("---")

    # Sección de gráficos con opciones adicionales
    st.markdown("#### Gráficos")
    show_graphs = st.checkbox("Mostrar gráficos", value=True)
    if show_graphs:
        show_mo_chart = st.checkbox("Gráfico de Momento Sísmico (Mo) vs Tiempo", value=True)
        show_energy_chart = st.checkbox("Gráfico de Energía vs Tiempo", value=True)
        show_magnitude_chart = st.checkbox("Gráfico de Magnitud vs Tiempo", value=True)
        show_histogram_magnitude = st.checkbox("Histograma de Magnitudes", value=True)
        show_scatter_correlation = st.checkbox("Correlación: Mo vs Energía", value=True)
    
    st.markdown("---")

    st.markdown("#### Intervalo entre Disparos")
    cols = st.columns(6)
    with cols[0]:
        hours_after_option = st.selectbox("Calcular intervalo de tiempo", options=["Mínimo", "Máximo", "Promedio", "Manual"], index=0)
    manual_hours_after = 0
    if hours_after_option == "Manual":
        with cols[1]:
            manual_hours_after = st.number_input("Horas después de cada disparo", value=0, step=1)
    
    st.markdown("---")
    
    # Botón de ejecución en el cuerpo principal
    run_process = st.button("Ejecutar Preprocesamiento")
    
    if run_process:
        # ──────────────────────────────────────────────
        # 1. Conexión y Lectura de Datos
        # ──────────────────────────────────────────────
        st.markdown("## 1. Conexión y Lectura de Datos")
        with st.spinner("Conectando a la base de datos y leyendo tablas..."):
            db_helper = DuckDBHelper(DB_NAME)
            tronaduras_reader = TronadurasFileReader()
            df_sismicidad = db_helper.select_df(table="Sismicidad", schema=SCHEMA_NAME)
            df_tronaduras = db_helper.select_df(table="Tronaduras", schema=SCHEMA_NAME)
            st.success("Datos leídos exitosamente.")
            
            st.markdown("**Sismicidad:**")
            st.dataframe(df_sismicidad, height=400)
            st.markdown("**Tronaduras:**")
            st.dataframe(df_tronaduras, height=400)
        
        # ──────────────────────────────────────────────
        # 2. Imputación de Datos en la Tabla Tronaduras
        # ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 2. Imputación de Datos en la Tabla Tronaduras")
        with st.spinner("Aplicando estrategias de imputación..."):
            strategies_df = db_helper.select_df(
                table="Variables_Description",
                columns='"Variable", "Imputacion", "Defecto"',
                where=f"Conjunto = '{CONJUNTO_ESPECIFICO}'",
                schema=SCHEMA_NAME
            )
            st.markdown("**Estrategias de imputación:**")
            st.dataframe(strategies_df, height=300)
            
            df_tronaduras_imputed = tronaduras_reader.impute_df(df=df_tronaduras, strategies_df=strategies_df)
            st.success("Imputación completada.")
            st.markdown("**Tronaduras después de imputación:**")
            st.dataframe(df_tronaduras_imputed, height=400)
        
        # ──────────────────────────────────────────────
        # 3. Creación de Nuevas Variables en Sismicidad
        # ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 3. Creación de Nuevas Variables en Sismicidad")
        with st.spinner("Calculando 'Mo'..."):
            df_sismicidad['Mo'] = 10 ** (factor_mo * (df_sismicidad['Local Magnitude'] + const_mo))
            st.success("Variable 'Mo' calculada.")
            st.markdown("**Sismicidad con 'Mo':**")
            st.dataframe(df_sismicidad, height=400)
        
        # ──────────────────────────────────────────────
        # Gráficos (Opcionales)
        # ──────────────────────────────────────────────
        if show_graphs:
            st.markdown("---")
            st.markdown("### Gráficos")
            with st.spinner("Generando gráficos..."):
                if show_mo_chart:
                    fig_mo = go.Figure()
                    fig_mo.add_trace(go.Scatter(x=df_sismicidad['Date Time'], y=df_sismicidad['Mo'], mode='lines', name='Mo'))
                    fig_mo.update_layout(title='Momento Sísmico (Mo) a lo largo del tiempo', xaxis_title='Fecha', yaxis_title='Mo')
                    st.plotly_chart(fig_mo, use_container_width=True)
                if show_energy_chart:
                    fig_energy = go.Figure()
                    fig_energy.add_trace(go.Scatter(x=df_sismicidad['Date Time'], y=df_sismicidad['Energy [J]'], mode='lines', name='Energía'))
                    fig_energy.update_layout(title='Energía a lo largo del tiempo', xaxis_title='Fecha', yaxis_title='Energía')
                    st.plotly_chart(fig_energy, use_container_width=True)
                if show_magnitude_chart:
                    fig_magnitude = go.Figure()
                    fig_magnitude.add_trace(go.Scatter(x=df_sismicidad['Date Time'], y=df_sismicidad['Local Magnitude'], mode='lines', name='Magnitud'))
                    fig_magnitude.update_layout(title='Local Magnitude a lo largo del tiempo', xaxis_title='Fecha', yaxis_title='Local Magnitude')
                    st.plotly_chart(fig_magnitude, use_container_width=True)
                if show_histogram_magnitude:
                    fig_hist = px.histogram(df_sismicidad, x='Local Magnitude', nbins=20, title='Histograma de Magnitudes')
                    st.plotly_chart(fig_hist, use_container_width=True)
                if show_scatter_correlation:
                    fig_scatter = px.scatter(df_sismicidad, x='Mo', y='Energy [J]', title='Correlación entre Mo y Energía', trendline="ols")
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        # ──────────────────────────────────────────────
        # 4. Relación entre Tronaduras y Sismicidad
        # ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 4. Relación entre Tronaduras y Sismicidad")
        console_relacion = ""
        
        with st.spinner("Asignando disparos a eventos sísmicos..."):
            # Calcular el delta entre fecha_inicio y fecha_fin (en horas)
            if hours_after_option != 'Manual':
                console_relacion += "⏲ Calculo automático del delta entre disparos"
                # Calcular el delta entre fecha_inicio y fecha_fin (en horas)
                deltas = []
                # Iterar por cada disparo en df_tronaduras para asignar los rangos
                for i in range(len(df_tronaduras) - 1):
                    disparo_actual = df_tronaduras.loc[i, 'N° Disparo']
                    fecha_inicio = df_tronaduras.loc[i, 'Fecha']
                    fecha_fin = df_tronaduras.loc[i + 1, 'Fecha']
                    
                    # Calcular el delta entre fecha_inicio y fecha_fin (en horas)
                    delta = (fecha_fin - fecha_inicio)
                    deltas.append(delta)
                    delta_min_calc = int(min(deltas).total_seconds()/3600)
                    delta_max_calc = int(max(deltas).total_seconds()/3600)
                    delta_mean_calc = int(np.mean([d.total_seconds() for d in deltas])/3600)

                console_relacion += f"⏲ Delta max: {int(max(deltas).total_seconds()/3600)} hrs. \t | Delta min: {int(min(deltas).total_seconds()/3600)} hrs. \t | Delta promedio: {int(np.mean(deltas).total_seconds()/3600)} hrs."
                

            df_sismicidad_con_disparo = df_sismicidad.copy()
            df_sismicidad_con_disparo['N° Disparo'] = None
            df_sismicidad_con_disparo['Disparo - Date Time'] = None
            
            deltas_calculados = []
            for i in range(len(df_tronaduras) - 1):
                disparo_actual = df_tronaduras.loc[i, 'N° Disparo']
                fecha_inicio = df_tronaduras.loc[i, 'Fecha']
                #fecha_fin = fecha_inicio + pd.Timedelta(hours=hours_after) if hours_after != 'Mínimo' and hours_after > 0 else df_tronaduras.loc[i + 1, 'Fecha']
                
                if hours_after_option == 'Mínimo':
                    fecha_fin = fecha_inicio + pd.Timedelta(hours=delta_min_calc)
                elif hours_after_option == 'Máximo':
                    # Define la lógica para el caso 'Máximo'
                    fecha_fin = fecha_inicio + pd.Timedelta(hours=delta_max_calc)  # Ejemplo: 24 horas después
                elif hours_after_option == 'Promedio':
                    # Define la lógica para el caso 'Promedio'
                    fecha_fin = fecha_inicio + pd.Timedelta(hours=delta_mean_calc)  # Ejemplo: 12 horas después
                elif (hours_after_option == 'Manual') & (hours_after_option > 0):
                    fecha_fin = fecha_inicio + pd.Timedelta(hours=hours_after_option)
                else:
                    fecha_fin = df_tronaduras.loc[i + 1, 'Fecha']
                
                
                delta = (fecha_fin - fecha_inicio)
                deltas_calculados.append(delta)
                
                mask = (df_sismicidad_con_disparo['Date Time'] >= fecha_inicio) & (df_sismicidad_con_disparo['Date Time'] < fecha_fin)
                n_eventos = len(df_sismicidad_con_disparo.loc[mask])
                df_sismicidad_con_disparo.loc[mask, 'N° Disparo'] = disparo_actual
                df_sismicidad_con_disparo.loc[mask, 'Disparo - Date Time'] = fecha_inicio
                
                console_relacion += f"Disparo {disparo_actual}: {fecha_inicio} - {fecha_fin} | Delta: {int(delta.total_seconds()/3600)} hrs | Eventos: {n_eventos}\n"
            
            if deltas_calculados:
                delta_max_calc = int(max(deltas_calculados).total_seconds()/3600)
                delta_min_calc = int(min(deltas_calculados).total_seconds()/3600)
                delta_mean_calc = int(np.mean([d.total_seconds() for d in deltas_calculados])/3600)
                console_relacion += f"Delta calculado: máximo {delta_max_calc} hrs | mínimo {delta_min_calc} hrs | promedio {delta_mean_calc} hrs\n"
            
            # Reordenar columnas y renombrar
            columns_order = ['N° Disparo', 'Disparo - Date Time'] + [col for col in df_sismicidad_con_disparo.columns if col not in ['N° Disparo', 'Disparo - Date Time']]
            df_sismicidad_con_disparo = df_sismicidad_con_disparo[columns_order]
            df_sismicidad_con_disparo.rename(columns={'Date Time': 'Sismicidad - Date Time'}, inplace=True)
            df_sismicidad_con_disparo = df_sismicidad_con_disparo.dropna(subset=['N° Disparo']).reset_index(drop=True)
            
            df_sismicidad_con_disparo["Mo_cumulative"] = df_sismicidad_con_disparo.groupby("N° Disparo")["Mo"].transform("sum")
            df_sismicidad_con_disparo["Energy_cumulative"] = df_sismicidad_con_disparo.groupby("N° Disparo")["Energy [J]"].transform("sum")
            console_relacion += "Asignación de disparos completada.\n"
        
        # Mostrar consola de relación
        st.markdown("**Consola de Relación entre Tronaduras y Sismicidad:**")
        st.text_area("Consola de relación", console_relacion, height=300)
        
        st.markdown("**Sismicidad con Disparo:**")
        st.dataframe(df_sismicidad_con_disparo, height=400)
        
        # ──────────────────────────────────────────────
        # 5. Creación de la Tabla Unificada
        # ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 5. Creación de la Tabla Unificada")
        with st.spinner("Realizando merge de tablas..."):
            df_sismicidad_con_disparo['-'] = '-'
            df_Tabla_Unificada = pd.merge(
                df_sismicidad_con_disparo[['N° Disparo', 'Mo_cumulative', 'Energy_cumulative', '-']],
                df_tronaduras,
                left_on="N° Disparo",
                right_on="N° Disparo",
                how="left"
            )
            df_Tabla_Unificada = df_Tabla_Unificada.drop(columns=['Fecha'])
            df_Tabla_Unificada = df_Tabla_Unificada.drop_duplicates(subset=['N° Disparo']).reset_index(drop=True)
            st.markdown("**Tabla Unificada:**")
            st.dataframe(df_Tabla_Unificada, height=400)
        
        # ──────────────────────────────────────────────
        # 6. Conversión de Variables Categóricas a Numéricas
        # ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 6. Conversión de Variables Categóricas a Numéricas")
        console_conversion = ""
        
        with st.spinner("Convirtiendo variables categóricas..."):
            vd = db_helper.select_df(table="Variables_Description", schema=SCHEMA_NAME)[['Conjunto', 'Variable', 'Tipo']]
            variables_categoricas = vd[(vd['Conjunto'] == CONJUNTO_ESPECIFICO) & (vd['Tipo'] == 'categorical')][['Variable']]
            for var in variables_categoricas['Variable']:
                if var in df_Tabla_Unificada.columns:
                    console_conversion += f"Convirtiendo {var} a numérico\n"
                    df_Tabla_Unificada[var] = pd.Categorical(df_Tabla_Unificada[var]).codes
                else:
                    console_conversion += f"La columna {var} no está presente en el DataFrame\n"
            console_conversion += "Conversión completada.\n"
        
        # Mostrar consola de conversión
        st.markdown("**Consola de Conversión de Variables Categóricas:**")
        st.text_area("Consola de conversión", console_conversion, height=300)
        
        st.markdown("**Tabla Unificada (Post Conversión):**")
        st.dataframe(df_Tabla_Unificada, height=400)
        
        # ──────────────────────────────────────────────
        # 7. Guardar Resultados en la Base de Datos
        # ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 7. Guardar Resultados en la Base de Datos")
        with st.spinner("Guardando resultados..."):
            db_helper.create_table_from_df("Tronaduras", df_tronaduras_imputed, schema="Processed_Data")
            db_helper.create_table_from_df("Sismicidad", df_sismicidad_con_disparo, schema="Processed_Data")
            db_helper.create_table_from_df("Tabla_Unificada", df_Tabla_Unificada, schema="Processed_Data")
            db_helper.close_connection()
            st.success("Datos guardados exitosamente en la base de datos.")