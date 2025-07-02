import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import re
from io import BytesIO

# --- Helper Function for Advanced Excel Parsing ---
def _process_special_excel_columns(df_pandas):
    """
    Procesa columnas específicas que contienen rangos o valores múltiples.
    """
    st.write("**Depuración (process_special_columns):** Iniciando limpieza de columnas especiales.")
    
    # Función para parsear y promediar números en una celda
    def parse_and_average(value):
        try:
            # Reemplazar comas por puntos para consistencia decimal
            s_value = str(value).replace(',', '.')
            # Encontrar todos los números (incluyendo decimales)
            numbers = re.findall(r'[\d.]+', s_value)
            if not numbers:
                return np.nan
            
            # Convertir a flotante y calcular promedio
            float_numbers = [float(n) for n in numbers]
            return sum(float_numbers) / len(float_numbers)
        except (ValueError, TypeError):
            return np.nan

    # Columnas a procesar y sus alias para un log más claro
    cols_to_process = {
        'UCS (MPa)': 'UCS',
        'Modulo de Young (GPa) ': 'Young Modulus', # Note el espacio al final
        'Razón Poisson': 'Poisson Ratio',
        'Puntaje': 'Score'
    }

    for col, alias in cols_to_process.items():
        if col in df_pandas.columns:
            st.write(f"  - Procesando columna: '{alias}'")
            df_pandas[col] = df_pandas[col].apply(parse_and_average)
            # Renombrar columnas para quitar espacios problemáticos
            if col.strip() != col:
                    df_pandas.rename(columns={col: col.strip()}, inplace=True)
                    
    return df_pandas

# --- Funciones Originales (Modificadas para usar st.session_state) ---

def convert_magnitude_to_moment(magnitude):
    """
    Convertir Local Magnitude (Mw) a Momento Sísmico (Mo).
    Fórmula: Mo = 10^((Mw + 6.21) * 3/2)
    """
    try:
        if pd.isna(magnitude):
            return None
        return 10**((magnitude + 6.21) * 3/2)
    except Exception:
        return None

def load_csv_data(uploaded_file):
    """
    Cargar y procesar datos CSV de sismicidad.
    """
    try:
        df_csv = pl.read_csv(uploaded_file.getvalue())
        
        date_col_name = None
        time_col_name = None
        
        if "#EventDate" in df_csv.columns and "EventTimeInDay" in df_csv.columns:
            date_col_name = "#EventDate"
            time_col_name = "EventTimeInDay"
        elif "EventDate" in df_csv.columns and "EventTime" in df_csv.columns:
            date_col_name = "EventDate"
            time_col_name = "EventTime"
        else:
            st.error("Columnas de fecha o tiempo requeridas no encontradas en el archivo CSV. Esperadas: '#EventDate' y 'EventTimeInDay' (o 'EventDate' y 'EventTime').")
            st.write(f"Columnas disponibles en CSV: {df_csv.columns}")
            return None

        df_csv = df_csv.with_columns([
            pl.concat_str([
                pl.col(date_col_name),
                pl.lit(" "),
                pl.col(time_col_name)
            ]).str.strptime(pl.Datetime, format="%Y/%m/%d %H:%M:%S", strict=False).alias("DATETIME")
        ])
        
        if "Local Magnitude" not in df_csv.columns:
            st.warning("Columna 'Local Magnitude' no encontrada en el CSV. La columna 'Mo' no se calculará.")
            df_csv = df_csv.with_columns(pl.lit(None).cast(pl.Float64).alias("Mo"))
        else:
            df_csv = df_csv.with_columns([
                pl.col("Local Magnitude").map_elements(convert_magnitude_to_moment, return_dtype=pl.Float64, skip_nulls=True).alias("Mo")
            ])
        
        df_csv = df_csv.with_columns([
            pl.lit("CSV_Sismicidad").alias("SOURCE")
        ])
        
        required_csv_cols = ["DATETIME", "LocX [m]", "LocY [m]", "LocZ [m]", 
                               "Local Magnitude", "Mo", "Energy [J]", "EnergyS/EnergyP", "SOURCE"]
        existing_csv_cols = [col for col in required_csv_cols if col in df_csv.columns]
        df_csv = df_csv.select(existing_csv_cols)

        if df_csv.filter(pl.col("DATETIME").is_null()).height > 0:
            st.warning(f"Se encontraron {df_csv.filter(pl.col('DATETIME').is_null()).height} filas en el CSV con DATETIME no válido. Estas filas se mantendrán pero no se utilizarán en filtros temporales o análisis de tiempo.")
            
        return df_csv
        
    except Exception as e:
        st.error(f"Error al procesar CSV: {str(e)}")
        try:
            uploaded_file.seek(0)
            debug_df = pl.read_csv(uploaded_file.getvalue())
            st.write("**Primeras filas del CSV (para depuración):**")
            st.dataframe(debug_df.head(5).to_pandas())
        except Exception as de:
            st.error(f"No se pudo mostrar información de depuración para CSV: {str(de)}")
        return None

def load_excel_data(uploaded_file):
    """
    Cargar y procesar datos Excel de tronaduras.
    """
    problematic_blast_ids = []
    try:
        df_pandas = pd.read_excel(uploaded_file, dtype={'N° Disparo': str})
        
        # --- NUEVO: Procesamiento avanzado de columnas ---
        df_pandas = _process_special_excel_columns(df_pandas)
        
        if 'N° Disparo' in df_pandas.columns:
            df_pandas['N° Disparo'] = df_pandas['N° Disparo'].astype(str)

        for col in df_pandas.select_dtypes(include=[np.number]).columns.tolist():
            df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            
        for col in df_pandas.select_dtypes(include=['object']).columns.tolist():
            if col != 'N° Disparo':
                df_pandas[col] = df_pandas[col].astype(str).str.strip().replace('nan', None)
        
        if "DATETIME" not in df_pandas.columns:
            st.error("Columna 'DATETIME' no encontrada en el archivo Excel. Esta columna es esencial.")
            st.write(f"Columnas disponibles en Excel: {list(df_pandas.columns)}")
            return None, []

        df_pandas['DATETIME_parsed'] = pd.to_datetime(df_pandas['DATETIME'], errors='coerce', format='mixed')
            
        invalid_datetime_rows = df_pandas[df_pandas['DATETIME_parsed'].isna()]
        if not invalid_datetime_rows.empty:
            st.warning(f"Se encontraron {len(invalid_datetime_rows)} filas en el Excel con DATETIME no válido.")
            problematic_blast_ids = invalid_datetime_rows['N° Disparo'].tolist()
            df_pandas['DATETIME_final'] = df_pandas['DATETIME_parsed'].replace({pd.NaT: None})
        else:
            df_pandas['DATETIME_final'] = df_pandas['DATETIME_parsed']

        df_pandas['DATETIME'] = df_pandas['DATETIME_final']
        df_pandas = df_pandas.drop(columns=['DATETIME_parsed', 'DATETIME_final'], errors='ignore')
        
        df_pandas['SOURCE'] = 'Excel_Tronaduras'
        
        # Convertir a Polars DataFrame
        df_excel = pl.from_pandas(df_pandas)

        if "DATETIME" in df_excel.columns:
            df_excel = df_excel.with_columns([
                pl.col("DATETIME").cast(pl.Datetime, strict=False).alias("DATETIME")
            ])
        
        return df_excel, problematic_blast_ids
        
    except Exception as e:
        st.error(f"Error al procesar Excel: {str(e)}")
        st.error(f"Tipo de error: {type(e).__name__}")
        try:
            uploaded_file.seek(0)
            debug_df = pd.read_excel(uploaded_file, nrows=5)
            st.write("**Primeras 5 filas del Excel (para depuración):**")
            st.dataframe(debug_df)
            st.write("**Información de tipos de datos (para depuración):**")
            st.write(debug_df.dtypes)
        except Exception as debug_error:
            st.error(f"No se pudo mostrar información de depuración para Excel: {str(debug_error)}")
        
        return None, []


def create_data_df_analysis(df_csv_filtered, df_excel_filtered):
    """
    Crea el DataFrame final Data_df.
    """
    try:
        df_excel_valid_dates = df_excel_filtered.filter(pl.col("DATETIME").is_not_null())
        
        if df_excel_valid_dates.height == 0:
            st.warning("No hay tronaduras con fechas válidas para el análisis. Data_df no se puede crear.")
            return pl.DataFrame()

        df_csv_sorted = df_csv_filtered.sort("DATETIME")
        df_excel_sorted = df_excel_valid_dates.sort("DATETIME")

        data_df_list = []
        
        csv_pd = df_csv_sorted.to_pandas()
        excel_pd = df_excel_sorted.to_pandas()

        num_blasts = len(excel_pd)

        for i in range(num_blasts):
            blast_row = excel_pd.iloc[i]
            blast_time = blast_row['DATETIME']
            
            # Definir el final de la ventana de tiempo
            if i + 1 < num_blasts:
                next_blast_time = excel_pd.iloc[i + 1]['DATETIME']
            else:
                # Para la última tronadura, la ventana se extiende hasta el infinito
                next_blast_time = pd.Timestamp.max.tz_localize(blast_time.tz) if blast_time.tz else pd.Timestamp.max

            # Filtrar eventos sísmicos en la ventana
            seismic_events_in_bag = csv_pd[
                (csv_pd['DATETIME'] >= blast_time) & (csv_pd['DATETIME'] < next_blast_time)
            ]
            
            cumulative_mo = seismic_events_in_bag['Mo'].sum() if 'Mo' in seismic_events_in_bag.columns and not seismic_events_in_bag.empty else 0
            num_seismic_events = len(seismic_events_in_bag)

            data_row = blast_row.to_dict()
            data_row['Cumulative_Mo'] = cumulative_mo
            data_row['Num_Seismic_Events'] = num_seismic_events
            
            data_df_list.append(data_row)
        
        if not data_df_list:
            return pl.DataFrame()

        data_df = pl.from_pandas(pd.DataFrame(data_df_list))

        # Lista expandida de columnas para mantener del Excel original
        excel_cols = [col for col in df_excel_filtered.columns if col not in ['Cumulative_Mo', 'Num_Seismic_Events']]
        final_cols = excel_cols + ['Cumulative_Mo', 'Num_Seismic_Events']
        
        # Seleccionar solo las columnas que existen en el dataframe resultante
        existing_columns = [col for col in final_cols if col in data_df.columns]
        
        return data_df.select(existing_columns)
        
    except Exception as e:
        st.error(f"Error creando Data_df para el análisis: {str(e)}")
        st.error(f"Tipo de error: {type(e).__name__}")
        return None

def app():
    st.title("🌋 Analizador de Sismicidad y Tronaduras")
    st.markdown("---")
    
    # --- Inicializar st.session_state ---
    if 'df_csv' not in st.session_state: st.session_state.df_csv = None
    if 'df_excel' not in st.session_state: st.session_state.df_excel = None
    if 'problematic_blasts' not in st.session_state: st.session_state.problematic_blasts = []
    if 'df_csv_filtered' not in st.session_state: st.session_state.df_csv_filtered = None
    if 'df_excel_filtered' not in st.session_state: st.session_state.df_excel_filtered = None
    if 'data_df' not in st.session_state: st.session_state.data_df = None

    # --- Sección 1: Carga de archivos ---
    st.subheader("📁 1. Carga de Archivos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_file = st.file_uploader("Archivo CSV (Eventos Sísmicos)", type=['csv'], key="csv_uploader")
    
    with col2:
        excel_file = st.file_uploader("Archivo Excel (Tronaduras)", type=['xlsx', 'xls'], key="excel_uploader")
    
    if csv_file and st.session_state.df_csv is None:
        with st.spinner("Cargando y procesando CSV..."):
            st.session_state.df_csv = load_csv_data(csv_file)

    if excel_file and st.session_state.df_excel is None:
        with st.spinner("Cargando y procesando Excel..."):
            st.session_state.df_excel, st.session_state.problematic_blasts = load_excel_data(excel_file)
    
    if st.session_state.df_csv is None or st.session_state.df_excel is None:
        st.info("Por favor, carga ambos archivos para continuar con el análisis.")
        return
    
    st.success("✅ Archivos cargados exitosamente!")
    if st.session_state.problematic_blasts:
        st.warning(f"⚠️ Las siguientes tronaduras tienen fechas/horas inválidas y no se incluirán en el análisis temporal: {', '.join(st.session_state.problematic_blasts)}.")
        
    # --- Sección 2: Filtros temporales ---
    st.markdown("---")
    st.subheader("📅 2. Filtros Temporales")
    
    csv_valid_dates = st.session_state.df_csv.filter(pl.col("DATETIME").is_not_null())
    excel_valid_dates = st.session_state.df_excel.filter(pl.col("DATETIME").is_not_null())
    
    if csv_valid_dates.height == 0 and excel_valid_dates.height == 0:
        st.error("No hay datos con fechas válidas en ninguno de los archivos.")
        return
        
    all_min_dates = [d for d in [csv_valid_dates.select(pl.min("DATETIME")).item(), excel_valid_dates.select(pl.min("DATETIME")).item()] if d]
    all_max_dates = [d for d in [csv_valid_dates.select(pl.max("DATETIME")).item(), excel_valid_dates.select(pl.max("DATETIME")).item()] if d]

    overall_min = min(all_min_dates) if all_min_dates else datetime(2000, 1, 1)
    overall_max = max(all_max_dates) if all_max_dates else datetime.now()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha Inicial", value=overall_min.date(), min_value=overall_min.date(), max_value=overall_max.date())
    with col2:
        end_date = st.date_input("Fecha Final", value=overall_max.date(), min_value=overall_min.date(), max_value=overall_max.date())
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    st.session_state.df_csv_filtered = st.session_state.df_csv.filter(
        (pl.col("DATETIME").is_not_null()) & (pl.col("DATETIME") >= start_datetime) & (pl.col("DATETIME") <= end_datetime)
    )
    st.session_state.df_excel_filtered = st.session_state.df_excel.filter(
        (pl.col("DATETIME").is_not_null()) & (pl.col("DATETIME") >= start_datetime) & (pl.col("DATETIME") <= end_datetime)
    )

    if st.session_state.df_excel_filtered.height == 0:
        st.warning("No hay tronaduras válidas en el rango de fechas seleccionado. Ajusta los filtros.")
        return

    # --- Sección 3: Procesamiento de datos ---
    st.markdown("---")
    st.subheader("⚙️ 3. Procesamiento de Datos")
    
    with st.spinner("Calculando momentos sísmicos acumulativos por tronadura..."):
        st.session_state.data_df = create_data_df_analysis(st.session_state.df_csv_filtered, st.session_state.df_excel_filtered)
    
    if st.session_state.data_df is None or st.session_state.data_df.height == 0:
        st.error("Error en el procesamiento o no se generaron resultados válidos. Revisa los filtros o los archivos de entrada.")
        return
    
    st.success("✅ Procesamiento completado!")
    
    # --- Sección 4: Métricas ---
    st.markdown("---")
    st.subheader("📊 4. Resultados del Análisis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Eventos Sísmicos Filtrados", st.session_state.df_csv_filtered.height)
    col2.metric("Tronaduras Analizadas", st.session_state.df_excel_filtered.height)
    total_mo = st.session_state.data_df.select(pl.col("Cumulative_Mo").sum()).item()
    col3.metric("Momento Sísmico Total", f"{total_mo:.2e}")
    total_events = st.session_state.data_df.select(pl.col("Num_Seismic_Events").sum()).item()
    col4.metric("Eventos Asociados Únicos", total_events)
    
    # --- Sección 5: Previsualización de Datos ---
    st.markdown("---")
    st.subheader("🔍 5. Previsualización de Datos")
    
    tab1, tab2, tab3 = st.tabs(["df_csv (Sismicidad Filtrada)", "df_excel (Tronaduras Cargadas)", "Data_df (Resultados por Tronadura)"])
    with tab1:
        st.dataframe(st.session_state.df_csv_filtered.to_pandas(), use_container_width=True)
    with tab2:
        st.dataframe(st.session_state.df_excel.to_pandas(), use_container_width=True)
    with tab3:
        st.dataframe(st.session_state.data_df.to_pandas(), use_container_width=True)

    # --- Sección 6: Visualización Principal ---
    st.markdown("---")
    st.subheader("📈 6. Visualización del Momento Sísmico Acumulativo")
    
    plot_data_6 = st.session_state.data_df.to_pandas()
    fig_bar = px.bar(plot_data_6, x='N° Disparo', y='Cumulative_Mo', title='Momento Sísmico Acumulativo por Tronadura',
                        hover_data=['Num_Seismic_Events', 'DATETIME'])
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_scatter = px.scatter(plot_data_6, x='DATETIME', y='Cumulative_Mo', size='Num_Seismic_Events',
                                 title='Evolución Temporal del Momento Sísmico por Tronadura', hover_data=['N° Disparo'])
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # --- SECCIÓN 7 CORREGIDA ---
    st.markdown("---")
    st.subheader("🎨 7. Gráficas Personalizadas (2D y 3D)")

    plot_data_7_base = st.session_state.data_df.to_pandas()
    numeric_cols = plot_data_7_base.select_dtypes(include=np.number).columns.tolist()
    all_cols = plot_data_7_base.columns.tolist()

    tab2d, tab3d = st.tabs(["Gráfica 2D", "Gráfica 3D"])

    with tab2d:
        st.markdown("**Crea tu propio gráfico 2D**")
        c1, c2, c3, c4 = st.columns(4)
        x_axis = c1.selectbox("Eje X", options=all_cols, index=all_cols.index('DATETIME') if 'DATETIME' in all_cols else 0, key="2d_x")
        y_axes = c2.multiselect("Eje(s) Y", options=numeric_cols, default=[c for c in ['Cumulative_Mo', 'Num_Seismic_Events'] if c in numeric_cols])
        y_axis2 = c3.selectbox("Eje Y Secundario (Opcional)", options=[None] + numeric_cols, key="2d_y2")
        plot_type = c4.selectbox("Tipo de Gráfico", options=["Línea", "Puntos", "Línea y Puntos"], key="2d_type")

        if x_axis and y_axes:
            cols_to_check_2d = [x_axis] + y_axes
            if y_axis2:
                cols_to_check_2d.append(y_axis2)
            
            plot_data_2d = plot_data_7_base.dropna(subset=cols_to_check_2d)

            if not plot_data_2d.empty:
                fig_2d = go.Figure()
                mode_map = {"Línea": "lines", "Puntos": "markers", "Línea y Puntos": "lines+markers"}
                
                for y_axis in y_axes:
                    fig_2d.add_trace(go.Scatter(x=plot_data_2d[x_axis], y=plot_data_2d[y_axis], name=y_axis, mode=mode_map[plot_type]))

                if y_axis2:
                    fig_2d.add_trace(go.Scatter(x=plot_data_2d[x_axis], y=plot_data_2d[y_axis2], name=f"{y_axis2} (Eje Der.)", mode=mode_map[plot_type], yaxis="y2"))
                    fig_2d.update_layout(yaxis2=dict(title=y_axis2, overlaying="y", side="right"))

                fig_2d.update_layout(title=f"Gráfico 2D Personalizado", xaxis_title=x_axis, yaxis_title=", ".join(y_axes), legend_title="Métricas")
                st.plotly_chart(fig_2d, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para graficar con las columnas seleccionadas. Por favor, revisa si las columnas tienen datos válidos.")

    with tab3d:
        st.markdown("**Crea tu propio gráfico 3D**")
        c1, c2, c3, c4 = st.columns(4)
        # Prevenir error si la columna no existe
        def get_index(col_list, col_name, default=0):
            try: return col_list.index(col_name)
            except ValueError: return default

        x_3d = c1.selectbox("Eje X (3D)", options=numeric_cols, index=get_index(numeric_cols, 'PK', 0))
        y_3d = c2.selectbox("Eje Y (3D)", options=numeric_cols, index=get_index(numeric_cols, 'Kg. de explosivos tronadura', 1))
        z_3d = c3.selectbox("Eje Z (3D)", options=numeric_cols, index=get_index(numeric_cols, 'Cumulative_Mo', 2))
        color_3d = c4.selectbox("Colorear por", options=[None] + all_cols)

        if x_3d and y_3d and z_3d:
            cols_to_check_3d = [x_3d, y_3d, z_3d]
            if color_3d:
                cols_to_check_3d.append(color_3d)

            plot_data_3d = plot_data_7_base.dropna(subset=cols_to_check_3d)

            if not plot_data_3d.empty:
                fig_3d = px.scatter_3d(plot_data_3d, x=x_3d, y=y_3d, z=z_3d, color=color_3d,
                                        title="Gráfico 3D Interactivo", hover_data=['N° Disparo'])
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para graficar en 3D con las columnas seleccionadas.")

    # --- SECCIÓN 8 CORREGIDA ---
    st.markdown("---")
    st.subheader("📈 8. Progresión del Momento Sísmico (MoT) por Tronadura")

    all_blasts = st.session_state.data_df.get_column('N° Disparo').to_list()
    selected_blasts = st.multiselect("Selecciona Tronaduras ('N° Disparo') para analizar", options=all_blasts)

    if selected_blasts:
        c1, c2 = st.columns([3, 1])
        x_axis_mot = c1.selectbox("Eje X para la progresión", options=['DATETIME', 'TimeDelta since blast'], index=0, key="mot_x")
        show_local_mag = c2.checkbox("Mostrar Local Magnitude", value=True, key="mot_mag")

        fig_mot = go.Figure()

        for blast_id in selected_blasts:
            blast_row = st.session_state.data_df.filter(pl.col('N° Disparo') == blast_id)
            if blast_row.height == 0: continue
            
            blast_time = blast_row.item(0, 'DATETIME')
            
            # Determinar el final de la ventana de tiempo para esta tronadura
            sorted_blasts = st.session_state.data_df.sort('DATETIME').with_row_count()
            blast_index_info = sorted_blasts.filter(pl.col('N° Disparo') == blast_id)
            if blast_index_info.height == 0: continue
            blast_index = blast_index_info.item(0, 'row_nr')
            
            next_blast_time = sorted_blasts[blast_index + 1].item(0, 'DATETIME') if blast_index + 1 < sorted_blasts.height else datetime.max

            events = st.session_state.df_csv_filtered.filter(
                (pl.col('DATETIME') >= blast_time) & (pl.col('DATETIME') < next_blast_time)
            ).sort('DATETIME').with_columns([
                pl.col('Mo').cum_sum().alias('MoT_Progression'),
                (pl.col('DATETIME') - blast_time).alias('TimeDelta since blast')
            ]).to_pandas()

            if not events.empty:
                x_col_to_plot = x_axis_mot
                
                # CORRECCIÓN: Convertir timedelta a segundos si es necesario
                if x_axis_mot == 'TimeDelta since blast':
                    events['TimeDelta_seconds'] = events['TimeDelta since blast'].dt.total_seconds()
                    x_col_to_plot = 'TimeDelta_seconds'

                # Eje Secundario: Progresión del MoT
                fig_mot.add_trace(go.Scatter(
                    x=events[x_col_to_plot], y=events['MoT_Progression'], name=f'MoT - {blast_id}',
                    mode='lines', yaxis='y2'
                ))
                # Eje Primario: Local Magnitude
                if show_local_mag and 'Local Magnitude' in events.columns:
                    fig_mot.add_trace(go.Scatter(
                        x=events[x_col_to_plot], y=events['Local Magnitude'], name=f'Mag - {blast_id}',
                        mode='markers', yaxis='y1', marker=dict(size=8, opacity=0.7)
                    ))

        # Títulos de ejes dinámicos
        x_title = "Fecha y Hora" if x_axis_mot == 'DATETIME' else "Tiempo desde Tronadura (segundos)"
        fig_mot.update_layout(
            title="Progresión del Momento Sísmico (MoT) y Eventos por Tronadura",
            xaxis_title=x_title,
            yaxis=dict(title="Local Magnitude"),
            yaxis2=dict(title="Momento Sísmico Acumulativo (MoT)", overlaying='y', side='right', type='log'),
            legend_title="Leyenda"
        )
        st.plotly_chart(fig_mot, use_container_width=True)


    # --- SECCIÓN 9 MODIFICADA ---
    st.markdown("---")
    st.subheader("🌐 9. Evolución Espacio-Temporal Sísmica")

    st.info("Visualiza la ubicación 3D de los eventos sísmicos asociados a una o más tronaduras.")

    c1, c2 = st.columns(2)

    # Asegurarse de que `all_blasts` esté definido en esta sección
    if st.session_state.data_df is not None and st.session_state.data_df.height > 0:
        all_blasts_for_3d = st.session_state.data_df.get_column('N° Disparo').to_list()
    else:
        all_blasts_for_3d = []

    # CAMBIO: Usar multiselect para permitir múltiples selecciones.
    blasts_to_view_3d = c1.multiselect(
        "Selecciona 'N° Disparo' para ver sus eventos 3D",
        options=all_blasts_for_3d,
        key="blast_3d_multiselector"
    )

    # CAMBIO: Añadir 'N° Disparo Asociado' como opción de color.
    color_options_3d = ['N° Disparo Asociado', 'Local Magnitude', 'Energy [J]', 'EnergyS/EnergyP']
    color_events_by = c2.selectbox(
        "Colorear eventos por", 
        options=color_options_3d, 
        key="color_3d_selector_v2"
    )

    if blasts_to_view_3d:
        # Lista para almacenar los dataframes de eventos de cada tronadura seleccionada
        events_for_plotting = []

        # Ordenar las tronaduras por tiempo una sola vez para buscar eficientemente
        sorted_blasts = st.session_state.data_df.sort('DATETIME').with_row_count()

        for blast_id in blasts_to_view_3d:
            blast_row = sorted_blasts.filter(pl.col('N° Disparo') == blast_id)

            if blast_row.height > 0:
                blast_time = blast_row.item(0, 'DATETIME')
                blast_index = blast_row.item(0, 'row_nr')

                # Determinar el final de la ventana de tiempo
                next_blast_time = sorted_blasts[blast_index + 1].item(0, 'DATETIME') if blast_index + 1 < sorted_blasts.height else datetime.max

                # Filtrar eventos y añadir una columna para identificar a qué tronadura pertenecen
                events_3d = st.session_state.df_csv_filtered.filter(
                    (pl.col('DATETIME') >= blast_time) & (pl.col('DATETIME') < next_blast_time)
                ).with_columns(
                    pl.lit(str(blast_id)).alias('N° Disparo Asociado') # Asegurar que sea string para categoría
                )

                if events_3d.height > 0:
                    events_for_plotting.append(events_3d)
        
        if events_for_plotting:
            # Combinar todos los dataframes de eventos en uno solo (usando Polars)
            final_events_df_pl = pl.concat(events_for_plotting)
            final_events_df_pd = final_events_df_pl.to_pandas()

            # Usar la columna seleccionada para colorear.
            color_col = color_events_by if color_events_by in final_events_df_pd.columns else None

            fig_events_3d = px.scatter_3d(
                final_events_df_pd,
                x='LocX [m]',
                y='LocY [m]',
                z='LocZ [m]',
                color=color_col,
                title=f"Eventos Sísmicos 3D para Tronaduras Seleccionadas",
                hover_data=['DATETIME', 'Local Magnitude', 'Energy [J]', 'N° Disparo Asociado']
            )
            fig_events_3d.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                legend_title_text=color_events_by # Título de la leyenda
            )
            st.plotly_chart(fig_events_3d, use_container_width=True)
        else:
            st.warning("No se encontraron eventos sísmicos asociados a las tronaduras seleccionadas en el rango de tiempo filtrado.")


    # --- Sección 10: Descargar Resultados ---
    st.markdown("---")
    st.subheader("💾 10. Descargar Resultados")
    
    if st.session_state.data_df is not None and st.session_state.data_df.height > 0:
        output = BytesIO()
        st.session_state.data_df.to_pandas().to_excel(output, index=False, sheet_name='AnalisisSismico')
        excel_data = output.getvalue()

        st.download_button(
            label="Descargar Data_df como Excel",
            data=excel_data,
            file_name=f"analisis_sismico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No hay datos procesados para descargar.")

if __name__ == "__main__":
    app()