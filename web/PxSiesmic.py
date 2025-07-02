import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, time, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import norm, lognorm, gamma, weibull_min, expon
import warnings
warnings.filterwarnings('ignore')

# --- Funciones de Análisis de Probabilidad (Originales) ---

def calculate_probability_stats(data, column_name):
    """
    Calcula estadísticas básicas de probabilidad para una columna de datos.
    """
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return None
    
    stats_dict = {
        'count': len(clean_data),
        'mean': np.mean(clean_data),
        'std': np.std(clean_data),
        'min': np.min(clean_data),
        'max': np.max(clean_data),
        'median': np.median(clean_data),
        'q25': np.percentile(clean_data, 25),
        'q75': np.percentile(clean_data, 75),
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data)
    }
    
    return stats_dict

def fit_distributions(data):
    """
    Ajusta diferentes distribuciones a los datos y devuelve los parámetros y métricas de bondad de ajuste.
    """
    clean_data = data.dropna()
    if len(clean_data) < 10:
        return None
    
    distributions = {
        'Normal': norm,
        'Log-Normal': lognorm,
        'Gamma': gamma,
        'Weibull': weibull_min,
        'Exponencial': expon
    }
    
    results = {}
    
    for dist_name, distribution in distributions.items():
        try:
            if dist_name == 'Exponencial' and np.min(clean_data) < 0:
                continue
            
            params = distribution.fit(clean_data)
            ks_stat, ks_p_value = stats.kstest(clean_data, lambda x: distribution.cdf(x, *params))
            
            log_likelihood = np.sum(distribution.logpdf(clean_data, *params))
            k = len(params)
            n = len(clean_data)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            results[dist_name] = {
                'params': params,
                'ks_stat': ks_stat,
                'ks_p_value': ks_p_value,
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood
            }
        except Exception as e:
            continue
    
    return results

def calculate_exceedance_probability(data, threshold):
    """
    Calcula la probabilidad de excedencia para un valor umbral dado.
    """
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return 0
    
    exceedances = np.sum(clean_data > threshold)
    probability = exceedances / len(clean_data)
    
    return probability

def generate_return_periods(data, percentiles=[50, 75, 90, 95, 99]):
    """
    Calcula períodos de retorno basados en percentiles.
    """
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return None
    
    return_periods = {}
    for p in percentiles:
        value = np.percentile(clean_data, p)
        return_period = len(clean_data) / (len(clean_data) * (1 - p/100))
        return_periods[f'P{p}'] = {
            'value': value,
            'return_period': return_period,
            'exceedance_prob': (100 - p) / 100
        }
    
    return return_periods

# --- **NUEVA FUNCIÓN DE CARGA** (Reemplaza la original) ---
def load_csv_and_prepare_data(uploaded_file):
    """
    Carga datos CSV, limpia la cabecera (quitando '#') y crea la columna DATETIME.
    """
    try:
        # 1. Leer CSV sin tratar '#' como comentario para leer la cabecera.
        df_csv = pl.read_csv(uploaded_file.getvalue())

        # 2. Limpiar el '#' de los nombres de las columnas.
        cleaned_columns = [col.lstrip('#').strip() for col in df_csv.columns]
        df_csv.columns = cleaned_columns

        # 3. Verificar que las columnas de fecha/hora existan.
        if 'EventDate' not in df_csv.columns or 'EventTimeInDay' not in df_csv.columns:
            st.error("El archivo CSV debe contener las columnas 'EventDate' y 'EventTimeInDay'.")
            st.write("Columnas encontradas después de la limpieza:", df_csv.columns)
            return None

        # 4. Crear columna DATETIME.
        df_csv = df_csv.with_columns(
            pl.concat_str(
                [pl.col("EventDate"), pl.lit(" "), pl.col("EventTimeInDay")]
            ).str.to_datetime("%Y/%m/%d %H:%M:%S", strict=False).alias("DATETIME")
        )

        # 5. Limpiar columnas numéricas.
        numeric_cols_to_check = ['LocX [m]', 'LocY [m]', 'LocZ [m]', 'Local Magnitude', 'Energy [J]', 'EnergyS/EnergyP']
        for col in numeric_cols_to_check:
            if col in df_csv.columns:
                df_csv = df_csv.with_columns([
                    pl.col(col).cast(pl.Float64, strict=False).fill_nan(None).alias(col)
                ])
        
        # 6. Filtrar filas donde DATETIME no se pudo parsear.
        df_csv = df_csv.filter(pl.col("DATETIME").is_not_null())
        
        return df_csv
        
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo CSV: {str(e)}")
        return None

# --- Aplicación Principal ---
def app():
    st.title("📊 Analizador de Probabilidad de Ocurrencia Sísmica")
    st.markdown("**Análisis estadístico y probabilístico de Local Magnitude, Energy [J] y EnergyS/EnergyP**")
    st.markdown("---")
    
    # --- Sección 1: Carga de archivo ---
    st.subheader("📁 1. Carga de Archivo CSV")
    
    csv_file = st.file_uploader(
        "Carga tu archivo CSV con datos sísmicos", 
        type=['csv'], 
        help="El archivo debe contener columnas de fecha, hora y variables numéricas."
    )
    
    if not csv_file:
        st.info("Por favor, carga un archivo CSV para comenzar el análisis.")
        return

    # Usar la nueva función de carga y procesamiento
    df = load_csv_and_prepare_data(csv_file)

    if df is None or df.is_empty():
        st.error("No se pudieron cargar datos válidos del archivo. Revisa el mensaje de error anterior.")
        return
    
    st.success("✅ Archivo cargado y procesado exitosamente!")
    original_row_count = len(df)
    
    # --- **NUEVA SECCIÓN DE FILTROS** ---
    st.markdown("---")
    st.subheader("🔎 2. Filtros de Datos")

    all_numeric_cols = ['LocX [m]', 'LocY [m]', 'LocZ [m]', 'Local Magnitude', 'Energy [J]', 'EnergyS/EnergyP']
    available_numeric_cols = [col for col in all_numeric_cols if col in df.columns]

    min_datetime = df['DATETIME'].min()
    max_datetime = df['DATETIME'].max()

    st.markdown("**Filtro por Fecha y Hora**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Fecha de inicio', value=min_datetime.date(), min_value=min_datetime.date(), max_value=max_datetime.date(), key='start_date')
        start_time = st.time_input('Hora de inicio (HH:MM:SS)', value=min_datetime.time(), key='start_time')
    with col2:
        end_date = st.date_input('Fecha de fin', value=max_datetime.date(), min_value=min_datetime.date(), max_value=max_datetime.date(), key='end_date')
        end_time = st.time_input('Hora de fin (HH:MM:SS)', value=max_datetime.time(), key='end_time')

    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

    with st.expander("Filtros para variables numéricas"):
        filters = {}
        for col in available_numeric_cols:
            col_series = df[col].drop_nulls()
            if not col_series.is_empty():
                min_val = float(col_series.min())
                max_val = float(col_series.max())
                
                if min_val >= max_val:
                    filters[col] = (min_val, max_val)
                else:
                    selected_range = st.slider(f"Rango para {col}", min_val, max_val, (min_val, max_val))
                    filters[col] = selected_range

    filter_conditions = [pl.col('DATETIME').is_between(start_datetime, end_datetime, closed="both")]
    for col, (min_val, max_val) in filters.items():
        if min_val <= max_val:
            filter_conditions.append(pl.col(col).is_between(min_val, max_val, closed="both"))

    final_filter = filter_conditions[0]
    for condition in filter_conditions[1:]:
        final_filter = final_filter & condition

    filtered_df = df.filter(final_filter)
    filtered_row_count = len(filtered_df)

    st.info(f"Mostrando **{filtered_row_count}** de **{original_row_count}** eventos después de aplicar los filtros.")

    if filtered_df.is_empty():
        st.warning("No hay datos que coincidan con los filtros seleccionados. Por favor, ajusta los filtros.")
        return

    # --- EL RESTO DEL CÓDIGO USA LOS DATOS FILTRADOS ---
    data_pd = filtered_df.to_pandas()
    
    # --- Sección 3: Selección de variable a analizar ---
    st.markdown("---")
    st.subheader("🎯 3. Selección de Variable")
    
    var_mapping = {
        'Local Magnitude': 'Local Magnitude',
        'Energy [J]': 'Energy [J]',
        'EnergyS/EnergyP': 'EnergyS/EnergyP'
    }
    available_vars = [var for var in var_mapping if var in data_pd.columns]
    
    if not available_vars:
        st.error("No se encontraron las variables requeridas para el análisis en los datos filtrados.")
        return
    
    selected_var = st.selectbox("Selecciona la variable para análisis de probabilidad", available_vars)
    selected_col = var_mapping[selected_var]
    
    valid_data = data_pd[selected_col].dropna()
    valid_data = valid_data[np.isfinite(valid_data)]
    
    if len(valid_data) == 0:
        st.error(f"No hay datos válidos para {selected_var} con los filtros actuales")
        return
    
    st.info(f"📈 Analizando **{selected_var}** | Datos válidos filtrados: **{len(valid_data)}**")
    
    # --- Sección 4: Estadísticas descriptivas ---
    st.markdown("---")
    st.subheader("📊 4. Estadísticas Descriptivas")
    
    stats_dict = calculate_probability_stats(valid_data, selected_var)
    
    if stats_dict:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Conteo", f"{stats_dict['count']:,}")
        col2.metric("Media", f"{stats_dict['mean']:.4f}")
        col3.metric("Desv. Estándar", f"{stats_dict['std']:.4f}")
        col4.metric("Mediana", f"{stats_dict['median']:.4f}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mínimo", f"{stats_dict['min']:.4f}")
        col2.metric("Q25", f"{stats_dict['q25']:.4f}")
        col3.metric("Q75", f"{stats_dict['q75']:.4f}")
        col4.metric("Máximo", f"{stats_dict['max']:.4f}")
        
        col1, col2 = st.columns(2)
        col1.metric("Asimetría", f"{stats_dict['skewness']:.4f}")
        col2.metric("Curtosis", f"{stats_dict['kurtosis']:.4f}")
    
    # --- Sección 5: Histograma y distribución ---
    st.markdown("---")
    st.subheader("📈 5. Distribución de Datos")
    
    col1, col2 = st.columns(2)
    with col1:
        bins = st.slider("Número de bins para histograma", 10, 100, 30)
    with col2:
        show_kde = st.checkbox("Mostrar estimación de densidad (KDE)", True)
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=valid_data, nbinsx=bins, name="Histograma", opacity=0.7, histnorm="probability density"))
    
    if show_kde:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(valid_data)
            x_range = np.linspace(valid_data.min(), valid_data.max(), 100)
            kde_values = kde(x_range)
            fig_hist.add_trace(go.Scatter(x=x_range, y=kde_values, mode='lines', name='KDE', line=dict(color='red', width=2)))
        except Exception as e:
            st.warning("No se pudo calcular KDE")
    
    fig_hist.update_layout(title=f"Distribución de {selected_var}", xaxis_title=selected_var, yaxis_title="Densidad de Probabilidad")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # --- Sección 6: Ajuste de distribuciones ---
    st.markdown("---")
    st.subheader("🎲 6. Ajuste de Distribuciones")
    
    with st.spinner("Ajustando distribuciones..."):
        dist_results = fit_distributions(valid_data)
    
    if dist_results:
        dist_comparison = []
        for dist_name, results in dist_results.items():
            dist_comparison.append({'Distribución': dist_name, 'AIC': results['aic'], 'BIC': results['bic'], 'KS Estadístico': results['ks_stat'], 'KS p-valor': results['ks_p_value'], 'Log-Likelihood': results['log_likelihood']})
        
        comparison_df = pd.DataFrame(dist_comparison).sort_values('AIC')
        st.dataframe(comparison_df, use_container_width=True)
        best_dist = comparison_df.iloc[0]['Distribución']
        st.success(f"🏆 **Mejor ajuste según AIC:** {best_dist}")
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=valid_data, nbinsx=30, name="Datos observados", opacity=0.5, histnorm="probability density"))
        
        x_range = np.linspace(valid_data.min(), valid_data.max(), 200)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        dist_map = {'Normal': norm, 'Log-Normal': lognorm, 'Gamma': gamma, 'Weibull': weibull_min, 'Exponencial': expon}
        for i, (dist_name, results) in enumerate(dist_results.items()):
            try:
                y_values = dist_map[dist_name].pdf(x_range, *results['params'])
                fig_dist.add_trace(go.Scatter(x=x_range, y=y_values, mode='lines', name=f"{dist_name}", line=dict(color=colors[i % len(colors)], width=2)))
            except Exception as e:
                continue
        
        fig_dist.update_layout(title=f"Comparación de Distribuciones - {selected_var}", xaxis_title=selected_var, yaxis_title="Densidad de Probabilidad")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # --- Sección 7: Análisis de probabilidad de excedencia ---
    st.markdown("---")
    st.subheader("⚡ 7. Análisis de Probabilidad de Excedencia")
    
    col1, col2 = st.columns(2)
    with col1:
        step_val = float(valid_data.std() / 10) if valid_data.std() > 0 else 0.01
        threshold_value = st.number_input(f"Valor umbral para {selected_var}", min_value=float(valid_data.min()), max_value=float(valid_data.max()), value=float(valid_data.median()), step=step_val)
    with col2:
        exceedance_prob = calculate_exceedance_probability(valid_data, threshold_value)
        st.metric("Probabilidad de Excedencia", f"{exceedance_prob:.4f} ({exceedance_prob*100:.2f}%)")
    
    thresholds = np.linspace(valid_data.min(), valid_data.max(), 100)
    exceedance_probs = [calculate_exceedance_probability(valid_data, t) for t in thresholds]
    
    fig_exceed = go.Figure()
    fig_exceed.add_trace(go.Scatter(x=thresholds, y=exceedance_probs, mode='lines', name='Probabilidad de Excedencia', line=dict(color='red', width=2)))
    fig_exceed.add_vline(x=threshold_value, line_dash="dash", line_color="blue", annotation_text=f"Umbral: {threshold_value:.3f}")
    fig_exceed.update_layout(title=f"Curva de Probabilidad de Excedencia - {selected_var}", xaxis_title=selected_var, yaxis_title="Probabilidad de Excedencia")
    st.plotly_chart(fig_exceed, use_container_width=True)
    
    # --- Sección 8: Períodos de retorno ---
    st.markdown("---")
    st.subheader("🔄 8. Análisis de Períodos de Retorno")
    
    return_periods = generate_return_periods(valid_data)
    
    if return_periods:
        return_period_data = [{'Percentil': p, 'Valor': d['value'], 'Período de Retorno (eventos)': d['return_period'], 'Probabilidad de Excedencia': d['exceedance_prob']} for p, d in return_periods.items()]
        return_df = pd.DataFrame(return_period_data)
        st.dataframe(return_df, use_container_width=True)
        
        fig_return = go.Figure()
        fig_return.add_trace(go.Scatter(x=return_df['Período de Retorno (eventos)'], y=return_df['Valor'], mode='markers+lines', name='Períodos de Retorno', marker=dict(size=8)))
        fig_return.update_layout(title=f"Períodos de Retorno - {selected_var}", xaxis_title="Período de Retorno (eventos)", yaxis_title=selected_var, xaxis_type="log")
        st.plotly_chart(fig_return, use_container_width=True)
    
    # --- Sección 9: Análisis de cola (valores extremos) ---
    st.markdown("---")
    st.subheader("🎯 9. Análisis de Valores Extremos")
    
    col1, col2 = st.columns(2)
    with col1:
        extreme_percentile = st.slider("Percentil para valores extremos", 90, 99, 95)
    
    extreme_threshold = np.percentile(valid_data, extreme_percentile)
    extreme_values = valid_data[valid_data >= extreme_threshold]
    
    with col2:
        st.metric("Umbral de Valores Extremos", f"{extreme_threshold:.4f}")
        st.metric("Número de Valores Extremos", len(extreme_values))
        st.metric("% de Valores Extremos", f"{len(extreme_values)/len(valid_data)*100:.2f}%")
    
    if len(extreme_values) > 1:
        st.markdown("**Estadísticas de Valores Extremos:**")
        extreme_stats = calculate_probability_stats(extreme_values, "Valores Extremos")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Media Extremos", f"{extreme_stats['mean']:.4f}")
        col2.metric("Desv. Est. Extremos", f"{extreme_stats['std']:.4f}")
        col3.metric("Máximo Absoluto", f"{extreme_stats['max']:.4f}")
        
        fig_extreme = go.Figure()
        nbins_extreme = max(1, min(20, int(len(extreme_values) / 2)))
        fig_extreme.add_trace(go.Histogram(x=extreme_values, nbinsx=nbins_extreme, name="Valores Extremos", opacity=0.7))
        fig_extreme.update_layout(title=f"Distribución de Valores Extremos (>{extreme_percentile}° percentil) - {selected_var}", xaxis_title=selected_var, yaxis_title="Frecuencia")
        st.plotly_chart(fig_extreme, use_container_width=True)
    
    # --- Sección 10: Resumen y recomendaciones ---
    st.markdown("---")
    st.subheader("📋 10. Resumen y Recomendaciones")
    
    with st.expander("🔍 Interpretación de Resultados"):
        if stats_dict:
            st.markdown(f"""
            ### Análisis de {selected_var}
            
            **Características de los datos:**
            - **Tamaño de muestra:** {len(valid_data):,} eventos
            - **Rango:** {valid_data.min():.4f} - {valid_data.max():.4f}
            - **Distribución:** {'Asimétrica positiva' if stats_dict['skewness'] > 0 else 'Asimétrica negativa' if stats_dict['skewness'] < 0 else 'Simétrica'}
            
            **Probabilidad de excedencia:**
            - Un valor de {threshold_value:.3f} tiene una probabilidad de excedencia de {exceedance_prob*100:.2f}%
            - Esto significa que aproximadamente {exceedance_prob*len(valid_data):.0f} eventos en la muestra superan este valor
            
            **Valores extremos:**
            - El {extreme_percentile}% de los eventos más intensos tienen valores ≥ {extreme_threshold:.4f}
            - Estos representan {len(extreme_values)} eventos ({len(extreme_values)/len(valid_data)*100:.1f}% del total)
            
            **Recomendaciones:**
            - Utiliza la distribución con menor AIC para modelado predictivo.
            - Considera los períodos de retorno para planificación de riesgos.
            - Los valores extremos requieren atención especial en el análisis de riesgos.
            """)
    
    # --- Sección 11: Comparación entre variables ---
    st.markdown("---")
    st.subheader("⚖️ 11. Comparación entre Variables")
    
    if len(available_vars) > 1:
        compare_vars = st.multiselect(
            "Selecciona variables para comparar",
            available_vars,
            default=available_vars[:2] if len(available_vars) >= 2 else available_vars
        )
        
        if len(compare_vars) >= 2:
            fig_compare = make_subplots(rows=len(compare_vars), cols=1, subplot_titles=[f"Distribución de {var}" for var in compare_vars], vertical_spacing=0.15)
            colors = px.colors.qualitative.Plotly
            
            for i, var in enumerate(compare_vars):
                var_data = data_pd[var_mapping[var]].dropna()
                var_data = var_data[np.isfinite(var_data)]
                if not var_data.empty:
                    fig_compare.add_trace(go.Histogram(x=var_data, name=var, opacity=0.7, nbinsx=30, histnorm="probability density", marker_color=colors[i % len(colors)]), row=i+1, col=1)
            
            fig_compare.update_layout(height=300 * len(compare_vars), title_text="Comparación de Distribuciones", showlegend=True)
            st.plotly_chart(fig_compare, use_container_width=True)
            
            comparison_stats = []
            for var in compare_vars:
                var_data = data_pd[var_mapping[var]].dropna()
                var_data = var_data[np.isfinite(var_data)]
                var_stats = calculate_probability_stats(var_data, var)
                
                if var_stats:
                    comparison_stats.append({'Variable': var, 'Conteo': var_stats['count'], 'Media': var_stats['mean'], 'Desv. Est.': var_stats['std'], 'Mediana': var_stats['median'], 'Asimetría': var_stats['skewness'], 'Curtosis': var_stats['kurtosis']})
            
            if comparison_stats:
                comp_df = pd.DataFrame(comparison_stats)
                st.dataframe(comp_df, use_container_width=True)

if __name__ == "__main__":
    app()