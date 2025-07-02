import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from scipy.stats import pearsonr
import random
import os
from datetime import datetime
from io import BytesIO

# --- INICIO: SECCIÓN DE FUNCIONES (CON NUEVAS ADICIONES) ---

# Directorio para guardar y cargar los modelos
# Consider making this configurable via Streamlit's secrets management or environment variables
# For now, it's hardcoded as in your original script.
MODEL_SAVE_DIR = r"C:\Users\Sergio Arias\Desktop\KEPLER - RAJO - SUBTE\kepler subte (in progress)\Models Streamlit"

def create_regression_plot(y_true, y_pred, title, r_value):
    """Crea un gráfico de dispersión de valores reales vs. predichos."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true.flatten(),
        y=y_pred.flatten(),
        mode='markers',
        name='Predicciones',
        marker=dict(color='#1f77b4', opacity=0.7)
    ))
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Línea Ideal',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=f'<b>{title}</b><br>R = {r_value:.4f}',
        xaxis_title='Valor Real',
        yaxis_title='Valor Predicho',
        height=400,
        showlegend=False,
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig

def create_timeseries_plot(y_true, y_pred, r_value, title, color_real, color_pred):
    """Crea un gráfico de serie de tiempo de valores reales vs. predichos para un set de datos."""
    fig = go.Figure()
    indices = np.arange(len(y_true))
    fig.add_trace(go.Scatter(
        x=indices,
        y=y_true.flatten(),
        mode='lines',
        name='Real',
        line=dict(color=color_real)
    ))
    fig.add_trace(go.Scatter(
        x=indices,
        y=y_pred.flatten(),
        mode='lines',
        name='Predicho',
        line=dict(color=color_pred, dash='dash')
    ))
    fig.update_layout(
        title=f'<b>{title} (R={r_value:.4f})</b>',
        xaxis_title='Índice de Muestra',
        yaxis_title='Valor de Salida',
        height=350,
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_combined_timeseries_plot(y_train, pred_train, idx_train, y_val, pred_val, idx_val, y_test, pred_test, idx_test):
    """Crea un gráfico de serie de tiempo combinado que replica el estilo de MATLAB."""
    fig = go.Figure()

    # Combinar y ordenar todos los datos por el índice original para trazar líneas continuas
    all_indices = np.concatenate([idx_train, idx_val, idx_test])
    all_y_true = np.concatenate([y_train, y_val, y_test])
    all_y_pred = np.concatenate([pred_train.flatten(), pred_val.flatten(), pred_test.flatten()])
    
    sort_order = np.argsort(all_indices)
    sorted_indices = all_indices[sort_order]
    sorted_y_true = all_y_true[sort_order]
    sorted_y_pred = all_y_pred[sort_order]

    # Trazar la línea completa de valores reales y predichos
    fig.add_trace(go.Scatter(
        x=sorted_indices, y=sorted_y_true,
        mode='lines+markers', name='Real',
        line=dict(color='blue'),
        marker=dict(size=4, symbol='circle'),
        legendgroup='1'
    ))
    fig.add_trace(go.Scatter(
        x=sorted_indices, y=sorted_y_pred,
        mode='lines', name='Predicho',
        line=dict(color='red', dash='dash'),
        legendgroup='1'
    ))
    
    # Añadir marcadores para la leyenda, simulando el gráfico de MATLAB
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Entrenamiento', marker=dict(color='blue', size=8)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Validación', marker=dict(color='green', size=8)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Test', marker=dict(color='red', size=8)))
    
    # Superponer los puntos de validación y test para destacarlos
    fig.add_trace(go.Scatter(x=idx_val, y=y_val, mode='markers', showlegend=False, marker=dict(color='green', size=6, line=dict(width=1, color='black'))))
    fig.add_trace(go.Scatter(x=idx_test, y=y_test, mode='markers', showlegend=False, marker=dict(color='red', size=6, line=dict(width=1, color='black'))))

    fig.update_layout(
        title='<b>Comparación Completa: Real vs Predicho</b>',
        xaxis_title='Índice de Muestra Global',
        yaxis_title='Valor de Salida',
        height=350,
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def save_model_bundle(model, scaler_X, scaler_y, selected_features, use_poly, poly_degree, best_metrics, hidden_layers, dropout_rate, activation_choice):
    """
    Guarda el estado del modelo, métricas y objetos necesarios para la predicción.
    """
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        st.info(f"Directorio creado: {MODEL_SAVE_DIR}")

    r_train = best_metrics.get('train_r', 0)
    r_val = best_metrics.get('val_r', 0)
    r_test = best_metrics.get('test_r', 0)

    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    filename = f"model_{timestamp} - Rtrain {r_train:.2f} - Rval {r_val:.2f} - Rtest {r_test:.2f}.pt"
    filepath = os.path.join(MODEL_SAVE_DIR, filename)

    bundle = {
        'model_state_dict': model.state_dict(),
        'scaler_X_state': scaler_X,
        'scaler_y_state': scaler_y,
        'selected_features': selected_features,
        'use_poly': use_poly,
        'poly_degree': poly_degree,
        'best_metrics': best_metrics,
        'residual_std_log': best_metrics.get('residual_std_log', 0),
        'input_size': model.network[0].in_features,
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'activation_fn': activation_choice
    }

    try:
        torch.save(bundle, filepath)
        st.success(f"✅ ¡Mejor modelo guardado exitosamente con R Train: {r_train:.4f}, R Val: {r_val:.4f}, R Test: {r_test:.4f}!")
        st.info(f"Ruta del archivo: {filepath}")
        return filepath
    except Exception as e:
        st.error(f"❌ Error al guardar el modelo: {e}")
        return None

def predict_with_confidence_bands(model, scaler_X, scaler_y, df_to_predict, selected_features, use_poly, poly_degree, device, residual_std_log):
    """
    Realiza predicciones, calcula bandas de confianza y deriva MwEQ.
    Esta función ha sido modificada para calcular los intervalos de confianza en el espacio logarítmico
    y luego transformarlos de nuevo a la escala original, evitando valores negativos y logaritmos de cero.
    """
    missing_cols = [col for col in selected_features if col not in df_to_predict.columns]
    if missing_cols:
        st.error(f"Faltan las siguientes columnas en el archivo subido: {', '.join(missing_cols)}")
        return None

    X_raw = df_to_predict[selected_features].values
    
    if use_poly:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
        # It's important to fit_transform on training data and then just transform on new data.
        # Since we load scaler_X_state which was fitted during training, poly should also be handled consistently.
        # If poly is recreated here, it should only be transformed. Assuming poly_degree is part of the loaded model's config.
        # For prediction, poly should NOT be fitted again on the new data. This could lead to incorrect transformations.
        # The scaler_X should have been fitted on the polynomial features from the training data.
        # So, the polynomial transformation should happen BEFORE scaling with scaler_X.
        
        # A more robust approach would be to save the `PolynomialFeatures` object itself
        # or at least ensure the column order/number is consistent.
        # For simplicity and given your original code's structure, we'll assume `poly_degree`
        # is enough to recreate a compatible `PolynomialFeatures` object for *transformation*.
        # However, for a real-world scenario, you might want to save the fitted `poly` object.
        dummy_poly_fitter = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
        # We need to fit this dummy to SOME data to get the feature names/ordering if we
        # were to enforce strict consistency, but for transformation it's just about the degree.
        # The key is that scaler_X was fitted on the *output* of PolynomialFeatures.
        
        # A safer way: if the model was trained with poly features, then the input_size of the model
        # already reflects the number of polynomial features. When predicting, we just need to generate
        # these same polynomial features.
        X_poly = dummy_poly_fitter.fit_transform(X_raw) # This fit_transform is OK here for *generating* the features
                                                     # as long as we don't fit the scaler_X again.
        st.info(f"Aplicando transformación polinómica de grado {poly_degree} a los datos de entrada para predicción.")
    else:
        X_poly = X_raw

    X_scaled = scaler_X.transform(X_poly) # Use the loaded scaler_X, which was fitted on training polynomial features
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_tensor)
        # Obtener predicciones en el espacio logarítmico (des-escalado por MinMaxScaler)
        predictions_log = scaler_y.inverse_transform(predictions_scaled.cpu().numpy())
    
    # Calcular bandas de confianza en el espacio logarítmico
    lower_bound_log = predictions_log - 1.96 * residual_std_log
    upper_bound_log = predictions_log + 1.96 * residual_std_log
    
    # Transformar todo de vuelta a la escala original de Cumulative_Mo usando la exponencial
    # Esto garantiza que todos los valores (predicción y límites) sean positivos.
    mean_preds_orig = np.exp(predictions_log.flatten())
    lower_bound_mo = np.exp(lower_bound_log.flatten())
    upper_bound_mo = np.exp(upper_bound_log.flatten())
    
    results_df = df_to_predict.copy()
    results_df['Cumulative_Mo_Predicho'] = mean_preds_orig
    results_df['Mo_Intervalo_Inferior'] = lower_bound_mo
    results_df['Mo_Intervalo_Superior'] = upper_bound_mo
    
    # Calcular MwEQ y su intervalo de confianza.
    # No se necesitan verificaciones de np.maximum porque np.exp() asegura positividad.
    # Se agrega un epsilon muy pequeño por seguridad numérica contra posibles ceros de punto flotante.
    epsilon = 1e-20
    results_df['MwEQ_Predicho'] = (2/3) * np.log10(mean_preds_orig + epsilon) - 6.21
    results_df['MwEQ_Intervalo_Inferior'] = (2/3) * np.log10(lower_bound_mo + epsilon) - 6.21
    results_df['MwEQ_Intervalo_Superior'] = (2/3) * np.log10(upper_bound_mo + epsilon) - 6.21
    
    return results_df


def to_excel(df):
    """Convierte un DataFrame a un archivo Excel en memoria."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicciones')
    processed_data = output.getvalue()
    return processed_data

def display_prediction_results(predictions_df, file_prefix):
    """
    Muestra los resultados de la predicción, incluyendo gráficos de IC para Mo y MwEQ,
    opciones de descarga y gráficos 2D/3D personalizables.
    """
    st.subheader("Resultados de la Predicción con Bandas de Confianza del 95%")
    
    plot_col1, plot_col2 = st.columns(2)
    x_axis_indices = predictions_df.index
    
    # --- Gráfico para Cumulative Mo ---
    with plot_col1:
        fig_mo = go.Figure()
        fig_mo.add_trace(go.Scatter(
            x=np.concatenate([x_axis_indices, x_axis_indices[::-1]]),
            y=np.concatenate([predictions_df['Mo_Intervalo_Superior'], predictions_df['Mo_Intervalo_Inferior'][::-1]]),
            fill='toself', fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", name='IC 95% Mo'
        ))
        fig_mo.add_trace(go.Scatter(
            x=x_axis_indices, y=predictions_df['Cumulative_Mo_Predicho'],
            mode='lines+markers', line=dict(color='rgb(0,176,246)'), name='Predicción Mo'
        ))
        fig_mo.update_layout(
            title="<b>Predicción: Cumulative_Mo</b>",
            xaxis_title="Índice de Fila de Datos", yaxis_title="Cumulative_Mo", height=450
        )
        st.plotly_chart(fig_mo, use_container_width=True, key=f"pred_mo_plot_{file_prefix}")

    # --- Gráfico para MwEQ ---
    with plot_col2:
        fig_mweq = go.Figure()
        fig_mweq.add_trace(go.Scatter(
            x=np.concatenate([x_axis_indices, x_axis_indices[::-1]]),
            y=np.concatenate([predictions_df['MwEQ_Intervalo_Superior'], predictions_df['MwEQ_Intervalo_Inferior'][::-1]]),
            fill='toself', fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", name='IC 95% MwEQ'
        ))
        fig_mweq.add_trace(go.Scatter(
            x=x_axis_indices, y=predictions_df['MwEQ_Predicho'],
            mode='lines+markers', line=dict(color='rgb(255,127,14)'), name='Predicción MwEQ'
        ))
        fig_mweq.update_layout(
            title="<b>Predicción: MwEQ (Magnitud Equivalente)</b>",
            xaxis_title="Índice de Fila de Datos", yaxis_title="MwEQ", height=450
        )
        st.plotly_chart(fig_mweq, use_container_width=True, key=f"pred_mweq_plot_{file_prefix}")

    st.write("Datos de la Predicción (incluye Mo y MwEQ):")
    st.dataframe(predictions_df)
    
    # --- SECCIÓN MEJORADA: Botones de Descarga ---
    st.markdown("##### Descargar Resultados")
    col1, col2 = st.columns(2)
    csv = predictions_df.to_csv(index=False).encode('utf-8')
    col1.download_button("📥 Descargar en formato CSV", csv, f"{file_prefix}.csv", "text/csv")
    
    try:
        excel_data = to_excel(predictions_df)
        col2.download_button(
            "📥 Descargar en formato Excel", excel_data, f"{file_prefix}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except ImportError:
        with col2:
            st.warning("Para descargar en Excel, instala la librería: `pip install xlsxwriter`")

    # --- NUEVA SECCIÓN: Gráficos Personalizados con Múltiples Ejes Y ---
    st.markdown("---")
    st.subheader("🔬 Visualización Interactiva de Datos")
    st.markdown("Selecciona variables para graficar. En 2D puedes seleccionar múltiples ejes Y.")
    
    plot_options = predictions_df.columns.tolist()
    c1, c2 = st.columns(2)
    plot_dim = c1.radio("Dimensiones del Gráfico", ('2D', '3D'), key=f"dim_{file_prefix}")
    plot_type_map = {'Puntos': 'markers', 'Línea': 'lines', 'Línea y Puntos': 'lines+markers'}
    plot_type_choice = c2.radio("Tipo de Trazo", list(plot_type_map.keys()), key=f"type_{file_prefix}")
    plot_mode = plot_type_map[plot_type_choice]

    if plot_dim == '2D':
        c1_plot, c2_plot, c3_plot = st.columns(3) 
        
        x_axis = c1_plot.selectbox("Eje X", plot_options, index=0, key=f"2d_x_{file_prefix}")
        
        default_y = []
        if 'Cumulative_Mo_Predicho' in plot_options: default_y.append('Cumulative_Mo_Predicho')
        if 'MwEQ_Predicho' in plot_options: default_y.append('MwEQ_Predicho')
        
        # Ensure default selected values are valid for the options
        y_axes_options = [y for y in plot_options if y in default_y]
        y_axes = c2_plot.multiselect("Eje(s) Y (Izquierdo)", plot_options, default=y_axes_options, key=f"2d_y_{file_prefix}")
        
        secondary_y_options = ["Sin eje secundario"] + plot_options
        secondary_y_axis = c3_plot.selectbox("Eje Y (Derecho - Opcional)", secondary_y_options, index=0, key=f"2d_y2_{file_prefix}")
        
        if x_axis and (y_axes or secondary_y_axis != "Sin eje secundario"):
            fig_2d = make_subplots(specs=[[{"secondary_y": True}]])

            if y_axes:
                for y_axis_col in y_axes: # Renamed to y_axis_col to avoid conflict with the function parameter
                    fig_2d.add_trace(
                        go.Scatter(x=predictions_df[x_axis], y=predictions_df[y_axis_col], mode=plot_mode, name=y_axis_col),
                        secondary_y=False,
                    )

            if secondary_y_axis != "Sin eje secundario":
                fig_2d.add_trace(
                    go.Scatter(x=predictions_df[x_axis], y=predictions_df[secondary_y_axis], mode=plot_mode, name=f"{secondary_y_axis} (der.)"),
                    secondary_y=True,
                )

            primary_y_title = ', '.join(y_axes) if y_axes else "Eje Y Izquierdo"
            
            fig_2d.update_layout(
                title_text='<b>Gráfico 2D Personalizado</b>',
                xaxis_title=x_axis,
                margin=dict(l=40, r=30, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig_2d.update_yaxes(title_text=primary_y_title, secondary_y=False)
            if secondary_y_axis != "Sin eje secundario":
                 fig_2d.update_yaxes(title_text=secondary_y_axis, secondary_y=True, showgrid=False)
            
            st.plotly_chart(fig_2d, use_container_width=True)
        else:
            st.info("Selecciona al menos un eje Y (izquierdo o derecho) para el gráfico 2D.")


    elif plot_dim == '3D':
        c1_plot, c2_plot, c3_plot = st.columns(3)
        x_axis = c1_plot.selectbox("Eje X", plot_options, index=0, key=f"3d_x_{file_prefix}")
        y_axis = c2_plot.selectbox("Eje Y", plot_options, index=min(1, len(plot_options)-1), key=f"3d_y_{file_prefix}")
        z_axis = c3_plot.selectbox("Eje Z", plot_options, index=min(2, len(plot_options)-1), key=f"3d_z_{file_prefix}")

        if x_axis and y_axis and z_axis:
            fig_3d = go.Figure()
            fig_3d.add_trace(go.Scatter3d(x=predictions_df[x_axis], y=predictions_df[y_axis], z=predictions_df[z_axis], mode=plot_mode, name="Data"))
            fig_3d.update_layout(
                title=f'Gráfico 3D: {z_axis} vs. {y_axis} vs. {x_axis}', 
                scene=dict(xaxis_title=x_axis, yaxis_title=y_axis, zaxis_title=z_axis),
                margin=dict(l=20, r=20, t=60, b=20)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Selecciona los tres ejes (X, Y, Z) para el gráfico 3D.")


# --- FIN: SECCIÓN DE FUNCIONES ---


# --- INICIO: SECCIÓN DE CLASES Y FUNCIONES DE ENTRENAMIENTO ---

st.set_page_config(
    page_title="Advanced Neural Network Trainer",
    page_icon="🚀",
    layout="wide"
)

class MATLABStyleNN(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate=0.1, activation_fn='leaky_relu'):
        super(MATLABStyleNN, self).__init__()
        
        activation_functions = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(0.01)
        }
        activation = activation_functions.get(activation_fn.lower(), nn.LeakyReLU(0.01))
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                n = module.in_features
                beta = 0.7 * (module.out_features ** (1.0/n))
                with torch.no_grad():
                    module.weight.uniform_(-1, 1)
                    module.weight.data *= beta
                    if module.bias is not None:
                        module.bias.uniform_(-beta, beta)
    
    def forward(self, x):
        return self.network(x)

def calculate_pearson_r(y_true, y_pred):
    if len(y_true) < 2 or len(y_pred) < 2: return np.nan
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    if np.sum(mask) < 2: return np.nan
    try:
        r, _ = pearsonr(y_true[mask], y_pred[mask])
        return r if not np.isnan(r) else 0.0
    except:
        return np.nan

def train_robust_model(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                       train_idx, val_idx, test_idx,
                       epochs, device, extra_iterations, target_r):
    
    st.info("🔄 Aplicando transformación logarítmica: ln(Cumulative_Mo)")
    y_train_log = np.log(np.maximum(y_train, 1e-10))
    y_val_log = np.log(np.maximum(y_val, 1e-10))
    y_test_log = np.log(np.maximum(y_test, 1e-10)) # Keep this for test R calculation, but not for scaler_y fit

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val_log.reshape(-1, 1))
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    overall_best_r = -np.inf
    overall_best_model_state = None
    overall_best_metrics = {}
    overall_best_predictions = {}
    
    status_message_placeholder = st.empty()
    metrics_container = st.empty()
    plots_container = st.empty()
    progress_bar = st.progress(0)
    
    target_achieved = False

    for iteration in range(extra_iterations):
        st.subheader(f"Intento de Entrenamiento {iteration + 1}/{extra_iterations}")
        
        # Re-initialize model weights for each iteration
        model._initialize_weights()
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, verbose=False)
        criterion = nn.HuberLoss()
        patience = max(150, epochs // 10)
        patience_counter = 0
        
        history = {'train_r': [], 'val_r': [], 'test_r': [], 'combined_r': []}

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                test_outputs = model(X_test_tensor)
                
                # Inverse transform predictions from scaled log space to original Mo scale
                train_pred_orig = np.exp(scaler_y.inverse_transform(train_outputs.cpu().numpy()))
                val_pred_orig = np.exp(scaler_y.inverse_transform(val_outputs.cpu().numpy()))
                test_pred_orig = np.exp(scaler_y.inverse_transform(test_outputs.cpu().numpy()))

                train_r = calculate_pearson_r(y_train, train_pred_orig)
                val_r = calculate_pearson_r(y_val, val_pred_orig)
                test_r = calculate_pearson_r(y_test, test_pred_orig)
                
                valid_rs = [r for r in [train_r, val_r, test_r] if not np.isnan(r)]
                weights = np.array([0.4, 0.3, 0.3][:len(valid_rs)])
                combined_r = np.average(valid_rs, weights=weights / weights.sum()) if valid_rs else -1

            history['train_r'].append(train_r)
            history['val_r'].append(val_r)
            history['test_r'].append(test_r)
            history['combined_r'].append(combined_r)
            
            scheduler.step(val_loss)

            if combined_r > overall_best_r:
                overall_best_r = combined_r
                overall_best_model_state = model.state_dict().copy()
                
                # Calculate residual standard deviation on the LOG-TRANSFORMED values (validation set)
                val_pred_log = scaler_y.inverse_transform(val_outputs.cpu().numpy())
                log_residuals = y_val_log.flatten() - val_pred_log.flatten()
                residual_std_log = np.std(log_residuals)
                
                overall_best_metrics = {
                    'train_r': train_r, 'val_r': val_r, 'test_r': test_r, 
                    'combined_r': combined_r, 
                    'residual_std_log': residual_std_log # Storing the log-space std dev
                }
                overall_best_predictions = {'train': train_pred_orig.copy(), 'val': val_pred_orig.copy(), 'test': test_pred_orig.copy()}
                patience_counter = 0
                
                if epoch > 10:
                    status_message_placeholder.success(f"🎉 Nuevo mejor modelo! R combinado: {overall_best_r:.4f} (Train: {train_r:.4f}, Val: {val_r:.4f}, Test: {test_r:.4f}) - Intento {iteration+1}, Época {epoch}")
            else:
                patience_counter += 1

            if epoch % 25 == 0 or epoch == epochs - 1:
                with metrics_container.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R Train", f"{train_r:.4f}")
                    c2.metric("R Val", f"{val_r:.4f}")
                    c3.metric("R Test", f"{test_r:.4f}")
                    c4.metric("🏆 R Combinado (Actual)", f"{combined_r:.4f}", f"Mejor Global: {overall_best_r:.4f}")

                with plots_container.container():
                    st.markdown("##### Gráficos de Regresión (En Vivo)")
                    plot_cols = st.columns(3)
                    with plot_cols[0]:
                        fig_train = create_regression_plot(y_train, train_pred_orig, "Entrenamiento", train_r)
                        st.plotly_chart(fig_train, use_container_width=True, key=f"live_train_reg_plot_{iteration}_{epoch}")
                    with plot_cols[1]:
                        fig_val = create_regression_plot(y_val, val_pred_orig, "Validación", val_r)
                        st.plotly_chart(fig_val, use_container_width=True, key=f"live_val_reg_plot_{iteration}_{epoch}")
                    with plot_cols[2]:
                        fig_test = create_regression_plot(y_test, test_pred_orig, "Test", test_r)
                        st.plotly_chart(fig_test, use_container_width=True, key=f"live_test_reg_plot_{iteration}_{epoch}")
                    
                    st.markdown("---")
                    st.markdown("##### Evolución del Coeficiente R (En Vivo)")
                    fig_evolution = go.Figure()
                    fig_evolution.add_trace(go.Scatter(y=history['train_r'], name='Train R', line=dict(color='blue')))
                    fig_evolution.add_trace(go.Scatter(y=history['val_r'], name='Val R', line=dict(color='orange')))
                    fig_evolution.add_trace(go.Scatter(y=history['test_r'], name='Test R', line=dict(color='red')))
                    fig_evolution.add_trace(go.Scatter(y=history['combined_r'], name='Combined R', line=dict(color='green', dash='dash')))
                    fig_evolution.update_layout(title=f'Evolución del R de Pearson (Intento {iteration+1})', xaxis_title='Época', yaxis_title='R', height=400)
                    st.plotly_chart(fig_evolution, use_container_width=True, key=f"live_r_evolution_plot_{iteration}_{epoch}")

                    st.markdown("---")
                    st.markdown("##### Comparación en Serie de Tiempo (En Vivo - Estilo MATLAB)")
                    
                    ts_cols_top = st.columns(2)
                    with ts_cols_top[0]:
                        fig_combined_ts = create_combined_timeseries_plot(
                            y_train, train_pred_orig, train_idx,
                            y_val, val_pred_orig, val_idx,
                            y_test, test_pred_orig, test_idx
                        )
                        st.plotly_chart(fig_combined_ts, use_container_width=True, key=f"live_ts_combined_{iteration}_{epoch}")
                    
                    with ts_cols_top[1]:
                        fig_train_ts = create_timeseries_plot(y_train, train_pred_orig, train_r, "ENTRENAMIENTO", 'blue', 'red')
                        st.plotly_chart(fig_train_ts, use_container_width=True, key=f"live_ts_train_{iteration}_{epoch}")

                    ts_cols_bottom = st.columns(2)
                    with ts_cols_bottom[0]:
                        fig_val_ts = create_timeseries_plot(y_val, val_pred_orig, val_r, "VALIDACIÓN", 'green', 'red')
                        st.plotly_chart(fig_val_ts, use_container_width=True, key=f"live_ts_val_{iteration}_{epoch}")

                    with ts_cols_bottom[1]:
                        fig_test_ts = create_timeseries_plot(y_test, test_pred_orig, test_r, "TEST", 'red', 'black')
                        st.plotly_chart(fig_test_ts, use_container_width=True, key=f"live_ts_test_{iteration}_{epoch}")


            total_progress = (iteration * epochs + epoch + 1) / (extra_iterations * epochs)
            progress_bar.progress(total_progress)
            
            if patience_counter >= patience:
                st.warning(f"Parada temprana (Early stopping) en época {epoch}. Pasando al siguiente intento.")
                break
            
            if overall_best_r > target_r and all(r > target_r for r in [overall_best_metrics.get('train_r', 0), overall_best_metrics.get('val_r', 0), overall_best_metrics.get('test_r', 0)]):
                st.balloons()
                st.success(f"🎯 ¡Objetivo R > {target_r} alcanzado en todos los sets! Deteniendo la búsqueda.")
                target_achieved = True
                break
        
        if target_achieved:
            break
            
    if overall_best_model_state:
        model.load_state_dict(overall_best_model_state)
        st.success(f"✅ Modelo cargado con el MEJOR rendimiento encontrado: R = {overall_best_r:.4f}")
        
        final_best_snapshot = {
            'metrics': overall_best_metrics,
            'predictions': overall_best_predictions,
            'true_values': {'train': y_train, 'val': y_val, 'test': y_test}
        }
    else:
        st.error("❌ No se encontró ningún modelo que mejorara el rendimiento inicial.")
        final_best_snapshot = {}
    
    return {'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y, 'best_snapshot': final_best_snapshot}

# --- FIN: SECCIÓN DE ENTRENAMIENTO ---


# --- INICIO: FUNCIÓN PRINCIPAL `main` ---

def main():
    st.title("🚀 Entrenador y Predictor de Redes Neuronales")
    st.markdown("### Entrena, guarda, carga y predice el Momento Total por Tronadura.")
    st.markdown("---")
    
    # Initialize session state variables
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'current_model_for_prediction' not in st.session_state:
        st.session_state.current_model_for_prediction = None
    if 'predictions_df' not in st.session_state: # New: to store prediction results for plotting
        st.session_state.predictions_df = None
    if 'current_file_prefix' not in st.session_state: # New: to store file prefix for plots
        st.session_state.current_file_prefix = None


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.success(f"✅ Dispositivo detectado: **{str(device).upper()}**")
    
    st.header("⚙️ 1. Entrenar un Modelo")
    
    with st.sidebar:
        st.header("Configurar Experimento")
        uploaded_file = st.file_uploader("Cargar archivo Excel de entrenamiento", type=['xlsx', 'xls'])
        
        if uploaded_file:
            st.subheader("Variables Predictoras (X)")
            try:
                # Read columns without loading the whole dataframe initially
                df_cols = pd.read_excel(uploaded_file, nrows=0).columns.tolist()
                numeric_cols = df_cols[:] # Create a copy
                if 'Cumulative_Mo' in numeric_cols:
                    numeric_cols.remove('Cumulative_Mo')
                selected_features = st.multiselect("Seleccionar variables:", numeric_cols, default=numeric_cols)
            except Exception as e:
                st.error(f"No se pudieron leer las columnas: {e}")
                selected_features = []

            st.subheader("🧬 Feature Engineering")
            use_poly = st.checkbox("Crear características polinómicas", True)
            poly_degree = st.slider("Grado del polinomio", 2, 4, 2, disabled=not use_poly)
            st.subheader("📊 División de Datos")
            train_ratio = st.slider("% Entrenamiento", 0.5, 0.9, 0.7, 0.05)
            val_ratio = st.slider("% Validación", 0.05, 0.25, 0.15, 0.05)
            test_ratio = round(1 - train_ratio - val_ratio, 2)
            st.info(f"% Test: {test_ratio*100:.0f}%")
            st.subheader("🏗️ Arquitectura de la Red")
            hidden_layers_str = st.text_input("Capas ocultas (ej: 12,10,8):", "12,10,8,6,4")
            dropout_rate = st.slider("Tasa de Dropout (robustez):", 0.0, 0.5, 0.1, 0.05)
            activation_choice = st.selectbox("Función de Activación", ['Tanh', 'ReLU', 'LeakyReLU'], index=2)
            st.subheader("⏱️ Parámetros de Entrenamiento")
            epochs = st.number_input("Épocas por intento", 200, 10000, 2000, 100)
            extra_iterations = st.number_input("Máximo de intentos", 1, 5000, 10, 1)
            target_r = st.slider("R objetivo", 0.7, 0.99, 0.80, 0.01)
            random_seed = st.number_input("Semilla aleatoria", 1, 1000, 42)
            
            train_button_pressed = st.button("🚀 Entrenar Modelo", type="primary")
        else:
            st.warning("Por favor, carga un archivo Excel para configurar el entrenamiento.")
            train_button_pressed = False

    training_execution_placeholder = st.container()

    if train_button_pressed:
        with training_execution_placeholder:
            try:
                df = pd.read_excel(uploaded_file)
                if 'Cumulative_Mo' not in df.columns:
                    st.error("Columna 'Cumulative_Mo' no encontrada en el archivo de entrenamiento."); return
            except Exception as e:
                st.error(f"Error al leer el archivo Excel: {e}"); return

            torch.manual_seed(random_seed); np.random.seed(random_seed); random.seed(random_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed(random_seed)
            if not selected_features:
                st.error("Por favor, selecciona al menos una variable predictora."); return
            
            data = df[selected_features + ['Cumulative_Mo']].dropna()
            if use_poly:
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X_raw = data[selected_features].values
                X = poly.fit_transform(X_raw)
                st.info(f"🧬 Características expandidas de {X_raw.shape[1]} a {X.shape[1]}.")
            else:
                X = data[selected_features].values
            y = data['Cumulative_Mo'].values
            
            # Ensure proper splits and use shuffle=True for reproducibility with random_state
            train_idx, temp_idx = train_test_split(np.arange(len(X)), test_size=(1-train_ratio), random_state=random_seed, shuffle=True)
            # Adjust val_size_adj based on the remaining data after training split
            if (val_ratio + test_ratio) > 0: # Avoid division by zero if both are zero
                val_size_adj = val_ratio / (val_ratio + test_ratio)
            else:
                val_size_adj = 0 # No validation or test data
            
            if len(temp_idx) > 0 and val_size_adj > 0:
                val_idx, test_idx = train_test_split(temp_idx, test_size=(1-val_size_adj), random_state=random_seed, shuffle=True)
            else: # No validation/test data or only one split
                val_idx, test_idx = [], temp_idx # If no val, all remaining go to test (if temp_idx exists)
                if val_ratio > 0: # If only val_ratio is set but test_ratio is 0
                    val_idx = temp_idx
                    test_idx = []

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx] if len(val_idx) > 0 else np.array([])
            X_test, y_test = X[test_idx], y[test_idx] if len(test_idx) > 0 else np.array([])


            st.success(f"Datos divididos: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(',')]
            model = MATLABStyleNN(X.shape[1], hidden_layers, dropout_rate, activation_choice).to(device)
            st.code(f"Usando HuberLoss, Activación: {activation_choice}\n{str(model)}")
            
            start_time = time.time()
            results = train_robust_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test, 
                train_idx, val_idx, test_idx, 
                epochs, device, extra_iterations, target_r
            )
            st.success(f"✅ Búsqueda completada en {time.time() - start_time:.1f}s")
            
            st.session_state.training_results = results
            st.session_state.training_results['config'] = {
                'selected_features': selected_features, 'use_poly': use_poly, 'poly_degree': poly_degree,
                'hidden_layers': hidden_layers, 'dropout_rate': dropout_rate, 'activation_choice': activation_choice
            }
            # Clear previous prediction results when retraining a model
            st.session_state.predictions_df = None
            st.session_state.current_file_prefix = None


    if st.session_state.training_results:
        st.markdown("---")
        st.header("📊 Resultados Finales del Mejor Modelo Entrenado")
        
        results = st.session_state.training_results
        
        if 'best_snapshot' in results and results['best_snapshot']:
            best_snapshot = results['best_snapshot']
            final_metrics = best_snapshot['metrics']
            config = results.get('config', {})
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 R Entrenamiento", f"{final_metrics.get('train_r', 0):.4f}")
            c2.metric("🔍 R Validación", f"{final_metrics.get('val_r', 0):.4f}")
            c3.metric("🎪 R Test", f"{final_metrics.get('test_r', 0):.4f}")
            c4.metric("🏆 R Combinado (Mejor)", f"{final_metrics.get('combined_r', 0):.4f}")
            
            st.markdown("---")
            st.subheader("Gráficos de Regresión Finales del Mejor Modelo")
            
            plot_cols = st.columns(3)
            preds = best_snapshot['predictions']
            trues = best_snapshot['true_values']

            with plot_cols[0]:
                fig_train = create_regression_plot(trues['train'], preds['train'], "Entrenamiento", final_metrics['train_r'])
                st.plotly_chart(fig_train, use_container_width=True, key="final_train_reg_plot")
            with plot_cols[1]:
                fig_val = create_regression_plot(trues['val'], preds['val'], "Validación", final_metrics['val_r'])
                st.plotly_chart(fig_val, use_container_width=True, key="final_val_reg_plot")
            with plot_cols[2]:
                fig_test = create_regression_plot(trues['test'], preds['test'], "Test", final_metrics['test_r'])
                st.plotly_chart(fig_test, use_container_width=True, key="final_test_reg_plot")
            
            final_model = results['model']
            if st.button("💾 Guardar este Modelo", key="save_final"):
                    save_model_bundle(final_model, results['scaler_X'], results['scaler_y'], 
                                    config['selected_features'], config['use_poly'], config['poly_degree'], 
                                    final_metrics, config['hidden_layers'], config['dropout_rate'], 
                                    config['activation_choice'])
            
            st.session_state.current_model_for_prediction = {
                'model': final_model, 'scaler_X': results['scaler_X'], 'scaler_y': results['scaler_y'],
                'selected_features': config['selected_features'], 'use_poly': config['use_poly'],
                'poly_degree': config['poly_degree'], 'device': device,
                'residual_std_log': final_metrics.get('residual_std_log', 0)
            }
        else:
            st.warning("El entrenamiento no produjo un modelo válido.")
    
    st.markdown("---")
    st.header("🔮 2. Usar un Modelo para Predecir")
    st.markdown("Una vez que hayas entrenado un modelo o si tienes uno guardado, puedes usar las siguientes opciones.")

    tab1, tab2 = st.tabs(["Opción 1: Predecir con el Modelo Actual", "Opción 2: Cargar Modelo Guardado y Predecir"])

    with tab1:
        st.subheader("Predecir con el Modelo Recién Entrenado")
        if not st.session_state.current_model_for_prediction:
            st.warning("Aún no has entrenado un modelo en esta sesión. Por favor, entrena un modelo primero.")
        else:
            st.info("El mejor modelo de la sesión de entrenamiento actual está listo para ser usado.")
            # Added a unique key for this file_uploader to avoid conflicts
            predict_file_1 = st.file_uploader("Cargar Excel con datos para predecir (Opción 1)", type=['xlsx', 'xls'], key="pred_uploader_1")
            
            # Button to trigger prediction for Option 1
            if predict_file_1 and st.button("Realizar Predicción (Modelo Actual)", key="predict_option1_button"):
                try:
                    df_to_predict = pd.read_excel(predict_file_1)
                    st.write("Vista previa de los datos a predecir:", df_to_predict.head())
                    results = st.session_state.current_model_for_prediction
                    
                    predictions_df = predict_with_confidence_bands(
                        results['model'], results['scaler_X'], results['scaler_y'],
                        df_to_predict, results['selected_features'], results['use_poly'],
                        results['poly_degree'], results['device'], results['residual_std_log']
                    )
                    
                    if predictions_df is not None:
                        st.session_state.predictions_df = predictions_df
                        st.session_state.current_file_prefix = "predicciones_modelo_actual"
                        st.success("Predicción realizada exitosamente.")
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar el archivo de predicción: {e}")

    with tab2:
        st.subheader("Cargar un Modelo desde Archivo y Predecir")
        if not os.path.exists(MODEL_SAVE_DIR):
            st.warning(f"El directorio de modelos '{MODEL_SAVE_DIR}' no existe. Guarda un modelo primero.")
        else:
            try:
                saved_models = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith(".pt")]
                if not saved_models:
                    st.warning("No se encontraron modelos guardados en el directorio.")
                else:
                    selected_model_file = st.selectbox("Selecciona un modelo guardado:", sorted(saved_models, reverse=True), key="model_selectbox")
                    predict_file_2 = st.file_uploader("Cargar Excel con datos para predecir (Opción 2)", type=['xlsx', 'xls'], key="pred_uploader_2")
                    
                    if st.button("Cargar Modelo y Predecir", key="load_and_predict_button"): # Renamed key
                        if predict_file_2 and selected_model_file:
                            try:
                                model_path = os.path.join(MODEL_SAVE_DIR, selected_model_file)
                                bundle = torch.load(model_path, map_location=device)
                                st.success(f"Modelo '{selected_model_file}' cargado exitosamente.")

                                if 'best_metrics' in bundle:
                                    bm = bundle['best_metrics']
                                    st.info(f"Métricas del modelo cargado -> R Train: {bm.get('train_r', 0):.4f}, R Val: {bm.get('val_r', 0):.4f}, R Test: {bm.get('test_r', 0):.4f}")

                                loaded_model = MATLABStyleNN(
                                    input_size=bundle['input_size'], hidden_layers=bundle['hidden_layers'],
                                    dropout_rate=bundle['dropout_rate'], activation_fn=bundle['activation_fn']
                                ).to(device)
                                loaded_model.load_state_dict(bundle['model_state_dict'])

                                df_to_predict = pd.read_excel(predict_file_2)
                                
                                predictions_df = predict_with_confidence_bands(
                                    loaded_model, bundle['scaler_X_state'], bundle['scaler_y_state'],
                                    df_to_predict, bundle['selected_features'], bundle['use_poly'],
                                    bundle['poly_degree'], device, bundle.get('residual_std_log', 0)
                                )

                                if predictions_df is not None:
                                    st.session_state.predictions_df = predictions_df
                                    st.session_state.current_file_prefix = f"predicciones_{os.path.splitext(selected_model_file)[0]}"
                                    st.success("Predicción realizada exitosamente.")

                            except Exception as e:
                                st.error(f"Error al cargar el modelo o predecir: {e}")
                        else:
                            st.warning("Por favor, selecciona un modelo y sube un archivo Excel para la predicción.")
            except Exception as e:
                st.error(f"No se pudo acceder al directorio de modelos. Verifica que la ruta es correcta y tienes permisos. Ruta: {MODEL_SAVE_DIR}. Error: {e}")

    # Display results only if predictions_df exists in session state
    if st.session_state.predictions_df is not None:
        display_prediction_results(st.session_state.predictions_df, st.session_state.current_file_prefix)

if __name__ == "__main__":
    main()