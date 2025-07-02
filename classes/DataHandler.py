import stat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from classes.DuckDB_Helper_v02 import DuckDBHelper
from typing import Dict, Tuple
import torch
import plotly.graph_objects as go
import statsmodels.api as sm

# =============================================================================
# Clase para manejar la carga y procesamiento de datos (optimizada)
# =============================================================================
class DataHandler:
    def __init__(self, target_cols=""):
        """Constructor de la clase DataHandler.
        Args:
            target_cols (list): Lista de nombres de columnas objetivo.
        """
        # Configuración de base de datos
        self.config: Dict[str, any] = {
            "DB_NAME": "data/Tronaduras_vs_Sismicidad.db",
            "SCHEMA_TRAIN": "Train_Data",
            "SCHEMA_PROCESSED": "Processed_Data", 
            "SCHEMA_RAW": "Raw_Data",
            "BASE_TABLE": "Tabla_Unificada",
            "VARS_DESCRIPTION": "Variables_Description",
            "TABLE_CONFIG_I_O": "Configuration_I_O"
        }
        self.config["TARGET_NAMES"] = target_cols
        self.input_scaled = None
        self.output_scaled = None
        self.input_scaler = None
        self.output_scaler = None
        # Configurar logging
        self.setup_logging()
        # Configuración de hardware y entorno
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # <= SIEMPRE DEFINIDO
        # Inicializar db_helper
        self.db_helper = DuckDBHelper(self.config["DB_NAME"])

    def set_targets(self, target_cols: list):
        """Actualizar los nombres de las columnas objetivo."""
        self.config["TARGET_NAMES"] = target_cols
        self.logger.info(f"Columnas objetivo actualizadas: {target_cols}")
        return self

    def setup_logging(self):
        """Configurar el sistema de logging para la clase DataHandler."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuración de logging inicializada correctamente.")

    @classmethod
    def get_logger(cls):
        """Método de clase para obtener el logger."""
        logger = logging.getLogger(cls.__name__)
        if not logger.handlers:
            instance = cls(target_cols=[])
            logger = instance.logger
        return logger

    def load_data(self):
        """Cargar datos desde DuckDB."""
        self.logger.info("Inicio de la carga de datos.")
        try:
            # Si necesitas columnas específicas para tablas grandes, podrías modificarse aquí
            table_unified = self.db_helper.select_df(
                table=self.config["BASE_TABLE"],
                schema=self.config["SCHEMA_PROCESSED"]
            )
            self.logger.info(f"Tabla {self.config['BASE_TABLE']} cargada correctamente.")

            vars_desc = self.db_helper.select_df(
                table=self.config["VARS_DESCRIPTION"],
                schema=self.config["SCHEMA_RAW"]
            )
            self.logger.info(f"Tabla {self.config['VARS_DESCRIPTION']} cargada correctamente.")
            
            self.db_helper.close_connection()
            self.logger.info("Conexión cerrada.")
            
            self.logger.info("Datos cargados exitosamente.")
            return table_unified, vars_desc
        except Exception as e:
            self.logger.error(f"Error al cargar los datos: {str(e)}")
            raise

    def preprocess_data(self, vars_desc: pd.DataFrame, table_unified: pd.DataFrame) -> Tuple[pd.DataFrame]:
        """Preprocesamiento de variables de entrada."""
        self.logger.info("Inicio del preprocesamiento de datos.")
        try:
            # Selección de variables de entrada en una sola pasada (optimizado)
            vars_inputs = vars_desc[
                (vars_desc['Conjunto'] == 'Tronaduras') &
                vars_desc['Uso'].str.lower().isin(['si', 'sí'])
            ][['Variable', 'Descripción']]
            # Toma sólo columnas que existan
            input_vars = [var for var in vars_inputs['Variable'] if var in table_unified.columns]
            inputs = table_unified[input_vars].copy()  # vectorizado, rápido
            drop_cols = ['N° Disparo', 'PK']
            inputs = inputs.drop(columns=drop_cols, errors='ignore')  # drop es vectorizado
            self.logger.info(f"Variables de entrada procesadas: {inputs.columns.tolist()}")
            # Preprocesamiento de variables target
            target_cols = self.config["TARGET_NAMES"]
            targets = table_unified[target_cols].copy()
            # Evita SettingWithCopyWarning usando assign
            if 'Mo_cumulative' in target_cols:
                targets = targets.assign(Mo_cumulative=(2/3) * np.log10(targets['Mo_cumulative']) - 6.1)
            self.logger.info(f"Variables target seleccionadas y transformadas: {target_cols}")
            self.logger.info("Preprocesamiento de datos completado exitosamente.")
            return inputs, targets
        except Exception as e:
            self.logger.error(f"Error al preprocesar los datos: {str(e)}")
            raise

    def normalize_data(self, inputs: pd.DataFrame, targets: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Normalizar datos de entrada y salida y obtener las escalas min-max."""
        self.logger.info("Inicio de la normalización de datos.")
        try:
            if inputs is None or targets is None:
                raise ValueError("Primero debes cargar y preprocesar los datos.")
            # Almacenar los escaladores para su uso futuro
            self.inputs_scaler = MinMaxScaler()
            self.targets_scaler = MinMaxScaler()
            # Escalar las entradas
            inputs_np = inputs.values.astype(float)
            inputs_scaled = self.inputs_scaler.fit_transform(inputs_np)
            self.logger.info("Datos de entrada normalizados correctamente.")
            # Escalar las salidas
            targets_np = targets.values.astype(float)
            targets_scaled = self.targets_scaler.fit_transform(targets_np)
            self.logger.info("Datos de salida normalizados correctamente.")
            # Crear un DataFrame con las escalas min-max
            scales_data = {
                'Variable': list(inputs.columns) + list(targets.columns),
                'Type': ['Input'] * len(inputs.columns) + ['Target'] * len(targets.columns),
                'Min': list(self.inputs_scaler.data_min_) + list(self.targets_scaler.data_min_),
                'Max': list(self.inputs_scaler.data_max_) + list(self.targets_scaler.data_max_)
            }
            scales_df = pd.DataFrame(scales_data)
            self.logger.info("DataFrame de escalas generado correctamente.")
            self.logger.info("Normalización de datos completada exitosamente.")
            return inputs_scaled, targets_scaled, scales_df
        except Exception as e:
            self.logger.error(f"Error al normalizar los datos: {str(e)}")
            raise

    def torch_tensor(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convertir datos a tensores de PyTorch."""
        self.logger.info("Inicio de la conversión a tensores de PyTorch.")
        try:
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.DEVICE)
            targets_tensor = torch.tensor(targets, dtype=torch.float32).to(self.DEVICE)
            self.logger.info("Conversión a tensores completada exitosamente.")
            return inputs_tensor, targets_tensor
        except Exception as e:
            self.logger.error(f"Error al convertir a tensores: {str(e)}")
            raise

    def plot_results(self, data1, data2, title, size=(1200, 800)):
        # (Sin cambios, es solo visualización)
        y1, nombre_y1 = data1
        y2, nombre_y2 = data2
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(y1))),
            y=y1.flatten(),
            mode='lines',
            name=nombre_y1,
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(y2))),
            y=y2.flatten(),
            mode='lines',
            name=nombre_y2,
            line=dict(color='red')
        ))
        h = size[1]
        w = size[0]
        fig.update_layout(
            title=title,
            xaxis_title='Indices',
            yaxis_title='Valor',
            legend_title='Leyenda',
            height=h,
            width=w,
            showlegend=True
        )
        fig.show()
        pass

    def plot_with_confidence_interval(self, data_real, data_pred, title, size=(1200, 800)):
        # (Sin cambios, solo visualización)
        if len(data_pred) == 4:
            y_pred, lower_bound, upper_bound, name_pred = data_pred
        else:
            raise ValueError("data_pred debe contener 4 elementos: (valores, lower_bound, upper_bound, nombre)")
        y_real, name_real = data_real
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(y_real))),
            y=y_real.flatten(),
            mode='lines',
            name=name_real,
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred.flatten(),
            mode='lines',
            name=name_pred,
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(upper_bound))),
            y=upper_bound.flatten(),
            mode='lines',
            line=dict(width=0),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(lower_bound))),
            y=lower_bound.flatten(),
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            showlegend=False,
            name='Intervalo de Confianza'
        ))
        h = size[1]
        w = size[0]
        fig.update_layout(
            title=title,
            xaxis_title='Indices',
            yaxis_title='Valor',
            legend_title='Leyenda',
            height=h,
            width=w,
            showlegend=True
        )
        fig.show()
        pass

    def plot_regression(self, y_true, y_pred, title="Predicciones vs Reales", size=(1200, 800)):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        X = sm.add_constant(y_true)
        model = sm.OLS(y_pred, X).fit()
        intercept, slope = model.params
        r_squared = model.rsquared
        x_vals = np.linspace(min(y_true), max(y_true), 100)
        y_vals = intercept + slope * x_vals
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred,
            mode='markers',
            name='Predicciones',
            marker=dict(color='blue', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            name='Regresión lineal',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=x_vals,
            mode='lines',
            name='Línea Perfecta (y = x)',
            line=dict(color='green', dash='dot')
        ))
        eq_text = f"y = {slope:.3f}x + {intercept:.3f}<br>R² = {r_squared:.3f}"
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            text=eq_text,
            showarrow=False,
            font=dict(size=14, color="black"),
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        fig.update_layout(
            title=title,
            xaxis_title="Valores Reales",
            yaxis_title="Valores Predichos",
            width=size[0],
            height=size[1],
            template="plotly_white"
        )
        fig.show()

    def denormalize_value(self, scales_df: pd.DataFrame, normalized_value: float, variable_name: str) -> float:
        try:
            variable_min = scales_df.loc[scales_df['Variable'] == variable_name, 'Min'].values[0]
            variable_max = scales_df.loc[scales_df['Variable'] == variable_name, 'Max'].values[0]
            denormalized_value = normalized_value * (variable_max - variable_min) + variable_min
            return denormalized_value
        except IndexError:
            raise ValueError(f"La variable {variable_name} no existe en el DataFrame de escalas.")
        pass

    def momentum_apply(self, value):
        return 10 ** ((value + 6.1) * 1.5)

    def momentum_inverse(self, value):
        return (np.log10(value) / 1.5) - 6.1

    def evaluar_metricas(self, y_true, y_pred, nombre_modelo="Modelo"):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n📊 Métricas para {nombre_modelo}:")
        print(f"   MAE  : {mae:.4f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   R²   : {r2:.4f}")
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
