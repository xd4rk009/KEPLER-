import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from .BaseTrainer_opt import BaseTrainer_opt

# =============================================================================
# TabNetNetwork_opt
# =============================================================================
"""
Entrenador para TabNet, hereda de BaseTrainer_opt y utiliza TabNetRegressor para regresión tabular.

Parámetros:
    input_dim (int): Dimensión de entrada.
    device (str): Dispositivo ('cpu' o 'cuda').
    show_info (str): Nivel de información a mostrar.
"""
class TabNetNetwork_opt(BaseTrainer_opt):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    """
    Inicializa el entrenador TabNet y la arquitectura TabNetRegressor.
    """
    def __init__(self, input_dim, device, show_info="True"):
        super().__init__(device, show_info)
        self.model = TabNetRegressor(
            device_name=device,
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-3),
            verbose=10
        )

    # -------------------------------------------------------------------------
    # train_model
    # -------------------------------------------------------------------------
    """
    Entrena la red TabNet y selecciona el mejor modelo según la métrica de validación.
    Args:
        X_train, y_train: Datos de entrenamiento.
        X_val, y_val: Datos de validación.
        max_epochs (int): Número máximo de épocas.
        patience (int): Paciencia para early stopping.
    Returns:
        TabNetRegressor: Modelo entrenado.
    """
    def train_model(self, X_train, y_train, X_val, y_val, max_epochs=1000, patience=50):
        net_name = "TabNet"
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        self._print("Entrenando red...", level="start", net_name=net_name)
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        for epoch in range(1, max_epochs + 1):
            self.model.fit(
                X_train=X_train,
                y_train=y_train,
                eval_set=[(X_val, y_val)],
                eval_name=["validación"],
                eval_metric=["rmse"],
                max_epochs=1,
                patience=0,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            train_preds = self.model.predict(X_train).flatten()
            val_preds = self.model.predict(X_val).flatten()
            train_loss = np.sqrt(np.mean((y_train.flatten() - train_preds) ** 2))
            val_loss = np.sqrt(np.mean((y_val.flatten() - val_preds) ** 2))
            msg = f"📈 Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}"
            self._print(msg, level="progress", net_name=net_name)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                msg = f"📈 Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} 🎉 Early stopping (mejor val_loss: {best_val_loss:.4f})"
                self._print(msg, level="progress", net_name=net_name)
                break
        self._print("Entrenamiento finalizado", level="stage", net_name=net_name)
        return best_model if best_model is not None else self.model

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    """
    Realiza predicciones usando el modelo TabNet entrenado.
    Args:
        X: Datos de entrada.
    Returns:
        np.ndarray: Predicciones.
    """
    def predict(self, X):
        return self.model.predict(np.asarray(X)).flatten()
