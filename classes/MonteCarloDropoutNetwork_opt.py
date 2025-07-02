import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .BaseTrainer_opt import BaseTrainer_opt

# =============================================================================
# MCDropoutNN_opt
# =============================================================================
"""
Red neuronal con Dropout para estimación de incertidumbre mediante muestreo Monte Carlo.

Parámetros:
    input_dim (int): Dimensión de entrada.
    hidden_dim (int): Dimensión de la capa oculta.
    dropout_prob (float): Probabilidad de dropout.
"""
class MCDropoutNN_opt(nn.Module):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    """
    Inicializa la arquitectura de la red con Dropout.
    """
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    """
    Propagación hacia adelante con Dropout.
    Args:
        x (Tensor): Entrada.
    Returns:
        Tensor: Salida de la red.
    """
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# =============================================================================
# MonteCarloDropoutNetwork_opt
# =============================================================================
"""
Entrenador para redes con Dropout Monte Carlo, hereda de BaseTrainer_opt.

Parámetros:
    input_dim (int): Dimensión de entrada.
    hidden_dim (int): Dimensión de la capa oculta.
    device (str): Dispositivo ('cpu' o 'cuda').
    dropout_prob (float): Probabilidad de dropout.
    show_info (str): Nivel de información a mostrar.
"""
class MonteCarloDropoutNetwork_opt(BaseTrainer_opt):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    """
    Inicializa el entrenador MonteCarloDropout.
    """
    def __init__(self, input_dim, hidden_dim, device, dropout_prob=0.2, show_info="True"):
        super().__init__(device, show_info)
        self.model = MCDropoutNN_opt(input_dim, hidden_dim, dropout_prob).to(device)
        self.dropout_prob = dropout_prob

    # -------------------------------------------------------------------------
    # train_model
    # -------------------------------------------------------------------------
    """
    Entrena la red con Dropout Monte Carlo y selecciona el mejor modelo según la pérdida de validación.
    Args:
        X, y: Datos de entrenamiento.
        X_val, y_val: Datos de validación.
        num_epochs (int): Número de épocas.
        lr (float): Tasa de aprendizaje.
    Returns:
        nn.Module: Modelo entrenado.
    """
    def train_model(self, X, y, X_val, y_val, num_epochs=1000, lr=0.001):
        net_name = "MonteCarloDropout"
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_loss = float('inf')
        best_model_state = None
        self._print("Entrenando red...", level="start", net_name=net_name)
        for epoch in range(num_epochs):
            self.model.train()
            outputs = self.model(X).squeeze()
            loss = criterion(outputs, y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Validación
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val.squeeze())
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                best_model_state = self.model.state_dict()
                self._print(f"Mejor modelo en época {epoch} 📉 train loss: {loss.item():.4f} 📊 val loss: {val_loss.item():.4f}", level="best", net_name=net_name)
            if epoch % 100 == 0:
                self._print(f"[Época {epoch}] 🧠 Pérdida entrenamiento: {loss.item():.4f} | Validación: {val_loss.item():.4f}", level="progress", net_name=net_name)
        self.model.load_state_dict(best_model_state)
        self._print("Entrenamiento finalizado", level="stage", net_name=net_name)
        return self.model

    # -------------------------------------------------------------------------
    # predict_mc_dropout
    # -------------------------------------------------------------------------
    """
    Realiza predicciones con muestreo Monte Carlo usando Dropout.
    Args:
        X: Datos de entrada.
        num_samples (int): Número de muestras MC.
    Returns:
        tuple: (media, desviación estándar) de las predicciones.
    """
    def predict_mc_dropout(self, X, num_samples=100):
        self.model.eval()
        self.model.dropout.train()
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(self.model(X).squeeze().cpu().numpy())
        preds = np.stack(preds)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std
