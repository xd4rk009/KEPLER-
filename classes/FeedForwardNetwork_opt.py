import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseTrainer_opt import BaseTrainer_opt

# =============================================================================
# FeedForwardNN_opt
# =============================================================================
"""
Red neuronal feedforward simple con dos capas ocultas.

Parámetros:
    input_dim (int): Dimensión de entrada.
    hidden_dim (int): Dimensión de las capas ocultas.
"""
class FeedForwardNN_opt(nn.Module):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    """
    Inicializa la arquitectura de la red FeedForward.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    """
    Propagación hacia adelante de la red.
    Args:
        x (Tensor): Entrada.
    Returns:
        Tensor: Salida de la red.
    """
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =============================================================================
# FeedForwardNetwork_opt
# =============================================================================
"""
Entrenador para la red FeedForward, hereda de BaseTrainer_opt y maneja el ciclo de entrenamiento y predicción.

Parámetros:
    input_dim (int): Dimensión de entrada.
    hidden_dim (int): Dimensión de las capas ocultas.
    device (str): Dispositivo ('cpu' o 'cuda').
    show_info (str): Nivel de información a mostrar.
"""
class FeedForwardNetwork_opt(BaseTrainer_opt):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    """
    Inicializa el entrenador FeedForward.
    """
    def __init__(self, input_dim, hidden_dim, device, show_info="True"):
        super().__init__(device, show_info)
        self.model = FeedForwardNN_opt(input_dim, hidden_dim).to(device)

    # -------------------------------------------------------------------------
    # train_model
    # -------------------------------------------------------------------------
    """
    Entrena la red FeedForward y selecciona el mejor modelo según la pérdida de validación.
    Args:
        X, y: Datos de entrenamiento.
        X_val, y_val: Datos de validación.
        num_epochs (int): Número de épocas.
        lr (float): Tasa de aprendizaje.
    Returns:
        nn.Module: Modelo entrenado.
    """
    def train_model(self, X, y, X_val, y_val, num_epochs=1000, lr=0.001):
        net_name = "FeedForward"
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
    # predict
    # -------------------------------------------------------------------------
    """
    Realiza predicciones usando el modelo entrenado.
    Args:
        X: Datos de entrada.
    Returns:
        Tensor: Predicciones.
    """
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).squeeze()
