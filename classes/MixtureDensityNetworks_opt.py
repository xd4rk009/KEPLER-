import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .BaseTrainer_opt import BaseTrainer_opt

# =============================================================================
# MixtureDensityModel_opt
# =============================================================================
"""
Red neuronal para Mixture Density Networks (MDN), produce parámetros de mezcla gaussiana.

Parámetros:
    input_dim (int): Dimensión de entrada.
    hidden_dim (int): Dimensión de las capas ocultas.
    n_components (int): Número de componentes de la mezcla.
"""
class MixtureDensityModel_opt(nn.Module):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    """
    Inicializa la arquitectura MDN.
    """
    def __init__(self, input_dim, hidden_dim, n_components):
        super().__init__()
        self.n_components = n_components
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 3 * n_components)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    """
    Propagación hacia adelante de la MDN.
    Args:
        x (Tensor): Entrada.
    Returns:
        tuple: (pi, mu, sigma) parámetros de la mezcla.
    """
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        params = self.fc4(x)
        pi = F.softmax(params[:, :self.n_components], dim=1)
        mu = params[:, self.n_components:2 * self.n_components]
        sigma = F.softplus(params[:, 2 * self.n_components:])
        return pi, mu, sigma

# =============================================================================
# MixtureDensityNetworks_opt
# =============================================================================
"""
Entrenador para Mixture Density Networks, hereda de BaseTrainer_opt.

Parámetros:
    input_dim (int): Dimensión de entrada.
    hidden_dim (int): Dimensión de las capas ocultas.
    n_components (int): Número de componentes de la mezcla.
    device (str): Dispositivo ('cpu' o 'cuda').
    show_info (str): Nivel de información a mostrar.
"""
class MixtureDensityNetworks_opt(BaseTrainer_opt):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    """
    Inicializa el entrenador MDN.
    """
    def __init__(self, input_dim, hidden_dim, n_components, device, show_info="True"):
        super().__init__(device, show_info)
        self.model = MixtureDensityModel_opt(input_dim, hidden_dim, n_components).to(device)
        self.n_components = n_components

    # -------------------------------------------------------------------------
    # mdn_loss
    # -------------------------------------------------------------------------
    """
    Calcula la pérdida negativa log-verosimilitud para MDN.
    Args:
        pi, mu, sigma: Parámetros de la mezcla.
        y: Valores reales.
    Returns:
        Tensor: Pérdida promedio.
    """
    def mdn_loss(self, pi, mu, sigma, y):
        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        y_expanded = y.expand(-1, mu.size(1))
        log_prob = normal.log_prob(y_expanded)
        weighted_log_prob = log_prob + torch.log(pi)
        return -torch.logsumexp(weighted_log_prob, dim=1).mean()

    # -------------------------------------------------------------------------
    # train_model
    # -------------------------------------------------------------------------
    """
    Entrena la red MDN y selecciona el mejor modelo según la pérdida de validación.
    Args:
        X_train, y_train: Datos de entrenamiento.
        X_val, y_val: Datos de validación.
        num_epochs (int): Número de épocas.
        lr (float): Tasa de aprendizaje.
    Returns:
        nn.Module: Modelo entrenado.
    """
    def train_model(self, X_train, y_train, X_val, y_val, num_epochs=5000, lr=0.001):
        net_name = "MixtureDensityNetworks"
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_loss = float('inf')
        best_model_state = None
        self._print("Entrenando red...", level="start", net_name=net_name)
        for epoch in range(num_epochs):
            self.model.train()
            pi, mu, sigma = self.model(X_train)
            loss = self.mdn_loss(pi, mu, sigma, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                pi_val, mu_val, sigma_val = self.model(X_val)
                val_loss = self.mdn_loss(pi_val, mu_val, sigma_val, y_val)
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
    Realiza predicciones usando el modelo MDN entrenado.
    Args:
        X: Datos de entrada.
    Returns:
        tuple: (pi, mu, sigma) parámetros de la mezcla.
    """
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            pi, mu, sigma = self.model(X)
        return pi, mu, sigma

    # -------------------------------------------------------------------------
    # calcular_estadisticas
    # -------------------------------------------------------------------------
    """
    Calcula media, varianza y desviación estándar esperadas de la mezcla.
    Args:
        pi, mu, sigma: Parámetros de la mezcla.
    Returns:
        tuple: (media_esperada, var_esperada, std_esperada)
    """
    def calcular_estadisticas(self, pi, mu, sigma):
        media_esperada = (pi * mu).sum(dim=1, keepdim=True)
        var_esperada = (pi * (sigma**2 + mu**2)).sum(dim=1, keepdim=True) - media_esperada**2
        std_esperada = torch.sqrt(var_esperada)
        return media_esperada, var_esperada, std_esperada
