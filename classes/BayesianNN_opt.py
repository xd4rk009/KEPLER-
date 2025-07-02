import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
from .BaseTrainer_opt import BaseTrainer_opt

# =============================================================================
# BayesianNN_opt
# =============================================================================
"""
Entrenador para redes neuronales bayesianas usando Pyro y SVI, hereda de BaseTrainer_opt.

Parámetros:
    input_dim (int): Dimensión de entrada.
    hidden_dim (int): Dimensión de la capa oculta.
    device (str): Dispositivo ('cpu' o 'cuda').
    show_info (str): Nivel de información a mostrar.
"""
class BayesianNN_opt(BaseTrainer_opt):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    """
    Inicializa el entrenador bayesiano.
    """
    def __init__(self, input_dim, hidden_dim, device, show_info="True"):
        super().__init__(device, show_info)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.svi = None
        self._trained = False

    # -------------------------------------------------------------------------
    # model
    # -------------------------------------------------------------------------
    """
    Define el modelo bayesiano (prior y likelihood) para Pyro.
    Args:
        X: Datos de entrada.
        y: Etiquetas (opcional).
    """
    def model(self, X, y=None):
        prior = 0.05
        w1 = pyro.sample("w1", dist.Normal(
            torch.zeros(self.input_dim, self.hidden_dim, device=self.device),
            prior * torch.ones(self.input_dim, self.hidden_dim, device=self.device)
        ).to_event(2))
        b1 = pyro.sample("b1", dist.Normal(
            torch.zeros(self.hidden_dim, device=self.device),
            prior * torch.ones(self.hidden_dim, device=self.device)
        ).to_event(1))
        w2 = pyro.sample("w2", dist.Normal(
            torch.zeros(self.hidden_dim, 1, device=self.device),
            prior * torch.ones(self.hidden_dim, 1, device=self.device)
        ).to_event(2))
        b2 = pyro.sample("b2", dist.Normal(
            torch.zeros(1, device=self.device),
            prior * torch.ones(1, device=self.device)
        ).to_event(1))
        z1 = torch.relu(X @ w1 + b1.unsqueeze(0))
        output = z1 @ w2 + b2
        output = output.squeeze(-1)
        sigma = pyro.param("sigma", torch.ones(1, device=self.device), constraint=dist.constraints.positive)
        with pyro.plate("data", X.shape[0]):
            pyro.sample("obs", dist.Normal(output, sigma), obs=y.view(-1) if y is not None else None)

    # -------------------------------------------------------------------------
    # guide
    # -------------------------------------------------------------------------
    """
    Define la guía variacional para Pyro (aproximación posterior).
    Args:
        X: Datos de entrada.
        y: Etiquetas (opcional).
    """
    def guide(self, X, y=None):
        scale_init = 0.05
        w1_loc = pyro.param("w1_loc", 0.1 * torch.randn(self.input_dim, self.hidden_dim, device=self.device))
        w1_scale = pyro.param("w1_scale", scale_init * torch.ones(self.input_dim, self.hidden_dim, device=self.device), constraint=dist.constraints.positive)
        b1_loc = pyro.param("b1_loc", 0.1 * torch.randn(self.hidden_dim, device=self.device))
        b1_scale = pyro.param("b1_scale", scale_init * torch.ones(self.hidden_dim, device=self.device), constraint=dist.constraints.positive)
        w2_loc = pyro.param("w2_loc", 0.1 * torch.randn(self.hidden_dim, 1, device=self.device))
        w2_scale = pyro.param("w2_scale", scale_init * torch.ones(self.hidden_dim, 1, device=self.device), constraint=dist.constraints.positive)
        b2_loc = pyro.param("b2_loc", 0.1 * torch.randn(1, device=self.device))
        b2_scale = pyro.param("b2_scale", scale_init * torch.ones(1, device=self.device), constraint=dist.constraints.positive)
        pyro.sample("w1", dist.Normal(w1_loc, w1_scale).to_event(2))
        pyro.sample("b1", dist.Normal(b1_loc, b1_scale).to_event(1))
        pyro.sample("w2", dist.Normal(w2_loc, w2_scale).to_event(2))
        pyro.sample("b2", dist.Normal(b2_loc, b2_scale).to_event(1))

    # -------------------------------------------------------------------------
    # train_model
    # -------------------------------------------------------------------------
    """
    Entrena la red bayesiana usando SVI y muestra pérdidas de entrenamiento y validación.
    Args:
        X_train, y_train: Datos de entrenamiento.
        X_val, y_val: Datos de validación (opcional).
        num_steps (int): Número de pasos de optimización.
        lr (float): Tasa de aprendizaje.
    Returns:
        self
    """
    def train_model(self, X_train, y_train, X_val=None, y_val=None, num_steps=10000, lr=1e-3):
        net_name = "BayesianNN"
        pyro.clear_param_store()
        pyro.set_rng_seed(42)
        optimizer = Adam({"lr": lr})
        self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        self._print("Entrenando red...", level="start", net_name=net_name)
        for step in range(num_steps):
            train_loss = self.svi.step(X_train, y_train)
            val_loss = None
            if X_val is not None and y_val is not None:
                # Calcular val loss usando el modelo y los parámetros actuales
                guide_trace = pyro.poutine.trace(self.guide).get_trace(X_val)
                w1 = guide_trace.nodes["w1"]["value"]
                b1 = guide_trace.nodes["b1"]["value"]
                w2 = guide_trace.nodes["w2"]["value"]
                b2 = guide_trace.nodes["b2"]["value"]
                z1 = torch.relu(X_val @ w1 + b1.unsqueeze(0))
                output = z1 @ w2 + b2
                output = output.squeeze(-1)
                sigma = pyro.param("sigma")
                val_loss = -pyro.distributions.Normal(output, sigma).log_prob(y_val.view(-1)).mean().item()
            if step % 500 == 0:
                if val_loss is not None:
                    self._print(f"[Paso {step}] train loss: {train_loss:.4f} | val loss: {val_loss:.4f}", level="progress", net_name=net_name)
                else:
                    self._print(f"[Paso {step}] train loss: {train_loss:.4f}", level="progress", net_name=net_name)
        self._trained = True
        self._print("Entrenamiento finalizado", level="stage", net_name=net_name)
        return self

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    """
    Realiza predicciones bayesianas con muestreo posterior.
    Args:
        X: Datos de entrada.
        num_samples (int): Número de muestras del posterior.
    Returns:
        tuple: (media, desviación estándar) de las predicciones.
    """
    def predict(self, X, num_samples=100):
        if not self._trained:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir.")
        preds = []
        for _ in range(num_samples):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(X)
            w1 = guide_trace.nodes["w1"]["value"]
            b1 = guide_trace.nodes["b1"]["value"]
            w2 = guide_trace.nodes["w2"]["value"]
            b2 = guide_trace.nodes["b2"]["value"]
            z1 = torch.relu(X @ w1 + b1.unsqueeze(0))
            output = z1 @ w2 + b2
            preds.append(output.squeeze().cpu().detach().numpy())
        preds = np.stack(preds)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std
