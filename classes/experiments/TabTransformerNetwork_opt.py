# TabTransformerNetwork_opt.py
# Implementación "raw" de TabTransformer para regresión tabular, compatible con BaseTrainer_opt
# Autor: GitHub Copilot
# Fecha: 2025-05-18

import torch
import torch.nn as nn
import torch.nn.functional as F
from classes.BaseTrainer_opt import BaseTrainer_opt

class TabTransformerNetwork_opt(BaseTrainer_opt):
    """
    TabTransformer para regresión tabular, hereda de BaseTrainer_opt.
    Soporta entrenamiento en GPU, guardado, predicción y pipeline modular.
    """
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, device, n_continuous=0, n_categories=0, category_dims=None, show_info="True", dropout=0.2):
        super().__init__(device, show_info)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_continuous = n_continuous
        self.n_categories = n_categories
        self.category_dims = category_dims
        self.dropout = dropout
        # Embedding para variables categóricas
        if n_categories > 0 and category_dims is not None:
            self.cat_embeds = nn.ModuleList([
                nn.Embedding(cat_dim, hidden_dim) for cat_dim in category_dims
            ])
        else:
            self.cat_embeds = None
        # Proyección para variables continuas
        if n_continuous > 0:
            self.cont_proj = nn.Linear(n_continuous, hidden_dim)
        else:
            self.cont_proj = None
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Capa final de regresión con Dropout configurable
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, 1)
        )
        # Mover manualmente cada submódulo relevante al dispositivo
        if self.cat_embeds is not None:
            for emb in self.cat_embeds:
                emb.to(device)
        if self.cont_proj is not None:
            self.cont_proj.to(device)
        self.transformer.to(device)
        self.regressor.to(device)

    def forward(self, x_cont, x_cat=None):
        # x_cont: [batch, n_continuous], x_cat: [batch, n_categories]
        tokens = []
        if self.cat_embeds is not None and x_cat is not None:
            for i, emb in enumerate(self.cat_embeds):
                tokens.append(emb(x_cat[:, i]))  # [batch, hidden_dim]
        if self.cont_proj is not None and x_cont is not None:
            tokens.append(self.cont_proj(x_cont))
        # Stack tokens: [batch, n_tokens, hidden_dim]
        x = torch.stack(tokens, dim=1)
        x = self.transformer(x)
        # Pooling (media)
        x = x.mean(dim=1)
        out = self.regressor(x)
        return out.squeeze(-1)

    def train_model(self, X_train, y_train, X_val, y_val, num_epochs=100, lr=1e-3):
        # No existe self.train() ni self.eval(), pero sí para submódulos
        for module in [self.cat_embeds, self.cont_proj, self.transformer, self.regressor]:
            if module is not None and hasattr(module, 'train'):
                module.train()
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            x_cont, x_cat = X_train
            y_pred = self.forward(x_cont, x_cat)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            if self.show_info == "True" and (epoch % 100 == 0 or epoch == num_epochs - 1):
                self._print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}", level="progress", net_name="TabTransformer")
        for module in [self.cat_embeds, self.cont_proj, self.transformer, self.regressor]:
            if module is not None and hasattr(module, 'eval'):
                module.eval()

    def predict(self, X, *args, **kwargs):
        # No existe self.eval(), pero sí para submódulos
        for module in [self.cat_embeds, self.cont_proj, self.transformer, self.regressor]:
            if module is not None and hasattr(module, 'eval'):
                module.eval()
        x_cont, x_cat = X
        with torch.no_grad():
            y_pred = self.forward(x_cont, x_cat)
        return y_pred.cpu().numpy()

    def save_model(self, model_name="TabTransformer", ext="pt"):
        super().save_model(model_name, ext=ext)

    # Métodos de denormalización, graficado y pipeline ya están en BaseTrainer_opt
