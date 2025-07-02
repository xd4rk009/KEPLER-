import torch
import datetime
import os
import numpy as np

class BaseTrainer_opt:
    """
    Clase base para modelos de entrenamiento. Incluye método run_pipeline para automatizar el flujo completo.
    """
    def __init__(self, device, show_info="True"):
        self.device = device
        self.show_info = show_info

    def _print(self, msg, level="info", net_name=None):
        if self.show_info == "False":
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        net = f"[{net_name}] " if net_name else ""
        if level == "start":
            prefix = f"[{now}] 🚀 {net}"
        elif level == "best":
            prefix = f"[{now}] ✨ {net}"
        elif level == "progress":
            prefix = f"[{now}] 🔄 {net}"
        elif level == "stage":
            prefix = f"[{now}] ✅ {net}"
        else:
            prefix = f"[{now}] {net}"
        if self.show_info == "Ligth":
            if level in ("start", "stage"):
                print(f"{prefix}{msg}")
        else:
            print(f"{prefix}{msg}")

    def to_device(self, *tensors):
        return [t.to(self.device) for t in tensors]

    def train_model(self, *args, **kwargs):
        raise NotImplementedError("Debe implementar train_model en la subclase.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Debe implementar predict en la subclase.")

    def save_model(self, model_name, ext='pt'):
        """
        Guarda el modelo en disco. Soporta modelos PyTorch y Pyro en formato homogéneo.
        """
        import pyro
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('nets', exist_ok=True)
        if model_name == 'TabNet' and ext == 'zip':
            model_path = f'nets/modelo_{model_name}_{now}'
        else:
            model_path = f'nets/modelo_{model_name}_{now}.{ext}'
        net = getattr(self, 'model', self)
        # --- Guardado homogéneo ---
        if model_name == 'BayesianNN':
            to_save = {'pyro_param_store': pyro.get_param_store().get_state()}
            torch.save(to_save, model_path)
        elif ext == 'pt':
            to_save = {'state_dict': net.state_dict()}
            torch.save(to_save, model_path)
        elif ext == 'zip':
            try:
                net.save_model(model_path)
            except Exception as e:
                print(f'No se pudo guardar el modelo {model_name}: {e}')
        else:
            raise ValueError('Extensión no soportada para el guardado del modelo')
        print(f'Modelo {model_name} guardado en: {model_path}')

    def load_model(self, path, model_name=None):
        """
        Carga el modelo desde disco. Soporta modelos PyTorch y Pyro en formato homogéneo.
        """
        import pyro
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        net = getattr(self, 'model', self)
        if model_name == 'BayesianNN':
            if 'pyro_param_store' in checkpoint:
                pyro.get_param_store().set_state(checkpoint['pyro_param_store'])
            else:
                print(f"Archivo {path} no contiene 'pyro_param_store'.")
        elif 'state_dict' in checkpoint and hasattr(net, 'load_state_dict'):
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
        else:
            print(f"Advertencia: no se pudo cargar el modelo para {model_name} desde {path}")

    def predict_and_denorm(self, X, y, scales, dh, model_name, std=False):
        with torch.no_grad():
            if model_name == 'FeedForward':
                y_pred = self.predict(X).cpu().numpy()
            elif model_name == 'MDN':
                try:
                    if hasattr(self, 'predict') and hasattr(self, 'calcular_estadisticas'):
                        pi, mu, sigma = self.predict(X)
                        media, _, std_esperada = self.calcular_estadisticas(pi, mu, sigma)
                        y_pred = media.cpu().numpy().squeeze()
                        if std:
                            std_dev = std_esperada.cpu().numpy().squeeze()
                    else:
                        mdn_out = self(X)
                        pi = mdn_out[0].cpu().numpy()
                        mu = mdn_out[1].cpu().numpy()
                        sigma = mdn_out[2].cpu().numpy()
                        y_pred = (pi * mu).sum(axis=1)
                        if std:
                            var = (pi * (sigma**2 + mu**2)).sum(axis=1) - y_pred**2
                            std_dev = np.sqrt(np.maximum(var, 0))
                except Exception as e:
                    print('Error en predict_and_denorm para MDN:', e)
                    raise
            elif model_name == 'TabNet':
                y_pred = self.model.predict(X.cpu().numpy()).squeeze()
            elif model_name == 'BayesianNN':
                mean, std_dev = self.predict(X, num_samples=100)
                y_pred = mean if isinstance(mean, np.ndarray) else mean.cpu().numpy()
                std_dev = std_dev if isinstance(std_dev, np.ndarray) else std_dev.cpu().numpy()
            elif model_name == 'MonteCarloDropout':
                mean, std_dev = self.predict_mc_dropout(X, num_samples=100)
                y_pred = mean
            else:
                raise ValueError('Modelo no soportado para predicción')
        y_np = y.cpu().numpy()
        y_mo = dh.denormalize_value(scales, y_np, 'Mo_cumulative')
        y_pred_mo = dh.denormalize_value(scales, y_pred, 'Mo_cumulative')
        if std:
            if model_name == 'MDN':
                std_mo = dh.denormalize_value(scales, std_dev, 'Mo_cumulative')
                return y_mo, y_pred_mo, std_mo
            else:
                return y_mo, y_pred_mo, std_dev
        else:
            return y_mo, y_pred_mo

    def plot_and_evaluate(self, dh, y_train_mo, y_pred_train_mo, y_val_mo, y_pred_val_mo, model_name, std_train=None, std_val=None):
        if std_train is not None and std_val is not None and model_name in ['BNN', 'MC Dropout', 'BayesianNN', 'MonteCarloDropout', 'MDN']:
            dh.plot_with_confidence_interval(
                (y_train_mo, 'Valores Reales'),
                (y_pred_train_mo, y_pred_train_mo - 1.96 * std_train, y_pred_train_mo + 1.96 * std_train, f'{model_name} - Predicciones'),
                title=f'{model_name}: Predicciones con Intervalo (Entrenamiento)', size=(1200, 400))
            dh.plot_with_confidence_interval(
                (y_val_mo, 'Valores Reales'),
                (y_pred_val_mo, y_pred_val_mo - 1.96 * std_val, y_pred_val_mo + 1.96 * std_val, f'{model_name} - Predicciones'),
                title=f'{model_name}: Predicciones con Intervalo (Validación)', size=(1200, 400))
        else:
            dh.plot_results((y_train_mo, 'Valores Reales'), (y_pred_train_mo, 'Predicciones'), title=f'Predicciones {model_name} vs Reales (Entrenamiento)', size=(1200, 400))
            dh.plot_results((y_val_mo, 'Valores Reales'), (y_pred_val_mo, 'Predicciones'), title=f'Predicciones {model_name} vs Reales (Validación)', size=(1200, 400))
        dh.evaluar_metricas(y_train_mo, y_pred_train_mo, f'{model_name} - Entrenamiento')
        dh.evaluar_metricas(y_val_mo, y_pred_val_mo, f'{model_name} - Validación')
        dh.plot_regression(y_train_mo, y_pred_train_mo, title=f'Entrenamiento {model_name}', size=(1200, 400))
        dh.plot_regression(y_val_mo, y_pred_val_mo, title=f'Validación {model_name}', size=(1200, 400))
        if std_train is not None and std_val is not None:
            print(f'Desviación estándar (Entrenamiento): {std_train.mean():.4f}')
            print(f'Desviación estándar (Validación): {std_val.mean():.4f}')

    @staticmethod
    def run_pipeline(
        model_type,
        X_train, y_train, X_val, y_val,
        scales, dh,
        hidden_dim=None,
        n_components=None,
        device='cpu',
        lr=0.001,
        num_epochs=100,
        save_model=True,
        plot_results=True,
        dropout_prob=0.2,
        ext=None
    ):
        input_dim = X_train.shape[1]
        model = None
        result = None
        if model_type == 'FeedForward':
            from classes.FeedForwardNetwork_opt import FeedForwardNetwork_opt
            model = FeedForwardNetwork_opt(input_dim, hidden_dim, device)
            model.train_model(X_train, y_train, X_val, y_val, num_epochs=num_epochs, lr=lr)
            if save_model:
                model.save_model('FeedForward')
            y_train_mo, y_pred_train_mo = model.predict_and_denorm(X_train, y_train, scales, dh, 'FeedForward')
            y_val_mo, y_pred_val_mo = model.predict_and_denorm(X_val, y_val, scales, dh, 'FeedForward')
            if plot_results:
                model.plot_and_evaluate(dh, y_train_mo, y_pred_train_mo, y_val_mo, y_pred_val_mo, 'FeedForward')
            result = (y_train_mo, y_pred_train_mo, y_val_mo, y_pred_val_mo)
        elif model_type == 'TabNet':
            from classes.TabNet_opt import TabNetNetwork_opt
            model = TabNetNetwork_opt(input_dim, device)
            model.train_model(X_train.cpu().numpy(), y_train.cpu().numpy(), X_val.cpu().numpy(), y_val.cpu().numpy(), max_epochs=num_epochs, patience=100)
            if save_model:
                model.save_model('TabNet', ext='zip')
            y_train_mo, y_pred_train_mo = model.predict_and_denorm(X_train, y_train, scales, dh, 'TabNet')
            y_val_mo, y_pred_val_mo = model.predict_and_denorm(X_val, y_val, scales, dh, 'TabNet')
            if plot_results:
                model.plot_and_evaluate(dh, y_train_mo, y_pred_train_mo, y_val_mo, y_pred_val_mo, 'TabNet')
            result = (y_train_mo, y_pred_train_mo, y_val_mo, y_pred_val_mo)
        elif model_type == 'MDN':
            from classes.MixtureDensityNetworks_opt import MixtureDensityNetworks_opt
            if n_components is None:
                n_components = 5
            model = MixtureDensityNetworks_opt(input_dim, hidden_dim, n_components, device)
            model.train_model(X_train, y_train, X_val, y_val, num_epochs=num_epochs, lr=lr)
            if save_model:
                model.save_model('MDN')
            y_train_mo, y_pred_train_mo, std_train = model.predict_and_denorm(X_train, y_train, scales, dh, 'MDN', std=True)
            y_val_mo, y_pred_val_mo, std_val = model.predict_and_denorm(X_val, y_val, scales, dh, 'MDN', std=True)
            if plot_results:
                model.plot_and_evaluate(dh, y_train_mo, y_pred_train_mo, y_val_mo, y_pred_val_mo, 'MDN', std_train=std_train, std_val=std_val)
            result = (y_train_mo, y_pred_train_mo, y_val_mo, y_pred_val_mo, std_train, std_val)
        elif model_type == 'BayesianNN':
            from classes.BayesianNN_opt import BayesianNN_opt
            model = BayesianNN_opt(input_dim, hidden_dim, device)
            model.train_model(X_train, y_train, X_val, y_val, num_steps=num_epochs, lr=lr)
            if save_model:
                model.save_model('BayesianNN', ext='pt')
            y_train_mo, mean_train_mo, std_train = model.predict_and_denorm(X_train, y_train, scales, dh, 'BayesianNN', std=True)
            y_val_mo, mean_val_mo, std_val = model.predict_and_denorm(X_val, y_val, scales, dh, 'BayesianNN', std=True)
            if plot_results:
                model.plot_and_evaluate(dh, y_train_mo, mean_train_mo, y_val_mo, mean_val_mo, 'BNN', std_train=std_train, std_val=std_val)
            result = (y_train_mo, mean_train_mo, y_val_mo, mean_val_mo, std_train, std_val)
        elif model_type == 'MonteCarloDropout':
            from classes.MonteCarloDropoutNetwork_opt import MonteCarloDropoutNetwork_opt
            model = MonteCarloDropoutNetwork_opt(input_dim, hidden_dim, device, dropout_prob=dropout_prob)
            model.train_model(X_train, y_train, X_val, y_val, num_epochs=num_epochs, lr=lr)
            if save_model:
                model.save_model('MonteCarloDropout')
            y_train_mo, mean_train_mo, std_train = model.predict_and_denorm(X_train, y_train, scales, dh, 'MonteCarloDropout', std=True)
            y_val_mo, mean_val_mo, std_val = model.predict_and_denorm(X_val, y_val, scales, dh, 'MonteCarloDropout', std=True)
            if plot_results:
                model.plot_and_evaluate(dh, y_train_mo, mean_train_mo, y_val_mo, mean_val_mo, 'MC Dropout', std_train=std_train, std_val=std_val)
            result = (y_train_mo, mean_train_mo, y_val_mo, mean_val_mo, std_train, std_val)
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")
        return result
