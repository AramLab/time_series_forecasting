import numpy as np
import pandas as pd
import importlib


def safe_import_ceemdan():
    """Безопасный импорт CEEMDAN с fallback на чистую Python реализацию
    
    В Google Colab: использует EMD-signal (быстро, C extension)
    На Mac/Linux: использует SimpleCEEMDAN (pure Python, всегда работает)
    """
    # Google Colab и Linux: EMD-signal / PyEMD
    try:
        from PyEMD import CEEMDAN as CEEMDAN_Class
        print("✅ Используется PyEMD/EMD-signal (C extension - быстро)")
        return CEEMDAN_Class, "PyEMD"
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Альтернатива: EMD из пакета EMD-signal
    try:
        from EMD import CEEMDAN as CEEMDAN_Class
        print("✅ Используется EMD (C extension - быстро)")
        return CEEMDAN_Class, "EMD"
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Fallback: ВСЕГДА работает на любой ОС (Mac, Linux, Windows)
    try:
        from utils.ceemdan_pure_python import SimpleCEEMDAN
        print("⚠️  Используется SimpleCEEMDAN (pure Python - стабильнее, медленнее)")
        return SimpleCEEMDAN, "SimpleCEEMDAN"
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return None, None


class CEEMDANEnsembleModel:
    def __init__(self, base_model_class, trials=20, noise_width=0.05):
        """
        Initialize CEEMDAN ensemble model with a base model class
        """
        self.base_model_class = base_model_class
        self.trials = trials
        self.noise_width = noise_width
        self.ceemdan_instance = None
        self.base_models = []
        self.is_fitted = False
    
    def fit(self, data):
        """
        Fit CEEMDAN ensemble model on the provided data
        """
        # Get CEEMDAN class
        CEEMDAN_Class, source = safe_import_ceemdan()
        if CEEMDAN_Class is None:
            raise ValueError("CEEMDAN is not available")
        
        # Create CEEMDAN instance
        self.ceemdan_instance = CEEMDAN_Class(trials=self.trials, noise_width=self.noise_width)
        
        # Decompose the signal
        imfs = self.ceemdan_instance(data.astype(float))
        
        # Fit base model for each IMF
        self.base_models = []
        for i, imf in enumerate(imfs):
            if not np.all(np.isfinite(imf)):
                print(f"⚠️ IMF {i + 1} contains non-finite values. Skipping.")
                continue
            
            # Create and fit base model for this IMF
            base_model = self.base_model_class()
            base_model.fit(imf)
            self.base_models.append(base_model)
        
        self.is_fitted = True
        return self

    def predict(self, steps):
        """
        Forecast future values using the trained CEEMDAN ensemble model
        """
        if not self.is_fitted or not self.base_models:
            raise ValueError("Model must be fitted before making predictions")
        
        # Forecast each IMF component
        imf_forecasts = []
        for i, base_model in enumerate(self.base_models):
            try:
                imf_forecast = base_model.predict(steps)
                imf_forecasts.append(imf_forecast)
            except Exception as e:
                print(f"⚠️ Error forecasting IMF {i + 1}: {str(e)}")
                # Use naive forecast as fallback
                imf_forecasts.append(np.full(steps, np.mean(base_model.data if hasattr(base_model, 'data') else [0])))
        
        if not imf_forecasts:
            raise ValueError("Could not generate forecasts for any IMF components")
        
        # Combine forecasts from all IMFs
        min_length = min(len(forecast) for forecast in imf_forecasts)
        combined_forecast = np.sum([forecast[:min_length] for forecast in imf_forecasts], axis=0)
        
        # Adjust length to match required steps
        if len(combined_forecast) < steps:
            # Pad with last value
            last_value = combined_forecast[-1] if len(combined_forecast) > 0 else 0
            padding = np.full(steps - len(combined_forecast), last_value)
            combined_forecast = np.concatenate([combined_forecast, padding])
        elif len(combined_forecast) > steps:
            combined_forecast = combined_forecast[:steps]
        
        return combined_forecast