"""
Base model class for all ML models
"""

from abc import ABC, abstractmethod
import joblib
from pathlib import Path
from typing import Any, Optional


class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name: str = "base_model"):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def build(self, **kwargs):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath: str):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        
    def get_params(self):
        """Get model parameters"""
        if self.model is not None and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params):
        """Set model parameters"""
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
