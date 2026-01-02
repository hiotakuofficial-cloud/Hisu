"""
Neural network model implementations using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple
from .base_model import BaseModel


class NeuralNetwork(BaseModel):
    """Neural network model using PyTorch"""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [64, 32],
        output_size: int = 1,
        task: str = 'regression',
        model_name: str = "neural_network"
    ):
        super().__init__(model_name)
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build(self, dropout: float = 0.2, activation: str = 'relu'):
        """Build neural network architecture"""
        layers = []
        
        prev_size = self.input_size
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.output_size))
        
        if self.task == 'classification' and self.output_size > 1:
            layers.append(nn.Softmax(dim=1))
        elif self.task == 'binary_classification':
            layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers).to(self.device)
        
        return self
    
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer_type: str = 'adam',
        verbose: bool = True
    ):
        """Train the neural network"""
        if self.model is None:
            self.build()
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if len(y_train_tensor.shape) == 1:
            y_train_tensor = y_train_tensor.unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if self.task == 'regression':
            criterion = nn.MSELoss()
        elif self.task == 'binary_classification':
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            self.training_history['train_loss'].append(avg_train_loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val, criterion)
                self.training_history['val_loss'].append(val_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def _validate(self, X_val, y_val, criterion):
        """Validate the model"""
        self.model.eval()
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        if len(y_val_tensor.shape) == 1:
            y_val_tensor = y_val_tensor.unsqueeze(1)
        
        with torch.no_grad():
            outputs = self.model(X_val_tensor)
            loss = criterion(outputs, y_val_tensor)
        
        return loss.item()
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'task': self.task
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        checkpoint = torch.load(filepath)
        self.input_size = checkpoint['input_size']
        self.hidden_layers = checkpoint['hidden_layers']
        self.output_size = checkpoint['output_size']
        self.task = checkpoint['task']
        
        self.build()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
