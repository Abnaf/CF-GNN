"""
CF-GNN: Class Fourier Graph Neural Network for 5G Core Failure Localization.
Author: Abubakar Isah

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tkinter import Tk, filedialog
import logging

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for high-dimensional feature projection.
    
    Args:
        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        output_dim (int): Number of target classes.
        num_layers (int): Total linear layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class GNFE(nn.Module):
    """
    Graph Node Feature Extraction (GNFE) Layer.
    Performs spectral-style aggregation: X' = \sigma(AXW).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(GNFE, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(adj, x)
        return F.relu(self.lin(x))

class MASL(nn.Module):
    """
    Multi-head Attention Spectral (MAS) Layer.
    Enables multi-path signal propagation across the network topology.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        super(MAS, self).__init__()
        self.heads = heads
        self.att_weights = nn.Parameter(torch.Tensor(heads, in_channels, out_channels))
        nn.init.xavier_uniform_(self.att_weights)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.heads):
            # Spectral projection per head
            h = torch.matmul(x, self.att_weights[i])
            outputs.append(torch.matmul(adj, h))
        return F.elu(torch.mean(torch.stack(outputs), dim=0))

class CF_GNN(nn.Module):
    """
    Class Fourier Graph Neural Network (CF-GNN).
    Integrates Graph Fourier Transform (GFT) concepts with gated spectral attention.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(CF_GNN, self).__init__()
        self.feature_extractor = GNFE(input_dim, hidden_dim)
        
        # Spectral Transformation Components
        self.eigenvalue_transform = nn.Linear(hidden_dim, hidden_dim)
        self.eigenvector_attention = nn.Linear(hidden_dim, hidden_dim)
        
        self.att_layers = nn.ModuleList([
            MAS(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.classifier = MLP(hidden_dim, hidden_dim, output_dim, num_layers=1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing spectral gating to mitigate over-smoothing.
        """
        x = self.feature_extractor(x, adj)
        
        # Gated Spectral Transform
        x_eigen = torch.relu(self.eigenvalue_transform(x))
        attn_weights = torch.sigmoid(self.eigenvector_attention(x_eigen))
        x = x_eigen * attn_weights
        
        for layer in self.att_layers:
            x = layer(x, adj)
        
        return self.classifier(x)

# --- Data Utilities ---

def load_data_from_gui():
    """Handles file selection and preprocessing."""
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(title="Select 5G Telemetry CSV")
    root.destroy()
    
    if not path:
        return None, None

    df = pd.read_csv(path)
    df.drop(['time', 'source_name'], axis=1, errors='ignore', inplace=True)
    
    X = df.drop('y_true(fc)', axis=1).values
    y = LabelEncoder().fit_transform(df['y_true(fc)'].values)
    X = MinMaxScaler().fit_transform(X)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    X, y = load_data_from_gui()
    
    if X is not None:
        # Model Configuration
        cfg = {
            "in": X.shape[1],
            "hid": 64,
            "out": len(torch.unique(y)),
            "nodes": X.shape[0]
        }

        # Mock Adjacency (Replace with your Knowledge Graph structure)
        adj = torch.eye(cfg["nodes"]) + torch.rand(cfg["nodes"], cfg["nodes"]) * 0.05
        
        model = CF_GNN(cfg["in"], cfg["hid"], cfg["out"])
        out = model(X, adj)
        
        logging.info(f"Model Forward Pass Successful. Output Dim: {out.shape}")
