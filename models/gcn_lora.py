"""
FedGNN_advanced/models/gcn_lora.py

A small GCN with LoRA applied only to the final linear classifier.
Production-ready implementation with comprehensive error handling and validation.
"""

import math
import logging
from typing import Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("GCNLoRA")
logger.setLevel("INFO")

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("PyTorch Geometric not available, using fallback implementation")

class LoRAError(Exception):
    """Base exception for LoRA-related errors"""
    pass

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    Implements the LoRA technique from https://arxiv.org/abs/2106.09685
    for parameter-efficient fine-tuning.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 1.0,
                 dropout: float = 0.0, bias: bool = True):
        """
        Initialize LoRALinear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            rank: Rank of the low-rank adaptation
            alpha: Scaling factor for LoRA weights
            dropout: Dropout probability for LoRA layers
            bias: Whether to include bias term
        """
        super().__init__()

        # Validate inputs
        if in_features <= 0 or out_features <= 0:
            raise LoRAError("in_features and out_features must be positive")

        if rank <= 0:
            raise LoRAError("rank must be positive")

        if alpha <= 0:
            raise LoRAError("alpha must be positive")

        if not 0 <= dropout < 1:
            raise LoRAError("dropout must be in [0, 1)")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapters
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA weights
        self._init_lora_weights()

        # Freeze original weights if requested
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        logger.debug(f"Initialized LoRALinear: in={in_features}, out={out_features}, "
                    f"rank={rank}, alpha={alpha}, dropout={dropout}")

    def _init_lora_weights(self):
        """Initialize LoRA weights with appropriate scaling"""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # Initialize B with zeros
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Original linear transformation
        base_output = self.linear(x)

        # LoRA adaptation
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        scaled_lora = lora_output * self.scaling

        return base_output + scaled_lora

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into the base linear layer.
        This is useful for inference to reduce computation.
        """
        with torch.no_grad():
            # Merge LoRA weights into base weights
            merged_weights = self.linear.weight.data + self.scaling * (
                self.lora_B.weight @ self.lora_A.weight
            )
            self.linear.weight.data = merged_weights

            # Reset LoRA weights
            self._init_lora_weights()

        logger.info("LoRA weights merged into base layer")

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, alpha={self.alpha}, bias={self.linear.bias is not None}')

class GCNWithLoRA(nn.Module):
    """
    GCN model with LoRA-adapted classifier for federated learning.

    This model uses GCN layers for feature extraction and a LoRA-adapted
    linear layer for classification, enabling efficient parameter updates
    in federated learning scenarios.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 lora_rank: int = 8, lora_alpha: float = 1.0,
                 num_layers: int = 2, dropout: float = 0.5,
                 use_pyg: bool = PYG_AVAILABLE):
        """
        Initialize GCN with LoRA.

        Args:
            in_dim: Number of input features
            hidden_dim: Number of hidden features
            out_dim: Number of output classes
            lora_rank: Rank for LoRA adaptation
            lora_alpha: Scaling factor for LoRA
            num_layers: Number of GCN layers
            dropout: Dropout probability
            use_pyg: Whether to use PyTorch Geometric layers
        """
        super().__init__()

        # Validate inputs
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("Input dimensions must be positive")

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")

        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_pyg = use_pyg and PYG_AVAILABLE

        # Create GCN layers
        self.convs = nn.ModuleList()

        if self.use_pyg:
            # PyG implementation
            for i in range(num_layers):
                in_channels = in_dim if i == 0 else hidden_dim
                out_channels = hidden_dim if i < num_layers - 1 else hidden_dim
                self.convs.append(GCNConv(in_channels, out_channels))
        else:
            # Fallback MLP implementation
            for i in range(num_layers):
                in_channels = in_dim if i == 0 else hidden_dim
                out_channels = hidden_dim if i < num_layers - 1 else hidden_dim
                self.convs.append(nn.Linear(in_channels, out_channels))

        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # LoRA-adapted classifier
        self.classifier = LoRALinear(hidden_dim, out_dim,
                                   rank=lora_rank, alpha=lora_alpha)

        logger.info(f"Initialized GCNWithLoRA: in_dim={in_dim}, hidden_dim={hidden_dim}, "
                   f"out_dim={out_dim}, layers={num_layers}, lora_rank={lora_rank}, "
                   f"use_pyg={self.use_pyg}")

    def forward(self, data: Union[torch.Tensor, Any]) -> torch.Tensor:
        """
        Forward pass for GCN with LoRA.

        Args:
            data: Input data. Can be:
                - PyG Data object (if use_pyg=True)
                - Tuple of (features, edge_index, batch)
                - Just features tensor (if use_pyg=False)

        Returns:
            Logits tensor of shape (batch_size, out_dim)
        """
        # Extract features and optional graph structure
        if self.use_pyg:
            if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                # PyG Data object
                x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                # Tuple (features, edge_index, [batch])
                x = data[0]
                edge_index = data[1]
                batch = data[2] if len(data) > 2 else None
            else:
                raise ValueError("Invalid input format for PyG mode")
        else:
            # Fallback MLP mode - expect just features
            if isinstance(data, (list, tuple)):
                x = data[0]
            else:
                x = data
            edge_index = None
            batch = None

        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            if self.use_pyg:
                x = conv(x, edge_index)
            else:
                x = conv(x)

            if i < len(self.convs) - 1:
                x = self.bns[i](x) if i < len(self.bns) else x
                x = F.relu(x)
                x = self.dropout_layer(x)

        # Pooling (if batch information is available)
        if batch is not None and self.use_pyg:
            x = global_mean_pool(x, batch)
        elif x.dim() > 2:
            # Global average pooling for non-graph data
            x = x.mean(dim=1)

        # Apply classifier
        x = self.classifier(x)

        return x

    def get_lora_parameters(self) -> list:
        """
        Get parameters that should be updated during LoRA fine-tuning.

        Returns:
            List of parameters that require gradients
        """
        return list(self.classifier.parameters())

    def get_base_parameters(self) -> list:
        """
        Get base model parameters (frozen during LoRA fine-tuning).

        Returns:
            List of base parameters
        """
        base_params = []
        for name, param in self.named_parameters():
            if 'lora' not in name and 'classifier' not in name:
                base_params.append(param)
        return base_params

    def merge_lora_weights(self) -> None:
        """
        Merge LoRA weights into the base classifier for inference.
        """
        self.classifier.merge_weights()

    def extra_repr(self) -> str:
        return (f'in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, out_dim={self.out_dim}, '
                f'num_layers={self.num_layers}, dropout={self.dropout}, use_pyg={self.use_pyg}')

# Factory function for creating models
def create_gcn_with_lora(in_dim: int, hidden_dim: int, out_dim: int,
                        lora_rank: int = 8, lora_alpha: float = 1.0,
                        num_layers: int = 2, dropout: float = 0.5,
                        use_pyg: Optional[bool] = None) -> GCNWithLoRA:
    """
    Factory function for creating GCNWithLoRA instances.

    Args:
        in_dim: Number of input features
        hidden_dim: Number of hidden features
        out_dim: Number of output classes
        lora_rank: Rank for LoRA adaptation
        lora_alpha: Scaling factor for LoRA
        num_layers: Number of GCN layers
        dropout: Dropout probability
        use_pyg: Whether to use PyTorch Geometric (None for auto-detect)

    Returns:
        Configured GCNWithLoRA instance
    """
    if use_pyg is None:
        use_pyg = PYG_AVAILABLE

    return GCNWithLoRA(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        num_layers=num_layers,
        dropout=dropout,
        use_pyg=use_pyg
    )
