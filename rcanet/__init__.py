"""RCANet: Row-Column Attention Networks for Tabular Data

A dual-axis transformer framework for enhanced representation learning on tabular data.
"""

__version__ = "0.1.0"
__author__ = "RCANet Team"

from .models.rcanet import RCANet
from .models.attention import RowAttention, ColumnAttention
from .models.fusion import HierarchicalAggregation

__all__ = [
    "RCANet",
    "RowAttention", 
    "ColumnAttention",
    "HierarchicalAggregation"
]