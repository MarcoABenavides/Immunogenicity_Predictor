import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphNorm, global_mean_pool

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, dropout=0.1):
        super(GNN, self).__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Initial Graph Attention Layer (Multi-head attention)
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim))  # More heads
        self.norms.append(GraphNorm(hidden_dim))

        # Intermediate GCN Layers (More depth)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))

        # Last GCN Layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.norms.append(GraphNorm(hidden_dim))

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)  # Use edge features if available

        for i, conv in enumerate(self.convs):
            residual = x  # Store input for residual connection

            # Apply GNN layers
            if isinstance(conv, GATConv):
                x = conv(x, edge_index, edge_attr) if edge_attr is not None else conv(x, edge_index)
            else:
                x = conv(x, edge_index)

            x = self.norms[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)

            # Apply residual connection if dimensions match
            if i > 0 and x.shape == residual.shape:
                x = x + residual

            x = self.dropout(x)

        # Global mean pooling for graph-level representation
        x = global_mean_pool(x, batch)

        # Fully connected layers with more depth
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
