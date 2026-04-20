import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.data import Data
from easydict import EasyDict


dtype_mapping = {
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "float64": torch.float64,
    "torch.float64": torch.float64,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "int32": torch.int32,
    "torch.int32": torch.int32,
}



class MAB(nn.Module):
    """Multi-head Attention Block with proper residual connections"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout=0.1):
        super(MAB, self).__init__()
        assert dim_V % num_heads == 0, "dim_V must be divisible by num_heads"
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.dim_head = dim_V // num_heads
        
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K):
        B, N_q, _ = Q.shape
        B, N_k, _ = K.shape
        Q_proj = self.fc_q(Q).view(B, N_q, self.num_heads, self.dim_head)
        K_proj = self.fc_k(K).view(B, N_k, self.num_heads, self.dim_head)
        V_proj = self.fc_v(K).view(B, N_k, self.num_heads, self.dim_head)
        

        Q_proj = Q_proj.transpose(1, 2)
        K_proj = K_proj.transpose(1, 2)
        V_proj = V_proj.transpose(1, 2)
        

        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(self.dim_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        

        out = torch.matmul(attn, V_proj)  
        

        out = out.transpose(1, 2).contiguous().view(B, N_q, self.dim_V)
        

        out = self.fc_o(out)
        out = self.dropout(out)
        

        if hasattr(self, 'ln0'):
            out = self.ln0(out)
        

        if hasattr(self, 'ln1'):
            out_ffn = F.relu(out)
            out = self.ln1(out + out_ffn)
            
        return out


class SAB(nn.Module):
    """Set Attention Block - self attention"""
    def __init__(self, dim_in, dim_out, num_heads, ln=False, dropout=0.1):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        return self.mab(X, X)




class EquivariantSetBlock(nn.Module):
    """
    Permutation-equivariant block using stacked SAB layers.
    Maps [B, N, in_dim] -> [B, N, out_dim] while preserving permutation equivariance.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=1, num_layers=2, 
                 ln=True, dropout=0.1):
        super().__init__()
        
        layers = []
        
        if num_layers == 1:

            layers.append(SAB(in_dim, out_dim, num_heads, ln=ln, dropout=dropout))
        else:

            layers.append(SAB(in_dim, hidden_dim, num_heads, ln=ln, dropout=dropout))
            
  
            for _ in range(num_layers - 2):
                layers.append(SAB(hidden_dim, hidden_dim, num_heads, ln=ln, dropout=dropout))
            

            if num_layers > 1:
                layers.append(SAB(hidden_dim, out_dim, num_heads, ln=ln, dropout=dropout))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, X):

        for i, layer in enumerate(self.layers):
            if i > 0 and X.size(-1) == layer.mab.dim_V:

                X_out = layer(X)
                X = X + X_out
            else:
                X = layer(X)
                
        return X




class SetTransformerModel(nn.Module):
    """
    Set-based Transformer model that treats the graph as a set of nodes (ignores edges).
    This is different from TransformerConv which is a graph convolution layer.
    Designed to match the interface of GraphModel for seamless integration.
    """
    def __init__(self, args: EasyDict):
        super().__init__()
        dtype = dtype_mapping.get(args.dtype, torch.float32)
        
        self.use_layer_norm = args.use_layer_norm
        self.use_residual = args.use_residual
        self.use_activation = args.use_activation
        self.num_layers = args.depth
        self.h_dim = 32 
        self.out_dim = args.out_dim
        self.in_dim = args.in_dim
        self.task_type = args.task_type
        

        self.num_heads = getattr(args, 'num_heads', 2)
        self.dropout = getattr(args, 'dropout', 0.1)
        

        self.transformer = EquivariantSetBlock(
            in_dim=self.in_dim,
            hidden_dim=self.h_dim,
            out_dim=self.h_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ln=self.use_layer_norm,
            dropout=self.dropout
        )
        

        self.out_layer = nn.Linear(self.h_dim, self.out_dim, dtype=dtype)
        

        self.init_model()
    
    def init_model(self):
        """Initialize model parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.out_layer.weight)
    
    def forward(self, data: Data):
        """
        Forward pass through the transformer model.
        
        Args:
            data (Data): PyTorch Geometric Data object
            
        Returns:
            Tensor: Output predictions of shape [num_nodes, out_dim]
        """
        x = self.compute_node_embedding(data)
        return self.out_layer(x)
    
    def compute_node_embedding(self, data: Data):
        """
        Compute node embeddings using transformer layers.
        This method matches GraphModel's interface.
        
        Args:
            data (Data): PyTorch Geometric Data object
            
        Returns:
            Tensor: Node embeddings of shape [num_nodes, h_dim]
        """
        x = data.x  
        
       
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = data.batch.max().item() + 1
            max_nodes = (data.ptr[1:] - data.ptr[:-1]).max().item()
            

            x_batched = torch.zeros(batch_size, max_nodes, x.size(1), 
                                  dtype=x.dtype, device=x.device)
            

            for i in range(batch_size):
                start_idx = data.ptr[i].item()
                end_idx = data.ptr[i + 1].item()
                num_nodes = end_idx - start_idx
                x_batched[i, :num_nodes] = x[start_idx:end_idx]
        else:
   
            x_batched = x.unsqueeze(0)

        x_batched = self.transformer(x_batched)  

        if hasattr(data, 'batch') and data.batch is not None:
            x_flat = []
            for i in range(batch_size):
                start_idx = data.ptr[i].item()
                end_idx = data.ptr[i + 1].item()
                num_nodes = end_idx - start_idx
                x_flat.append(x_batched[i, :num_nodes])
            x = torch.cat(x_flat, dim=0)
        else:
            x = x_batched.squeeze(0)
        
        return x