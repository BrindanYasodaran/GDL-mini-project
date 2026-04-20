
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from easydict import EasyDict

dtype_mapping = {
    "float32": torch.float32, "torch.float32": torch.float32,
    "float64": torch.float64, "torch.float64": torch.float64,
    "float16": torch.float16, "torch.float16": torch.float16,
    "int32": torch.int32, "torch.int32": torch.int32,
}

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2, dropout=0.0, use_ln=False, dtype=torch.float32):
        super().__init__()
        layers = []
        d = in_dim
        
        for i in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(d, hidden_dim, dtype=dtype))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_ln:
                layers.append(nn.LayerNorm(hidden_dim))
            d = hidden_dim
            
        layers.append(nn.Linear(d, out_dim, dtype=dtype))
        self.net = nn.Sequential(*layers)
        
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)

class SumformerBlock(nn.Module):
    """
    Sumformer block from the paper:
    S([x1, ..., xn]) = [?(x1, ?), ..., ?(xn, ?)]
    where ? = ?_k ?(xk)
    """
    def __init__(self, in_dim, sum_dim, out_dim, hidden_dim, dropout=0.0, use_ln=False, residual=True, dtype=torch.float32):
        super().__init__()
        
        self.residual = residual and (in_dim == out_dim)
        

        self.phi = MLP(in_dim, sum_dim, hidden_dim, num_layers=2, dropout=dropout, use_ln=use_ln, dtype=dtype)
        

        self.psi = MLP(in_dim + sum_dim, out_dim, hidden_dim, num_layers=2, dropout=dropout, use_ln=use_ln, dtype=dtype)
        
        self.out_ln = nn.LayerNorm(out_dim) if use_ln else None

    def forward(self, x, batch):

        phi_x = self.phi(x)  
        

        global_sum = global_add_pool(phi_x, batch)  
        

        expanded_sum = global_sum[batch]  
        

        combined = torch.cat([x, expanded_sum], dim=-1) 
        output = self.psi(combined)  
        
        if self.residual:
            output = x + output
            
        if self.out_ln is not None:
            output = self.out_ln(output)
            
        return output

class SumformerModel(nn.Module):
    """
    Sumformer model from "Sumformer: Universal Approximation for Efficient Transformers"
    """
    def __init__(self, args: EasyDict):
        super().__init__()
        
        dtype = dtype_mapping.get(args.dtype, torch.float32)
        
        self.use_layer_norm = args.use_layer_norm
        self.use_residual = args.use_residual
        self.num_layers = args.depth
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        
        self.h_dim = getattr(args, "dim", 256)
        self.sum_dim = getattr(args, "sum_dim", 256) 
        self.dropout = getattr(args, "dropout", 0.0)
        

        blocks = []
        in_d = self.in_dim
        
        for i in range(self.num_layers):
            blocks.append(
                SumformerBlock(
                    in_dim=in_d,
                    sum_dim=self.sum_dim,
                    out_dim=self.h_dim,
                    hidden_dim=self.h_dim,
                    dropout=self.dropout,
                    use_ln=self.use_layer_norm,
                    residual=self.use_residual,
                    dtype=dtype
                )
            )
            in_d = self.h_dim
            
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(self.h_dim, self.out_dim, dtype=dtype)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, data):
        x = data.x
        
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        

        for block in self.blocks:
            x = block(x, batch)
        
        output = self.head(x)
        
        return output
        
        