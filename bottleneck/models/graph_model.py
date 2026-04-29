import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from easydict import EasyDict
from utils import get_layer


# Mapping for data types
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


class GraphModel(nn.Module):
    """
    Base graph neural network model for node-level classification.
    Includes support for residual connections, layer normalization, and activation.
    """
    def __init__(self, args: EasyDict):
        super().__init__()
        dtype = dtype_mapping[args.dtype]
        

        self.use_layer_norm = args.use_layer_norm
        self.use_residual = args.use_residual
        self.use_activation = args.use_activation
        self.num_layers = args.depth
        self.h_dim = getattr(args, 'dim', 256)
        self.out_dim = args.out_dim
        self.in_dim = args.in_dim   
        self.task_type = args.task_type
        

        self.layers = nn.ModuleList([
            get_layer(in_dim=self.in_dim, out_dim=self.h_dim, args=args)
        ] + [
            get_layer(in_dim=self.h_dim, out_dim=self.h_dim, args=args)
            for i in range(self.num_layers-1)
        ])
        

        self.layer_norms = (
            nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_layers)])
            if self.use_layer_norm else None
        )
        

        self.out_layer = nn.Linear(self.h_dim, self.out_dim) 
        

        self.init_model()
 
    def init_model(self):
        """Initialize model parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.out_layer.weight)

    def forward(self, data: Data):
        """Forward pass through the graph model."""
        x = self.compute_node_embedding(data)
        return self.out_layer(x) 
    
    def compute_node_embedding(self, data: Data):
        """Compute node embeddings through message passing layers."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            layer_output = x.clone()
            

            layer_output = layer(layer_output, edge_index, edge_attr)
            

            if self.use_residual and i > 0:
                x = layer_output + x
            else:
                x = layer_output
                
 
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
                

            if self.use_activation:
                x = F.leaky_relu(x)
                
        return x


class GraphModelWithVirtualNode(GraphModel):
    """
    Graph model with single virtual node support.
    Inherits from GraphModel and adds virtual node functionality.
    """
    def __init__(self, args: EasyDict):

        super().__init__(args)
        
        dtype = dtype_mapping[args.dtype]
        

        self.use_virtual_nodes = getattr(args, 'use_virtual_nodes', True)
        self.vn_residual = getattr(args, 'vn_residual', True)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.vn_aggregation = getattr(args, 'vn_aggregation', 'sum')

        if self.use_virtual_nodes:

            self.virtualnode_embedding = nn.Embedding(1, self.h_dim, dtype=dtype)
            nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            

            self.mlp_virtualnode_list = nn.ModuleList()
            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(
                    nn.Sequential(
                        nn.Linear(self.h_dim, self.h_dim, dtype=dtype),
                        nn.BatchNorm1d(self.h_dim, dtype=dtype),
                        nn.ReLU(),
                        nn.Linear(self.h_dim, self.h_dim, dtype=dtype),
                        nn.BatchNorm1d(self.h_dim, dtype=dtype),
                        nn.ReLU()
                    )
                )

    def compute_node_embedding(self, data: Data):
        """Override to add virtual node functionality."""
        x, edge_index = data.x, data.edge_index
        device = x.device
        
        if self.use_virtual_nodes:
            if hasattr(data, 'batch') and data.batch is not None:
                batch_tensor = data.batch
                num_graphs = batch_tensor.max().item() + 1
            else:
                batch_tensor = torch.zeros(x.size(0), dtype=torch.long, device=device)
                num_graphs = 1
            
            vn_emb = self.virtualnode_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=device)
            )
        
        for i, layer in enumerate(self.layers):
            x_input = x
            if hasattr(layer, '__call__'):
                if edge_index.size(1) > 0: 
                    x = layer(x, edge_index)
                else:
                    if hasattr(layer, 'lin'):
                        x = layer.lin(x)
                    elif hasattr(layer, 'linear'):
                        x = layer.linear(x)
                    else:
                        x = layer(x, edge_index)

            if self.use_residual and i > 0:
                x = x + x_input

            if self.use_virtual_nodes:
                x = x + vn_emb[batch_tensor]
            
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            if self.use_activation:
                x = F.leaky_relu(x)
            
            if self.use_virtual_nodes and i < self.num_layers - 1:
                if self.vn_aggregation == 'mean':
                    vn_emb_temp = global_mean_pool(x, batch_tensor) + vn_emb
                else:
                    vn_emb_temp = global_add_pool(x, batch_tensor) + vn_emb
                
                vn_update = self.mlp_virtualnode_list[i](vn_emb_temp)
                vn_update = F.dropout(vn_update, p=self.dropout, training=self.training)
                
                if self.vn_residual:
                    vn_emb = vn_emb + vn_update
                else:
                    vn_emb = vn_update
        
        return x


class GraphModelWithProbabilisticVirtualNodes(GraphModel):
    def __init__(self, args: EasyDict):
        super().__init__(args)

        dtype = dtype_mapping[args.dtype]

        self.num_vn = int(getattr(args, 'num_vn', 10))

        self.oracle_routing = bool(getattr(args, 'oracle_routing', False))

        self.vn_router_type = str(getattr(args, 'vn_router', 'tied')).lower()
        assert self.vn_router_type in ('decoupled', 'tied', 'adaptive'), (
            f"vn_router must be 'decoupled', 'tied', or 'adaptive'; "
            f"got {self.vn_router_type!r}"
        )

        self.vn_d_router = int(getattr(args, 'vn_d_router', 64))

        self.register_buffer('tau', torch.tensor(float(getattr(args, 'vn_tau_start', 5.0))))

        self.in_lin = nn.Linear(self.in_dim, self.h_dim, dtype=dtype)
        self.layers = nn.ModuleList([
            get_layer(in_dim=self.h_dim, out_dim=self.h_dim, args=args)
            for _ in range(self.num_layers)
        ])

        if self.vn_router_type == 'decoupled':
            d_r = self.vn_d_router
            self.router_proj = nn.Linear(self.in_dim, d_r, bias=False, dtype=dtype)
            self.vn_anchors = nn.Parameter(
                torch.empty(self.num_vn, d_r, dtype=dtype)
            )
            nn.init.normal_(self.vn_anchors)
            with torch.no_grad():
                self.vn_anchors.data = F.normalize(self.vn_anchors.data, dim=-1)
        else:
            self.router_proj = None
            self.vn_anchors = None

        self.vn_emb = nn.Embedding(self.num_vn, self.h_dim, dtype=dtype)

        self.vn_update = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.h_dim),
                nn.Linear(self.h_dim, self.h_dim, dtype=dtype),
                nn.LeakyReLU(),
                nn.Linear(self.h_dim, self.h_dim, dtype=dtype),
            ) for _ in range(self.num_layers)
        ])

        self._last_routing_entropy = None
        self._last_mean_top1 = None

    def set_tau(self, tau_value: float):
        """Set routing temperature."""
        self.tau.fill_(float(tau_value))

    def _oracle_S(self, data: Data, N: int, m: int) -> torch.Tensor:
        x = data.x
        device = x.device
        dtype = x.dtype

        id_dim = self.in_dim - self.out_dim
        assert id_dim > 0, f"Expected positive id_dim, got {id_dim}"
        ids = x[:, :id_dim].argmax(dim=-1)

        S = torch.zeros(N, m, device=device, dtype=dtype)
        vn_idx = (ids % m).long()
        rows = torch.arange(N, device=device)
        S[rows, vn_idx[rows]] = 1.0
        return S

    def compute_node_embedding(self, data: Data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

        if hasattr(data, 'batch') and data.batch is not None:
            B = int(data.batch.max().item()) + 1
        else:
            B = 1
        N = x.size(0)
        assert N % B == 0, (
            f"GraphModelWithProbabilisticVirtualNodes requires uniform graph size; "
            f"got N={N}, B={B} (N % B = {N % B})"
        )
        n_pg = N // B
        d = self.h_dim
        m = self.num_vn

        H_real = self.in_lin(x)

        S_b = None
        if self.oracle_routing:
            S = self._oracle_S(data, N, m)
            with torch.no_grad():
                self._last_routing_entropy = torch.tensor(0.0, device=x.device)
                self._last_mean_top1 = S.max(dim=-1).values.mean().detach()
            S_b = S.view(B, n_pg, m)
        elif self.vn_router_type == 'decoupled':
            tau = self.tau.clamp_min(1e-4)
            Z = self.router_proj(x)
            logits = Z @ self.vn_anchors.t()
            S = F.softmax(logits / tau, dim=-1)
            with torch.no_grad():
                eps = 1e-9
                ent = -(S.clamp_min(eps) * S.clamp_min(eps).log()).sum(-1).mean()
                self._last_routing_entropy = ent.detach()
                self._last_mean_top1 = S.max(dim=-1).values.mean().detach()
            S_b = S.view(B, n_pg, m)
        elif self.vn_router_type == 'tied':
            tau = self.tau.clamp_min(1e-4)
            logits = H_real @ self.vn_emb.weight.t() / math.sqrt(d)
            S = F.softmax(logits / tau, dim=-1)
            with torch.no_grad():
                eps = 1e-9
                ent = -(S.clamp_min(eps) * S.clamp_min(eps).log()).sum(-1).mean()
                self._last_routing_entropy = ent.detach()
                self._last_mean_top1 = S.max(dim=-1).values.mean().detach()
            S_b = S.view(B, n_pg, m)
        else:  # 'adaptive' (APW)
            pass

        H_real_b = H_real.view(B, n_pg, d)

        H_VN = self.vn_emb.weight.unsqueeze(0).expand(B, m, d).contiguous()

        for t, (gnn_layer, vn_mlp) in enumerate(zip(self.layers, self.vn_update)):
            if self.vn_router_type == 'adaptive' and not self.oracle_routing:
                tau = self.tau.clamp_min(1e-4)
                logits_t = torch.bmm(H_real_b, H_VN.transpose(1, 2)) / math.sqrt(d)
                S_b = F.softmax(logits_t / tau, dim=-1)

            H_local_flat = gnn_layer(H_real_b.reshape(N, d), edge_index, edge_attr) \
                if edge_attr is not None else gnn_layer(H_real_b.reshape(N, d), edge_index)
            H_local_b = H_local_flat.view(B, n_pg, d)

            vn_agg = torch.bmm(S_b.transpose(1, 2), H_local_b)
            H_VN = vn_mlp(H_VN + vn_agg)

            H_global_b = torch.bmm(S_b, H_VN)

            H_next = H_local_b + H_global_b
            if self.use_residual and t > 0:
                H_next = H_next + H_real_b
            if self.use_layer_norm:
                H_next = self.layer_norms[t](H_next)
            if self.use_activation:
                H_next = F.leaky_relu(H_next)
            H_real_b = H_next

        if self.vn_router_type == 'adaptive' and not self.oracle_routing and S_b is not None:
            with torch.no_grad():
                S_flat = S_b.reshape(N, m)
                eps = 1e-9
                ent = -(S_flat.clamp_min(eps) * S_flat.clamp_min(eps).log()).sum(-1).mean()
                self._last_routing_entropy = ent.detach()
                self._last_mean_top1 = S_flat.max(dim=-1).values.mean().detach()

        return H_real_b.reshape(N, d)


class GraphModelWithMultipleVirtualNodes(GraphModel):
    """
    Graph model with multiple virtual nodes.
    Inherits from GraphModel and adds multiple virtual node functionality.
    """
    def __init__(self, args: EasyDict):

        super().__init__(args)
        
        dtype = dtype_mapping[args.dtype]


        self.use_virtual_nodes = getattr(args, 'use_virtual_nodes', True)
        self.num_virtual_nodes = getattr(args, 'num_virtual_nodes', 3)
        self.vn_aggregation = getattr(args, 'vn_aggregation', 'mean')
        self.vn_residual = getattr(args, 'vn_residual', True)
        self.dropout = getattr(args, 'dropout', 0.1)

        if self.use_virtual_nodes:
            self.virtualnode_embedding = nn.Embedding(self.num_virtual_nodes, self.h_dim, dtype=dtype)
            nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = nn.ModuleList()
            for _ in range(self.num_layers - 1):
                layer_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.h_dim, 128, dtype=dtype),
                        nn.LayerNorm(128),
                        nn.ReLU(),
                        nn.Linear(128, self.h_dim, dtype=dtype),
                        nn.LayerNorm(self.h_dim),
                        nn.ReLU()
                    ) for _ in range(self.num_virtual_nodes)
                ])
                self.mlp_virtualnode_list.append(layer_mlps)

    def compute_node_embedding(self, data: Data):
        """Override to add multiple virtual node functionality."""
        x, edge_index = data.x, data.edge_index
        device = x.device
        batch_tensor = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)
        num_graphs = batch_tensor.max().item() + 1

        if self.use_virtual_nodes:
            vn_emb = self.virtualnode_embedding.weight.unsqueeze(0).repeat(num_graphs, 1, 1)
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index) if edge_index.size(1) > 0 else x

            if self.use_virtual_nodes:
                aggregated_vn = vn_emb.mean(dim=1)
                x = x + aggregated_vn[batch_tensor]

            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            if self.use_activation:
                x = F.leaky_relu(x)

            if self.use_virtual_nodes and i < self.num_layers - 1:
                if self.vn_aggregation == 'mean':
                    global_info = global_mean_pool(x, batch_tensor)
                else:
                    global_info = global_add_pool(x, batch_tensor)

                for vn_idx in range(self.num_virtual_nodes):
                    mlp = self.mlp_virtualnode_list[i][vn_idx]
                    vn_input = global_info + vn_emb[:, vn_idx]
                    vn_output = mlp(vn_input)
                    vn_output = F.dropout(vn_output, p=self.dropout, training=self.training)
                    
                    if self.vn_residual:
                        vn_emb[:, vn_idx] = vn_emb[:, vn_idx] + vn_output
                    else:
                        vn_emb[:, vn_idx] = vn_output

        return x