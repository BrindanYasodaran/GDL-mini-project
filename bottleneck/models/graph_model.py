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
    """
    Graph model with **probabilistic** virtual-node rewiring.

    Architecture (once per forward):
      1. Input projection  : X        -> H_real^(0)   in (N, dim)
      2. Router (once)     : H_real^(0) -> logits     in (N, m)
         Gumbel top-k softmax with temperature tau gives soft assignment
         S in (N, m); each row sums to 1 and has exactly `vn_per_node` (=k)
         non-zero entries.
      3. Per-layer t = 1..L, with VN table H_VN in (B, m, dim):
         A. H_local    = GNN_layer(H_real, original_edge_index)
         B. vn_agg     = S^T  @ H_local        (bmm over batch)
            H_VN      <- MLP_up( H_VN + vn_agg )
         C. H_global   = S    @ H_VN           (bmm over batch)
            H_real'   = H_local + H_global  (+ residual / layernorm / activation)

    Differs from IPR-MPNN (Qian et al. 2024):
      - router is an MLP, not an MPNN
      - sampling is Gumbel top-k with soft softmax (paper uses SIMPLE k-subset)
      - no VN<->VN complete graph
      - single sample per forward (q=1)
      - VN init is a learnable table broadcast across graphs
    """

    def __init__(self, args: EasyDict):
        super().__init__(args)

        dtype = dtype_mapping[args.dtype]

        self.num_vn = int(getattr(args, 'num_vn', 10))
        self.vn_per_node = int(getattr(args, 'vn_per_node', 2))
        assert 1 <= self.vn_per_node <= self.num_vn, (
            f"vn_per_node must satisfy 1 <= d <= m; got d={self.vn_per_node}, m={self.num_vn}"
        )

        # Oracle routing: if True, bypass the router MLP and Gumbel sampling and
        # build S directly from node identifiers so that every source shares a VN
        # with its matching target. Used to ablate "is the VN mechanism itself
        # useful, given perfect routing?".
        self.oracle_routing = bool(getattr(args, 'oracle_routing', False))
        # Whether centers also participate in the VN mechanism under oracle routing.
        # False (default) -> centers get zero rows (clean source/target channel).
        # True              -> centers route to VN `id mod m`, matching what a learned
        #                      id-keyed router would do.
        self.oracle_route_centers = bool(getattr(args, 'oracle_route_centers', False))

        # Straight-through estimator: forward pass uses hard top-k one-hot routing
        # (discriminative VN pathway from epoch 0, matches the oracle's forward),
        # backward pass uses dense soft Gumbel-softmax gradient (all logits learn).
        # Fixes the "uniform S => information-free VN pathway => dead router" failure
        # mode observed with dense soft routing.
        self.vn_ste = bool(getattr(args, 'vn_ste', False))

        # Router logit function:
        #   'mlp'    (original): logits_i = MLP(H_real_i). Two-layer MLP with
        #            LeakyReLU. Blind projection; fails on two-radius because the
        #            MLP scrambles the natural similarity between source_i and
        #            target_i's raw features, so they route differently.
        #   'dot'    (Solution 1): logits = (W x) @ vn_emb^T / sqrt(h). Learnable
        #            projection W + attention against shared vn_emb anchors.
        #   'simple' (Solution 1 minimal): logits = x @ vn_anchors^T, with
        #            vn_anchors living in raw INPUT space (no projection) and
        #            initialized as orthogonal unit rows. Separate anchor table
        #            from vn_emb (which stays in hidden space for VN states).
        #            No Gumbel, no STE, no annealing — plain softmax at a fixed
        #            small tau. Should work if orthogonal anchors + sharp softmax
        #            give enough rendezvous at init.
        self.vn_router_type = str(getattr(args, 'vn_router', 'mlp')).lower()
        assert self.vn_router_type in ('mlp', 'dot', 'simple'), (
            f"vn_router must be 'mlp', 'dot', or 'simple'; got {self.vn_router_type!r}"
        )

        self.register_buffer('tau', torch.tensor(float(getattr(args, 'vn_tau_start', 5.0))))

        self.in_lin = nn.Linear(self.in_dim, self.h_dim, dtype=dtype)
        self.layers = nn.ModuleList([
            get_layer(in_dim=self.h_dim, out_dim=self.h_dim, args=args)
            for _ in range(self.num_layers)
        ])

        if self.vn_router_type == 'mlp':
            router_hidden = int(getattr(args, 'vn_rewire_hidden', self.h_dim))
            self.router = nn.Sequential(
                nn.Linear(self.h_dim, router_hidden, dtype=dtype),
                nn.LeakyReLU(),
                nn.Linear(router_hidden, self.num_vn, dtype=dtype),
            )
            self.router_proj = None
            self.vn_anchors = None
        elif self.vn_router_type == 'dot':
            # Dot-product router: project raw node features (not H_real) into
            # hidden space and attend to shared vn_emb anchors.
            self.router = None
            self.router_proj = nn.Linear(self.in_dim, self.h_dim, bias=False, dtype=dtype)
            self.vn_anchors = None
        else:
            # 'simple': project real nodes and VN anchors into a shared router
            # hidden space (d_router), then dot product. Two separate learnable
            # tables: router_proj for reals, vn_anchors for VNs. d_router < h
            # (e.g. 256) keeps the routing subspace small and decoupled from the
            # main GAT hidden dim.
            self.router = None
            d_router = 64
            self.router_proj = nn.Linear(self.in_dim, d_router, bias=False, dtype=dtype)
            self.vn_anchors = nn.Parameter(
                torch.empty(self.num_vn, d_router, dtype=dtype)
            )
            # Random Gaussian init + unit-norm rows on anchors (same choice as
            # before, just now in d_router-dim space rather than in_dim-dim).
            nn.init.normal_(self.vn_anchors)
            with torch.no_grad():
                self.vn_anchors.data = F.normalize(self.vn_anchors.data, dim=-1)

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
        """Mutate the Gumbel temperature (called from Lightning per-epoch hook)."""
        self.tau.fill_(float(tau_value))

    def _gumbel_topk_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Sparse Gumbel top-k softmax: rows sum to 1, exactly k non-zeros.

        If self.vn_ste is True, uses straight-through: forward is hard top-k
        (each row has k entries each = 1/k, rest = 0); backward is dense soft
        softmax over all m logits.
        """
        k = self.vn_per_node
        tau = self.tau.clamp_min(1e-4)
        U = torch.rand_like(logits).clamp_(1e-9, 1 - 1e-9)
        gumbel = -torch.log(-torch.log(U))
        noisy = (logits + gumbel) / tau

        if self.vn_ste:
            # Dense soft distribution over all m logits — carries gradient.
            soft = F.softmax(noisy, dim=-1)
            # Hard top-k assignment — used in forward only.
            hard = torch.zeros_like(soft)
            if k == 1:
                idx = noisy.argmax(dim=-1, keepdim=True)
                hard.scatter_(1, idx, 1.0)
            else:
                topk_idx = noisy.topk(k, dim=-1).indices
                hard.scatter_(1, topk_idx, 1.0 / k)
            # Straight-through: forward = hard, gradient flows through soft.
            return hard - soft.detach() + soft

        topk_vals, topk_idx = noisy.topk(k, dim=-1)
        masked = torch.full_like(noisy, float('-inf'))
        masked.scatter_(1, topk_idx, topk_vals)
        return F.softmax(masked, dim=-1)

    def _oracle_S(self, data: Data, N: int, B: int, m: int) -> torch.Tensor:
        """
        Hand-crafted routing matrix S of shape (N, m):
          - Each source / target node v is routed to exactly one VN, at index
            `identifier(v) mod m`. Sources and matching targets share an
            identifier, so they share a VN.
          - Centers: routed to `id mod m` iff `self.oracle_route_centers` is
            True; otherwise their row is left all-zero (excluded from the VN
            mechanism).
        Assumes uniform graph size and the default TwoRadius node layout
        [A (sources) | C (centers) | B (targets)].
        """
        x = data.x
        device = x.device
        dtype = x.dtype

        # First (in_dim - num_classes) feature dims are the one-hot identifier.
        id_dim = self.in_dim - self.out_dim
        assert id_dim > 0, f"Expected positive id_dim, got {id_dim}"
        ids = x[:, :id_dim].argmax(dim=-1)                                 # (N,)

        # Infer per-graph layout. For TwoRadius: graph_size = 2n + K, with
        # `test_mask` True on the last n nodes of each graph (the targets).
        graph_size = N // B
        n_targets_total = int(data.test_mask.sum().item())
        assert n_targets_total % B == 0, (
            f"test_mask sum {n_targets_total} not divisible by batch size {B}; "
            "oracle routing assumes uniform target count per graph."
        )
        n = n_targets_total // B
        K_centers = graph_size - 2 * n

        local_idx = torch.arange(N, device=device) % graph_size            # (N,)
        is_source = local_idx < n
        is_target = local_idx >= (n + K_centers)
        is_st = is_source | is_target                                      # (N,)

        S = torch.zeros(N, m, device=device, dtype=dtype)
        vn_idx = (ids % m).long()                                          # (N,)
        route_mask = is_st if not self.oracle_route_centers else torch.ones_like(is_st)
        rows = torch.nonzero(route_mask, as_tuple=False).squeeze(-1)
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

        H_real = self.in_lin(x)                                              # (N, d)

        if self.oracle_routing:
            S = self._oracle_S(data, N, B, m)                                # (N, m)
            with torch.no_grad():
                self._last_routing_entropy = torch.tensor(0.0, device=x.device)
                self._last_mean_top1 = S.max(dim=-1).values.mean().detach()
        elif self.vn_router_type == 'simple':
            # Project real nodes via router_proj(x) -> Z in R^{d_router}, and
            # attend against vn_anchors living in the same d_router space.
            # Plain temperature-softmax, no Gumbel, no STE, no annealing.
            tau = self.tau.clamp_min(1e-4)
            Z = self.router_proj(x)                                           # (N, d_router)
            logits = Z @ self.vn_anchors.t()                                  # (N, m)
            S = F.softmax(logits / tau, dim=-1)                               # (N, m)
            with torch.no_grad():
                eps = 1e-9
                ent = -(S.clamp_min(eps) * S.clamp_min(eps).log()).sum(-1).mean()
                self._last_routing_entropy = ent.detach()
                self._last_mean_top1 = S.max(dim=-1).values.mean().detach()
        else:
            if self.vn_router_type == 'dot':
                # Attention-style router: project raw x, dot against VN anchors.
                # Source_i and target_i have ~identical raw features on the id
                # bits, so their projections align and they get nearly-identical
                # logits -> same argmax -> natural rendezvous.
                Z = self.router_proj(x)                                       # (N, d)
                logits = Z @ self.vn_emb.weight.t() / math.sqrt(d)            # (N, m)
            else:
                logits = self.router(H_real)                                  # (N, m)
            S = self._gumbel_topk_softmax(logits)                            # (N, m)
            with torch.no_grad():
                eps = 1e-9
                ent = -(S.clamp_min(eps) * S.clamp_min(eps).log()).sum(-1).mean()
                self._last_routing_entropy = ent.detach()
                self._last_mean_top1 = S.max(dim=-1).values.mean().detach()

        S_b = S.view(B, n_pg, m)                                             # (B, n, m)
        H_real_b = H_real.view(B, n_pg, d)                                   # (B, n, d)

        H_VN = self.vn_emb.weight.unsqueeze(0).expand(B, m, d).contiguous()  # (B, m, d)

        for t, (gnn_layer, vn_mlp) in enumerate(zip(self.layers, self.vn_update)):
            H_local_flat = gnn_layer(H_real_b.reshape(N, d), edge_index, edge_attr) \
                if edge_attr is not None else gnn_layer(H_real_b.reshape(N, d), edge_index)
            H_local_b = H_local_flat.view(B, n_pg, d)

            vn_agg = torch.bmm(S_b.transpose(1, 2), H_local_b)               # (B, m, d)
            H_VN = vn_mlp(H_VN + vn_agg)

            H_global_b = torch.bmm(S_b, H_VN)                                # (B, n, d)

            H_next = H_local_b + H_global_b
            if self.use_residual and t > 0:
                H_next = H_next + H_real_b
            if self.use_layer_norm:
                H_next = self.layer_norms[t](H_next)
            if self.use_activation:
                H_next = F.leaky_relu(H_next)
            H_real_b = H_next

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