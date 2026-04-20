import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

class MLPModel(nn.Module):
    """
    Simple MLP for two-radius star graph.
    Concatenates all nodes and predicts B node labels.
    """
    
    def __init__(self, args: EasyDict):
        super(MLPModel, self).__init__()
        
        self.n = args.n
        self.K = getattr(args, 'K', 1)
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.hidden_dim = getattr(args, 'mlp_hidden_dim', 1024)
        self.depth = args.depth
        self.dropout = getattr(args, 'dropout', 0.3)
        

        self.total_nodes = 2 * self.n + self.K
        self.flat_dim = self.total_nodes * self.in_dim
        

        layers = []
        in_features = self.flat_dim
        
        for i in range(self.depth - 1):
            layers.append(nn.Linear(in_features, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_features = self.hidden_dim
        layers.append(nn.Linear(in_features, self.n * self.out_dim))
        
        self.mlp = nn.Sequential(*layers)
        

        self.embedding_mlp = nn.Sequential(*layers[:-1])  
    
    def compute_node_embedding(self, batch):
        """
        Compute node embeddings for compatibility with energy computation.
        Returns embeddings of shape [num_nodes, hidden_dim].
        """
        x = batch.x
        batch_idx = batch.batch
        batch_size = batch_idx.max().item() + 1
        

        embeddings = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        

        for b in range(batch_size):

            mask = batch_idx == b
            node_indices = mask.nonzero(as_tuple=True)[0]
            nodes = x[mask]
            flat = nodes.reshape(-1)
            emb = self.embedding_mlp(flat)  
            b_start = self.n + self.K
            b_indices = node_indices[b_start:b_start + self.n]
            embeddings[b_indices] = emb.unsqueeze(0).expand(self.n, -1)
        
        return embeddings
    
    def forward(self, batch):
        x = batch.x
        batch_idx = batch.batch
        batch_size = batch_idx.max().item() + 1
        

        outputs = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        

        for b in range(batch_size):
            mask = batch_idx == b
            node_indices = mask.nonzero(as_tuple=True)[0]
            nodes = x[mask]
            flat = nodes.reshape(-1)
            pred = self.mlp(flat)  
            pred = pred.reshape(self.n, self.out_dim)  
            

            b_start = self.n + self.K
            b_indices = node_indices[b_start:b_start + self.n]
            outputs[b_indices] = pred  
        
        return outputs