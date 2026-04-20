import pathlib
import yaml
from easydict import EasyDict
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv, SAGEConv, TransformerConv
from models.transformer import SetTransformerModel
from models.Sumformer import SumformerModel
from data_generate.graphs_generation import TwoRadiusProblemStarGraph, TwoRadiusProblemDisconnectedGraph, OneRadiusProblemStarGraph

def get_layer(args: EasyDict, in_dim: int, out_dim: int):
    """
    Get a GNN layer based on the specified type in args.

    Args:
        args (EasyDict): Configuration dictionary with `gnn_type`.
        in_dim (int): Input dimension of the layer.
        out_dim (int): Output dimension of the layer.

    Returns:
        nn.Module: Initialized GNN layer.
    """
    gnn_layers = {
        'GCN': lambda: GCNConv(in_channels=in_dim, out_channels=out_dim),
        'GGNN': lambda: GatedGraphConv(out_channels=out_dim, num_layers=3),
        'GIN': lambda: GINConv(
            nn.Sequential(
                nn.Linear(in_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, out_dim),
                nn.ReLU(),
            ),
            eps=args.eps if hasattr(args, 'eps') else 0.2,
            train_eps=True
        ),
        'GAT': lambda: GATConv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=args.heads,  
            concat=False  
        ),
        'SAGE': lambda: SAGEConv(in_channels=in_dim, out_channels=out_dim, aggr='sum'),
        'Transformer': lambda: TransformerConv(in_channels=in_dim, out_channels=out_dim)
    }
    

    if args.gnn_type not in gnn_layers:
        raise ValueError(f"Unknown GNN type: {args.gnn_type}. Available types: {list(gnn_layers.keys())}")
    
    return gnn_layers[args.gnn_type]()


def get_args(gnn_type: str, task_type: str, n: int, depth: int = 2, star_variant: str = 'connected'):
    """
    Load and update arguments from a YAML configuration file.

    Args:
        depth (int): Depth of the model.
        gnn_type (str): Type of GNN layer or 'SetTransformer' for our set-based model.
        task_type (str): Task type for dataset generation ('two' or 'one').
        n (int): Number of A and B nodes.
        star_variant (str): Variant of star graph ('connected' or 'disconnected').

    Returns:
        tuple: Configuration arguments and task-specific settings.
    """
    clean_args = EasyDict(depth=depth, gnn_type=gnn_type, task_type=task_type, n=n, star_variant=star_variant)
    config_path = pathlib.Path(__file__).parent / "configs" / "task_config.yaml"
    
    with open(config_path) as f:
        config = EasyDict(yaml.safe_load(f))


    clean_args.update(config['Common'])

    if gnn_type in ['SetTransformer', 'MLP', 'Sumformer']:

        if gnn_type in config['Task_specific']:
            clean_args.update(config['Task_specific'][gnn_type][task_type])
        else:

            clean_args.update(config['Task_specific']['GIN'][task_type])
        clean_args.gnn_type = gnn_type 
    else:

        if gnn_type in config['Task_specific'] and task_type in config['Task_specific'][gnn_type]:
            clean_args.update(config['Task_specific'][gnn_type][task_type])
        else:
            raise ValueError(f"Configuration not found for {gnn_type} with task {task_type}")
    
    return clean_args, config['Task_specific'].get(gnn_type, {}).get(task_type, {})


def return_datasets(args):
    """
    Load datasets based on task type and update configuration dimensions.

    Args:
        args (EasyDict): Configuration arguments.

    Returns:
        tuple: Training, testing, and validation datasets, and K value.
    """
    if args.task_type == "one":

        rad_star = OneRadiusProblemStarGraph(args=args)
        dataset = rad_star.generate_data()
        K = rad_star.K
        

        sample_data = dataset[0][0]  
        args.in_dim = sample_data.x.size(1)  
        args.out_dim = rad_star.num_classes  
        
        return dataset, K
        
    elif args.task_type == "two":

        if args.star_variant == 'disconnected':
            rad_star = TwoRadiusProblemDisconnectedGraph(args=args)
        else:  
            rad_star = TwoRadiusProblemStarGraph(args=args)
        
        dataset = rad_star.generate_data()
        K = rad_star.K
        

        sample_data = dataset[0][0]  
        args.in_dim = sample_data.x.size(1)  
        args.out_dim = rad_star.num_classes  
        
        return dataset, K
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}. Must be 'one' or 'two'")


def create_model_dir(args, task_specific):
    """Create model directory path for saving checkpoints."""
    if args.gnn_type == 'SetTransformer':
        model_type_name = 'SetTransformer'
    else:
        model_type_name = args.gnn_type
        
    model_name = '_'.join([f"{key}_{val}" for key, val in task_specific.items()])
    model_name = model_name[:100]
    
    path_to_project = pathlib.Path(__file__).parent.parent
    model_dir = path_to_project / f"data/models/{args.task_type}/{model_type_name}/Radius_{args.depth}/{args.n}/{model_name}"
    return str(model_dir), str(path_to_project)


def compute_os_energy_batched(model, Data):
    """Compute oversquashing energy using gradient norms."""
    model = model.eval()


    if torch.isnan(Data.x).any() or torch.isinf(Data.x).any():
        raise ValueError("Data.x contains NaNs or Infs")


    x = Data.x.clone().detach().requires_grad_(True)


    M = Data.test_mask.sum()
    G = Data.num_graphs
    T = M // G if G > 0 else 0

    if M == 0:
        raise ValueError("No test nodes found in test_mask.")
    
    ptc = Data.ptr[:-1].view(-1, 1)
    sources = (ptc.squeeze().repeat_interleave(Data.n[0]) + Data.sources)


    def model_target(x_local):
        Data.x = x_local
        return model(Data)[Data.test_mask]
    with torch.autograd.set_detect_anomaly(True):
        jacobian = torch.autograd.functional.jacobian(model_target, x)
    
    num_targets = jacobian.shape[0]
    out = jacobian[torch.arange(num_targets, device=jacobian.device), :, sources, :]
    norms = out.pow(2).mean(dim=(1, 2)).sqrt()
    return norms.mean()


def compute_energy(model, test_dl):
    """Compute energy metrics for the model."""
    energies = {}
    model = model.to('cuda')
    num_samples = len(test_dl)
    
    for batch in test_dl:
        batch = batch.to('cuda')
        forb_norm = compute_os_energy_batched(model.model, batch)
        derichlet_energy = compute_b_node_mad(model.model, batch)
        energies['grad_norm'] = energies.get('grad_norm', 0) + forb_norm / num_samples
        energies['dirichlet'] = derichlet_energy
        break

    return energies


def compute_b_node_mad(model, test_batch):
    """
    Computes the mean pairwise distance between B node representations, 
    normalized by the mean norm of the B node embeddings.
    
    B nodes are identified by test_batch.test_mask.
    """

    model = model.eval().to('cuda')
    batch = test_batch.to('cuda')
    
    with torch.no_grad():

        h = model.compute_node_embedding(batch)
        b_embeddings = h[batch.test_mask]
        n = b_embeddings.shape[0]

        if n < 2:
            return torch.tensor(0.0, device=b_embeddings.device)

        unique_pairwise_dists = torch.pdist(b_embeddings, p=2)
        mean_dist = unique_pairwise_dists.mean()
        embedding_norms = torch.norm(b_embeddings, p=2, dim=1)
        avg_norm = embedding_norms.mean()

        if avg_norm == 0:
            return torch.tensor(0.0, device=b_embeddings.device)

        final_metric = mean_dist / avg_norm
        
    return final_metric