import argparse
import os
import random
import numpy as np
import torch
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything, callbacks
from torch_geometric.loader import DataLoader
from models.lightning_model import LightningModel, StopAtValAccCallback, CSVLogger, AccuracyPrintCallback
from models.graph_model import GraphModelWithVirtualNode, GraphModel, GraphModelWithMultipleVirtualNodes
from models.transformer import SetTransformerModel
from models.Sumformer import SumformerModel
from models.MLP import MLPModel
from utils import get_args, create_model_dir, return_datasets, compute_energy
# from pytorch_lightning.loggers import WandbLogger  # Uncomment if using Weights & Biases


def worker_init_fn(seed: int):
    """
    Initializes random seeds for reproducibility in data loading workers.

    Args:
        seed (int): The seed value to ensure consistent data shuffling.
    """
    np.random.seed(seed)
    random.seed(seed)


def train_graphs(args: EasyDict, task_specific: dict, task_id: int, seed: int) -> tuple:
    """
    Train, validate, and test a graph model on the specified dataset.

    Args:
        args (EasyDict): Configuration containing hyperparameters and dataset details.
        task_specific (dict): Task-specific identifier used for creating model directories.
        task_id (int): Unique identifier for multi-task settings.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (test_accuracy, energy) - Test accuracy and energy values
    """

    (X_train, X_test, X_val), K = return_datasets(args=args)
    
    model_dir, path_to_project = create_model_dir(args, task_specific)
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-{val_acc:.5f}' + f'K_{K}',
        save_top_k=1,
        monitor='val_acc',
        save_last=True,
        mode='max'
    )

    if args.gnn_type == 'SetTransformer':
        base_model = SetTransformerModel(args=args)
        print(f"Using SetTransformer Model (ignores edges)")
        print(f"  - Heads: {getattr(args, 'num_heads', 4)}, Layers: {args.depth}")
    elif args.gnn_type == 'Sumformer':
        base_model = SumformerModel(args=args)
        print(f"Using Sumformer (phi-sum-psi) | depth={args.depth}, dim={getattr(args, 'dim', 256)}, dropout={getattr(args,'dropout', 0.0)}")
    elif args.gnn_type == 'MLP':
        base_model = MLPModel(args=args)
        print(f"Using MLP Model (ignores edges, no inter-node communication)")
        print(f"  - Hidden dim: {getattr(args, 'mlp_hidden_dim', 256)}, Layers: {args.depth}")
    elif args.use_virtual_nodes:
        num_vns = getattr(args, 'num_virtual_nodes', 1)
        if num_vns > 1:
            base_model = GraphModelWithMultipleVirtualNodes(args=args)
        else:
            base_model = GraphModelWithVirtualNode(args=args)
            print("Using Single Virtual Node")
    else:
        base_model = GraphModel(args=args)
        print(f"Using {args.gnn_type} (without virtual nodes)")

    model = LightningModel(args=args, task_id=task_id, model=base_model)

    csv_logger = CSVLogger('csv_logs', name=f"{args.gnn_type}_{args.task_type}_{args.star_variant}_{args.n}_{K}_VN_{args.use_virtual_nodes}")


    stop_callback = StopAtValAccCallback(target_acc=0.92) if args.task_type == 'two' else None
    callbacks_list = [callback for callback in [checkpoint_callback, stop_callback] if callback]
    print_callback = AccuracyPrintCallback()
    callbacks_list = [callback for callback in [checkpoint_callback, stop_callback, print_callback] if callback]
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if num_gpus >= 2:

        trainer = Trainer(
            logger=csv_logger,
            max_epochs=args.max_epochs,
            accelerator='gpu',
            devices=1, 
            enable_progress_bar=True,
            check_val_every_n_epoch=args.eval_every,
            callbacks=callbacks_list,
            enable_checkpointing=True,
            default_root_dir=f'{path_to_project}/data/lightning_logs',

        )
        print(f"Training on 1 GPU")  
    elif num_gpus == 1:
        trainer = Trainer(
            logger=csv_logger,
            max_epochs=args.max_epochs,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=True,
            check_val_every_n_epoch=args.eval_every,
            callbacks=callbacks_list,
            enable_checkpointing=True,
            default_root_dir=f'{path_to_project}/data/lightning_logs'
        )
        print(f"Training on 1 GPU")
    else:

        trainer = Trainer(
            logger=csv_logger,
            max_epochs=args.max_epochs,
            accelerator='cpu',
            enable_progress_bar=True,
            check_val_every_n_epoch=args.eval_every,
            callbacks=callbacks_list,
            enable_checkpointing=True,
            default_root_dir=f'{path_to_project}/data/lightning_logs'
        )
        print(f"Training on CPU")


    train_batch_size = args.batch_size
    val_batch_size = args.val_batch_size


    train_loader = DataLoader(
        X_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.loader_workers,
        persistent_workers=True,
        worker_init_fn=lambda _: worker_init_fn(seed)
    )
    
    val_loader = DataLoader(
        X_val,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.loader_workers
    )
    
    test_loader = DataLoader(
        X_test,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.loader_workers
    )

    # Train the model
    print(f'Starting training with star_variant: {args.star_variant}, virtual_nodes: {args.use_virtual_nodes}...')
    trainer.fit(model, train_loader, val_loader)


    print("Loading best model checkpoint...")
    best_checkpoint_path = checkpoint_callback.best_model_path
    

    if args.gnn_type == 'SetTransformer':
        base_model = SetTransformerModel(args=args)
    elif args.gnn_type == 'Sumformer':
        base_model = SumformerModel(args=args)
    elif args.gnn_type == 'MLP':
        base_model = MLPModel(args=args)
    elif args.use_virtual_nodes:
        num_vns = getattr(args, 'num_virtual_nodes', 1)
        if num_vns > 1:
            base_model = GraphModelWithMultipleVirtualNodes(args=args)
        else:
            base_model = GraphModelWithVirtualNode(args=args)
    else:
        base_model = GraphModel(args=args)
    
    model = LightningModel.load_from_checkpoint(best_checkpoint_path, args=args, task_id=task_id, model=base_model)
    

    test_loader_energy = DataLoader(
        X_test[:5],
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.loader_workers
    )


    energy = compute_energy(model, test_loader_energy)
    

    if isinstance(energy, torch.Tensor):
        energy = energy.item()
        
    test_results = trainer.test(model, test_loader, verbose=False)
    test_accuracy = test_results[0]['test_acc'] * 100
    
    return test_accuracy, energy


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train graph models on specified datasets.")
    parser.add_argument('--model_type', type=str, default='GIN', help='Model type for training.')
    parser.add_argument('--task_type', type=str, default='two', 
                        choices=['two', 'one'], 
                        help='Task type for training: two (two-radius) or one (one-radius).')
    parser.add_argument('--star_variant', type=str, default='connected', 
                        choices=['connected', 'disconnected'], 
                        help='Variant for star graph: connected (with central nodes) or disconnected.')
    parser.add_argument('--start', type=int, default=2, help='Starting value for parameter n.')
    parser.add_argument('--end', type=int, default=3, help='Ending value (exclusive) for parameter n.')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512,
                        help='Hidden dimension for MLP model.')
    parser.add_argument('--use_virtual_nodes', action='store_true', default=False, 
                        help='Enable virtual nodes.')
    parser.add_argument('--no_virtual_nodes', dest='use_virtual_nodes', action='store_false', 
                        help='Disable virtual nodes.')
    parser.add_argument('--num_virtual_nodes', type=int, default=None, 
                        help='Number of virtual nodes to use (default: use config file).')
    parser.add_argument('--vn_aggregation', type=str, default=None,
                        choices=['sum', 'mean'],
                        help='Aggregation method for multiple virtual nodes (default: sum).')
    parser.add_argument('--K', type=int, default=1, 
                        help='Number of central nodes for two-radius problem (default: 1).')
    parser.add_argument('--num_heads', type=int, default=1, 
                        help='Number of attention heads for SetTransformer model.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for SetTransformer model.')
    return parser.parse_args()


def main():
    """
    Main function to execute training and testing over various depth values.
    """
    args = parse_arguments()
    depth = 4
    model_type, task_type, start, end = args.model_type, args.task_type, args.start, args.end
    test_accs = []
    test_energies = []
    
    for n in range(start, end):
        config_args, task_specific = get_args(
            depth=depth, 
            gnn_type=model_type, 
            n=n, 
            task_type=task_type, 
            star_variant=args.star_variant
        )
        
        if model_type == 'SetTransformer':
            config_args.num_heads = args.num_heads
            config_args.dropout = args.dropout
        elif model_type == 'MLP':
            config_args.mlp_hidden_dim = args.mlp_hidden_dim
        

        if hasattr(args, 'use_virtual_nodes'):
            config_args.use_virtual_nodes = args.use_virtual_nodes
        if args.num_virtual_nodes is not None:
            config_args.num_virtual_nodes = args.num_virtual_nodes
        if args.vn_aggregation is not None:
            config_args.vn_aggregation = args.vn_aggregation
        

        if task_type == 'two' and args.star_variant == 'connected' and not config_args.use_virtual_nodes:
            config_args.K = args.K
        

        seed = 0
        config_args.need_one_hot = True
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        seed_everything(seed, workers=True)
        test_acc, energy = train_graphs(args=config_args, task_specific=task_specific, task_id=0, seed=seed)
        test_accs.append(test_acc)
        test_energies.append(energy)
    

    for i, n in enumerate(range(start, end)):
        vn_info = ""
        k_info = ""
        
        if model_type == 'SetTransformer':
            print(f"SetTransformer: {depth} layers, n={n}, variant={args.star_variant}, "
                  f"heads={args.num_heads}, dropout={args.dropout}, "
                  f"accuracy: {test_accs[i]:.2f}, "
                  f"grad_energy: {test_energies[i]['grad_norm']:.6f}, "
                  f"dirichlet_energy: {test_energies[i]['dirichlet']:.6f}")
        elif model_type == 'Sumformer':
            print(f"Sumformer: {depth} layers, n={n}, variant={args.star_variant}, "
                  f"accuracy: {test_accs[i]:.2f}, "
                  f"grad_energy: {test_energies[i]['grad_norm']:.6f}, "
                  f"dirichlet_energy: {test_energies[i]['dirichlet']:.6f}")
        elif model_type == 'MLP':
            print(f"MLP: {depth} layers, n={n}, variant={args.star_variant}, "
                  f"hidden_dim={args.mlp_hidden_dim}, "
                  f"accuracy: {test_accs[i]:.2f}, "
                  f"grad_energy: {test_energies[i]['grad_norm']:.6f}, "
                  f"dirichlet_energy: {test_energies[i]['dirichlet']:.6f}")
        else:
            if config_args.use_virtual_nodes:
                num_vns = getattr(config_args, 'num_virtual_nodes', 1)
                if num_vns > 1:
                    vn_info = f", VNs={num_vns}, agg={config_args.vn_aggregation}"
                else:
                    vn_info = ", VN=1"
            elif task_type == 'two' and args.star_variant == 'connected':
                k_info = f", K={getattr(config_args, 'K', args.K)}"
            
            print(f"Using {depth} layers, n={n}, variant={args.star_variant}{vn_info}{k_info}, "
                  f"accuracy: {test_accs[i]:.2f}, "
                  f"grad_energy: {test_energies[i]['grad_norm']:.6f}, "
                  f"dirichlet_energy: {test_energies[i]['dirichlet']:.6f}")


if __name__ == "__main__":
    main()