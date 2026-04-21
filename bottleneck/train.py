import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything, callbacks
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from models.lightning_model import LightningModel, StopAtValAccCallback, CSVLogger, AccuracyPrintCallback
from models.graph_model import (
    GraphModelWithVirtualNode,
    GraphModel,
    GraphModelWithMultipleVirtualNodes,
    GraphModelWithProbabilisticVirtualNodes,
)
from models.transformer import SetTransformerModel
from models.Sumformer import SumformerModel
from models.MLP import MLPModel
from utils import get_args, create_model_dir, return_datasets, compute_energy


RESULTS_CSV_COLUMNS = [
    'timestamp', 'gnn_type', 'task_type', 'star_variant', 'n', 'K',
    'depth', 'dim', 'lr', 'lr_schedule', 'lr_factor',
    'batch_size', 'max_epochs', 'target_acc',
    'use_virtual_nodes', 'num_virtual_nodes', 'vn_aggregation',
    'prob_vn', 'num_vn', 'vn_per_node',
    'vn_tau_schedule', 'vn_tau_start', 'vn_tau_end', 'vn_tau_anneal_epochs',
    'num_heads', 'dropout', 'seed',
    'test_acc', 'best_val_acc', 'epochs_run',
    'grad_norm', 'dirichlet',
    'num_train_samples', 'num_test_samples', 'run_name',
]


def _to_float(x):
    """Coerce tensor / number / None into a plain Python float (or None)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def append_result_row(results_csv_path: str, row: dict) -> None:
    """Append a single run's results to a CSV file; create header if new."""
    path = Path(results_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
    elif getattr(args, 'prob_vn', False):
        assert not getattr(args, 'use_virtual_nodes', False), (
            "Cannot combine --prob_vn with --use_virtual_nodes; pick one."
        )
        base_model = GraphModelWithProbabilisticVirtualNodes(args=args)
        print(
            f"Using Probabilistic Virtual Nodes | m={getattr(args,'num_vn','NA')}, "
            f"d={getattr(args,'vn_per_node','NA')}, "
            f"tau_sched={getattr(args,'vn_tau_schedule','exp')}, "
            f"tau_start={getattr(args,'vn_tau_start','NA')} -> "
            f"tau_end={getattr(args,'vn_tau_end','NA')} over "
            f"{getattr(args,'vn_tau_anneal_epochs','NA')} epochs"
        )
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

    prob_vn_tag = ""
    if getattr(args, 'prob_vn', False):
        prob_vn_tag = (
            f"_probVN_m{getattr(args, 'num_vn', 'NA')}"
            f"_d{getattr(args, 'vn_per_node', 'NA')}"
            f"_tau{getattr(args, 'vn_tau_start', 'NA')}->"
            f"{getattr(args, 'vn_tau_end', 'NA')}"
            f"_{getattr(args, 'vn_tau_schedule', 'exp')}"
        )
    run_name = (
        f"{args.gnn_type}_{args.task_type}_{args.star_variant}"
        f"_n{args.n}_K{K}_dim{getattr(args, 'dim', 'NA')}"
        f"_lr{getattr(args, 'lr', 'NA')}_VN{args.use_virtual_nodes}"
        f"_h{getattr(args, 'heads', 'NA')}_seed{getattr(args, 'seed', 0)}"
        f"{prob_vn_tag}"
    )
    csv_logger = CSVLogger('csv_logs', name=run_name)

    loggers = [csv_logger]
    wandb_logger = None
    if getattr(args, 'use_wandb', False):
        wandb_logger = WandbLogger(
            project=getattr(args, 'wandb_project', 'short-range-oversquashing'),
            name=run_name,
            config=dict(args),
            reinit=True,
        )
        loggers.append(wandb_logger)

    target_acc = float(getattr(args, 'target_acc', 0.92))
    stop_callback = StopAtValAccCallback(target_acc=target_acc) if args.task_type == 'two' else None
    callbacks_list = [callback for callback in [checkpoint_callback, stop_callback] if callback]
    print_callback = AccuracyPrintCallback()
    callbacks_list = [callback for callback in [checkpoint_callback, stop_callback, print_callback] if callback]
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if num_gpus >= 2:

        trainer = Trainer(
            logger=loggers,
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
            logger=loggers,
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
            logger=loggers,
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
    elif getattr(args, 'prob_vn', False):
        base_model = GraphModelWithProbabilisticVirtualNodes(args=args)
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

    best_val_acc = None
    if getattr(checkpoint_callback, 'best_model_score', None) is not None:
        best_val_acc = float(checkpoint_callback.best_model_score)
    epochs_run = int(trainer.current_epoch) + 1

    grad_norm_val = _to_float(energy.get('grad_norm')) if isinstance(energy, dict) else None
    dirichlet_val = _to_float(energy.get('dirichlet')) if isinstance(energy, dict) else None

    row = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
        'gnn_type': args.gnn_type,
        'task_type': args.task_type,
        'star_variant': args.star_variant,
        'n': args.n,
        'K': K,
        'depth': args.depth,
        'dim': getattr(args, 'dim', None),
        'lr': getattr(args, 'lr', None),
        'lr_schedule': getattr(args, 'lr_schedule', 'none'),
        'lr_factor': getattr(args, 'lr_factor', None),
        'batch_size': getattr(args, 'batch_size', None),
        'max_epochs': getattr(args, 'max_epochs', None),
        'target_acc': target_acc if args.task_type == 'two' else None,
        'use_virtual_nodes': bool(getattr(args, 'use_virtual_nodes', False)),
        'num_virtual_nodes': getattr(args, 'num_virtual_nodes', None),
        'vn_aggregation': getattr(args, 'vn_aggregation', None),
        'prob_vn': bool(getattr(args, 'prob_vn', False)),
        'num_vn': getattr(args, 'num_vn', None),
        'vn_per_node': getattr(args, 'vn_per_node', None),
        'vn_tau_schedule': getattr(args, 'vn_tau_schedule', None),
        'vn_tau_start': getattr(args, 'vn_tau_start', None),
        'vn_tau_end': getattr(args, 'vn_tau_end', None),
        'vn_tau_anneal_epochs': getattr(args, 'vn_tau_anneal_epochs', None),
        'num_heads': getattr(args, 'num_heads', None),
        'dropout': getattr(args, 'dropout', None),
        'seed': getattr(args, 'seed', seed),
        'test_acc': test_accuracy,
        'best_val_acc': best_val_acc,
        'epochs_run': epochs_run,
        'grad_norm': grad_norm_val,
        'dirichlet': dirichlet_val,
        'num_train_samples': getattr(args, 'num_train_samples', None),
        'num_test_samples': getattr(args, 'num_test_samples', None),
        'run_name': run_name,
    }
    results_csv_path = getattr(args, 'results_csv', 'results/results.csv')
    append_result_row(results_csv_path, row)
    print(f"Appended run summary to {results_csv_path}")

    if wandb_logger is not None:
        try:
            wandb_logger.experiment.finish()
        except Exception as exc:
            print(f"(wandb finish failed, continuing) {exc}")

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
    parser.add_argument('--use_residual', dest='use_residual', action='store_true',
                        default=None,
                        help='Enable backbone MPNN residual (input→output skip per layer). '
                             'Overrides the YAML. Default: use YAML value.')
    parser.add_argument('--no_residual', dest='use_residual', action='store_false',
                        help='Disable backbone MPNN residual. Overrides the YAML.')
    parser.add_argument('--vn_aggregation', type=str, default=None,
                        choices=['sum', 'mean'],
                        help='Aggregation method for multiple virtual nodes (default: sum).')
    # Probabilistic Virtual Nodes (IPR-MPNN-style, simplified)
    parser.add_argument('--prob_vn', action='store_true', default=False,
                        help='Enable probabilistic VN rewiring (Gumbel top-k soft '
                             'routing + per-layer VN update). Cannot be combined '
                             'with --use_virtual_nodes.')
    parser.add_argument('--num_vn', type=int, default=None,
                        help='Number of virtual nodes m for --prob_vn (default from YAML: 10).')
    parser.add_argument('--vn_per_node', type=int, default=None,
                        help='Connections-per-node d for --prob_vn; must satisfy d <= m '
                             '(default from YAML: 2).')
    parser.add_argument('--vn_tau_schedule', type=str, default=None,
                        choices=['constant', 'linear', 'exp'],
                        help='Gumbel temperature schedule for --prob_vn '
                             '(default from YAML: exp).')
    parser.add_argument('--vn_tau_start', type=float, default=None,
                        help='Initial Gumbel temperature tau at epoch 0 (default 5.0).')
    parser.add_argument('--vn_tau_end', type=float, default=None,
                        help='Final Gumbel temperature tau after annealing (default 0.1).')
    parser.add_argument('--vn_tau_anneal_epochs', type=int, default=None,
                        help='Number of epochs over which tau anneals from start to end '
                             '(default 100). Epochs beyond this are pinned at vn_tau_end.')
    parser.add_argument('--K', type=int, default=1, 
                        help='Number of central nodes for two-radius problem (default: 1).')
    parser.add_argument('--num_heads', type=int, default=1, 
                        help='Number of attention heads for SetTransformer model.')
    parser.add_argument('--heads', type=int, default=None,
                        help=('Override GAT attention heads (Task_specific.GAT.<task>.heads '
                              'in the YAML). Leave unset to use the YAML value.'))
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for SetTransformer model.')
    parser.add_argument('--dim', type=int, default=None,
                        help=('Override the hidden-feature dimension (Common.dim in the YAML). '
                              'Leave unset to use the YAML value. Example: --dim 256.'))
    parser.add_argument('--lr', type=float, default=None,
                        help=('Override the learning rate (Task_specific.<model>.<task>.lr in '
                              'the YAML). Leave unset to use the YAML value. Example: --lr 5e-4.'))
    parser.add_argument('--max_epochs', type=int, default=None,
                        help=('Override the maximum number of training epochs '
                              '(Task_specific.<model>.<task>.max_epochs in the YAML). '
                              'Leave unset to use the YAML value. Example: --max_epochs 500.'))
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help=('Override num_train_samples (number of training graphs '
                              'per epoch). Leave unset to use the YAML value.'))
    parser.add_argument('--num_test_samples', type=int, default=None,
                        help=('Override num_test_samples (size of both the test and '
                              'val splits). Leave unset to use the YAML value.'))
    parser.add_argument('--seed', type=int, default=0,
                        help=('Random seed for data generation, model init, and '
                              'DataLoader workers. Default 0 (reproduces prior runs). '
                              'Vary this to assess seed-to-seed variance.'))
    parser.add_argument('--lr_schedule', type=str, default='none',
                        choices=['none', 'plateau_train', 'plateau_val'],
                        help=('Learning-rate schedule. '
                              '"none" (default) keeps lr constant. '
                              '"plateau_train" is the original repo behaviour '
                              '(ReduceLROnPlateau monitoring train_acc). '
                              '"plateau_val" monitors val_acc instead — note that '
                              'with eval_every>1 the effective patience is inflated, '
                              'so set --eval_every 1 if you want patience to mean epochs.'))
    parser.add_argument('--lr_factor', type=float, default=None,
                        help=('Override lr_factor used by ReduceLROnPlateau '
                              '(multiplicative LR decay on plateau). Paper uses '
                              '0.1 for GCN. Only applies when --lr_schedule != none.'))
    parser.add_argument('--target_acc', type=float, default=0.92,
                        help=('Validation-accuracy threshold for early-exit on two-radius '
                              'runs (default 0.92 reproduces paper). Pass >1.0 (e.g. 1.01) '
                              'to disable and train the full max_epochs.'))
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Log training curves (train/val/test accuracy, loss, lr) to '
                             'Weights & Biases alongside the existing CSV logger.')
    parser.add_argument('--wandb_project', type=str, default='short-range-oversquashing',
                        help='W&B project name (only used when --wandb is set).')
    parser.add_argument('--results_csv', type=str, default='results/results.csv',
                        help='Path to a CSV file that receives one row per completed run '
                             '(final test accuracy + metadata). Created on first write.')
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

        if args.dim is not None:
            config_args.dim = args.dim
        if args.lr is not None:
            config_args.lr = args.lr
        if args.max_epochs is not None:
            config_args.max_epochs = args.max_epochs
        if args.num_train_samples is not None:
            config_args.num_train_samples = args.num_train_samples
        if args.num_test_samples is not None:
            config_args.num_test_samples = args.num_test_samples
        if args.heads is not None:
            config_args.heads = args.heads
        if args.use_residual is not None:
            config_args.use_residual = args.use_residual
        config_args.lr_schedule = args.lr_schedule
        if args.lr_factor is not None:
            config_args.lr_factor = args.lr_factor

        # Probabilistic virtual nodes
        config_args.prob_vn = bool(args.prob_vn)
        if args.prob_vn:
            # Ensure the base VN code path is never picked alongside prob_vn.
            config_args.use_virtual_nodes = False
        if args.num_vn is not None:
            config_args.num_vn = args.num_vn
        if args.vn_per_node is not None:
            config_args.vn_per_node = args.vn_per_node
        if args.vn_tau_schedule is not None:
            config_args.vn_tau_schedule = args.vn_tau_schedule
        if args.vn_tau_start is not None:
            config_args.vn_tau_start = args.vn_tau_start
        if args.vn_tau_end is not None:
            config_args.vn_tau_end = args.vn_tau_end
        if args.vn_tau_anneal_epochs is not None:
            config_args.vn_tau_anneal_epochs = args.vn_tau_anneal_epochs

        config_args.target_acc = args.target_acc
        config_args.use_wandb = args.wandb
        config_args.wandb_project = args.wandb_project
        config_args.results_csv = args.results_csv


        seed = args.seed
        config_args.seed = seed
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