# Short-Range Oversquashing
![Alt text](image/image.png)
Research project investigating oversquashing phenomena in Message Passing Neural Networks (MPNNs), specifically focusing on short-range information bottlenecks in graph neural networks.

## Description

This Paper addresses the oversquashing problem in Graph Neural Networks (GNNs), where information bottlenecks prevent effective message passing between nodes. The research implements and evaluates various mechanisms to mitigate oversquashing issues, by Analysis of the two-radius problem in star-graphs with Multiple architectural of GNN variants, Transformer variants, Multilayer Perceptron and Virtual nodes.
Implementation with PyTorch, PyTorch Geometric, and PyTorch Lightning.
---

## Features

- **Multiple Architectures**: Support for GIN, GAT, GCN, GGNN, GraphSAGE, Graph Transformer, FSW ,SetTransformer and Sumformer implementations, MLP
- **Virtual Nodes**: Single and multiple virtual node configurations
- **Synthetic Benchmarks**: Two-radius and one-radius problem generators
- **Comprehensive Logging**: CSV logging with PyTorch Lightning
- **Flexible Configuration**: Command-line arguments and YAML config files
- **Multi-GPU Training**: Distributed training support with DDP for parallel processing across multiple GPUs
- **weights and biases(wandb) option to use**
---

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended)
- Conda or pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YakovM93/short-range-oversquashing.git
   cd short-range-oversquashing
   ```

2. **Create a virtual environment**:
   ```bash
   conda create --name oversquash python=3.11 -c conda-forge
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
```bash
cd bottleneck 
```



### Basic Training

Train a GIN model on the two-radius star graph problem:

```bash
python train.py --model_type GIN --task_type two --star_variant connected --start 2 --end 10
```

# Training Examples

## Default Settings
By default, the following parameters are used:
- `model_type`: GIN
- `task_type`: two (two-radius problem)
- `star_variant`: connected (with central nodes)
- `use_virtual_nodes`: False (no virtual nodes)
- `K`: 1 (single central node)
- `start`: 2, `end`: 3 (trains on n=2 only)
- `mlp_hidden_dim`: 512
- `num_heads`: 1 (for SetTransformer)
- `dropout`: 0.1

## Experiment Categories

### One-Radius Problem
```bash

python train.py --model_type GCN --task_type one --start 2 --end 10

python train.py --model_type GIN --task_type one --start 10 --end 50

python train.py --model_type SAGE --task_type one --start 50 --end 100

python train.py --model_type GAT --task_type one --start 50 --end 100
```

### Two-Radius Problem (Connected)

```bash

python train.py --model_type GAT --task_type two --star_variant connected --start 2 --end 20

python train.py --model_type GIN --task_type two --star_variant connected --start 20 --end 50

python train.py --model_type GCN --task_type two --star_variant connected --start 50 --end 100

python train.py --model_type SAGE --task_type one --start 50 --end 100
```

### Two-Radius Problem (Disconnected)
```bash
python train.py --model_type GCN --task_type two --star_variant disconnected --start 2 --end 15

```

### Varying Number of Central Nodes (K)
```bash
# You don't need to write it if it's default

python train.py --model_type GIN --task_type two --star_variant connected --K 2 --start 10 --end 30

python train.py --model_type GCN --task_type two --star_variant connected --K 3 --start 10 --end 30


```
### One Virtual Node
```bash
python train.py --model_type GCN --task_type two --star_variant connceted --use_virtual_nodes  --start 10 --end 40

python train.py --model_type SAGE --task_type two --star_variant connceted --use_virtual_nodes   --vn_aggregation sum --start 10 --end 11

python train.py --model_type GCN --task_type two --star_variant disconnceted --use_virtual_nodes  --start 10 --end 40
```

### Virtual Nodes
```bash
python train.py --model_type GAT --use_virtual_nodes --num_virtual_nodes 2 --vn_aggregation mean --start 10 --end 100

python train.py --model_type GIN --task_type two --star_variant disconnected --use_virtual_nodes --num_virtual_nodes 5 --start 10 --end 50

```

### Disable virtual nodes explicitly (if needed)
```bash
python train.py --model_type GCN --no_virtual_nodes --start 10 --end 200
```


### SetTransformer (Attention-Based)
```bash
python train.py --model_type SetTransformer --task_type two --num_heads 8 --start 2 --end 50

python train.py --model_type SetTransformer --task_type two --star_variant disconnected --start 2 --end 50

```

### MLP - Multilayer Perceptron
```bash
python train.py --model_type MLP --task_type two --mlp_hidden_dim 512 --start 2 --end 20

```


###  Sumformer
```bash
python train.py --model_type Sumformer --task_type two --dropout 0.15 --start 2 --end 50

```

### SAGE with virtual nodes and multiple central nodes and connected graph
```bash
python train.py --model_type SAGE --task_type two --star_variant connected --use_virtual_nodes --num_virtual_nodes 3 --vn_aggregation sum --K 5 --start 100 --end 500
```


### Quick test run 
```bash
python train.py --model_type GAT  --start 10 --end 200
```

## Notes
- For MLP models, the `mlp_hidden_dim` parameter controls model capacity
- The `K` parameter only affects two-radius problems with connected variant
- SetTransformer ignores graph structure and treats nodes as a set
- Disconnected variant removes central nodes entirely



## Configuration

Model configurations are defined in `configs/task_config.yaml`. Key parameters:

- **Learning rate**: Model-specific learning rates
- **Batch size**: Training batch size
- **Hidden dimensions**: Model hidden layer dimensions
- **Number of layers**: GNN depth
- **Dropout**: Dropout rate for regularization

---

## Experiments

The project supports experiments on:

1. **Two-Radius Problem**: Nodes are separated by distance 2 through central nodes
2. **One-Radius Problem**: Source nodes connectet to one target node
3. **Disconnected Graphs**: Testing information flow without direct paths

### Energy Metrics

The code computes two  metrics:
- **Gradient Norm**: Measures gradient flow from source to target nodes
- **Dirichlet Energy**: Measures smoothness of node representations (MAD normalized by embedding norms)

---
 
This  research conducted at the Technion – Israel Institute of Technology.
---


---



## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

**Yaakov Mishayev**  
Technion – Israel Institute of Technology  
Email: yakov-m@campus.technion.ac.il

**Yonatan Sverdlov**  
Technion – Israel Institute of Technology  
Email: yonatans@campus.technion.ac.il

For questions, feedback, or collaboration opportunities, feel free to reach out!

---
