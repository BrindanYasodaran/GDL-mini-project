# A18332S1 Geometric Deep Learning 

## Contribution

This submission builds on the existing codebase from Mishayev et al. for Two-Radius
data generation, the training loop, baseline GNNs, and the SetTransformer model
My main contribution is the `GraphModelWithProbabilisticVirtualNodes` class in
`bottleneck/models/graph_model.py` which allow for DPW/TPW/APW virtual node options on top of any base MPNN. This class also has the option for fixed identifier-matched virtual node wiring. I also added the corresponding CLI/config options in
`bottleneck/train.py`.

## How to run virtual node experiments

**Single run (example: GAT, Two-Radius, probabilistic VN, tied router):**

```bash
cd bottleneck
python train.py --task_type two --star_variant connected \
  --start 100 --end 101 --model_type GAT --dim 1024 --heads 2 --lr 2e-4 \
  --prob_vn --num_vn 25 --vn_router tied --seed 1 --max_epochs 100
```

Main added arguments:

- `--prob_vn`: use probabilistic virtual nodes.
- `--num_vn`: number of virtual nodes.
- `--vn_router {decoupled,tied,adaptive}`: DPW, TPW, or APW routing.
- `--vn_d_router`: DPW routing subspace dimension.
- `--vn_tau_start`, `--vn_tau_end`, `--vn_tau_schedule`, `--vn_tau_anneal_epochs`: routing-temperature schedule.
- `--oracle_routing`: use fixed identifier-matched routing.

Default arguments are in `bottleneck/configs/task_config.yaml` and are overridden by CLI arguments.


