import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src import cfg,logger,get_device
from src.models import *
import ecole
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_scatter(gnn_vals, scip_vals, metric_name="Running Time (s)"):
    plt.figure(figsize=(6, 6))
    
    # 获取坐标轴范围
    min_val = min(min(gnn_vals), min(scip_vals)) * 0.5
    max_val = max(max(gnn_vals), max(scip_vals)) * 2
    
    # 绘制 y=x 参考线
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', alpha=0.6, label='Equal Performance')
    
    # 绘制散点
    plt.scatter(scip_vals, gnn_vals, alpha=0.7, edgecolors='w', s=50, label='Instances')
    
    plt.xlabel(f"SCIP {metric_name}")
    plt.ylabel(f"GNN {metric_name}")
    plt.title(f"Comparison of {metric_name}")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_performance_profile(gnn_vals, scip_vals):
    gnn_vals = np.array(gnn_vals)
    scip_vals = np.array(scip_vals)
    
    # 计算每个实例的最佳性能
    best_vals = np.minimum(gnn_vals, scip_vals)
    
    # 计算比率 r_p,s
    r_gnn = gnn_vals / best_vals
    r_scip = scip_vals / best_vals
    
    # 设置横轴 tau 的范围 (从 1 到 最大倍数)
    max_tau = max(np.max(r_gnn), np.max(r_scip))
    taus = np.linspace(1, min(max_tau, 10), 100) # 通常取到 5 或 10 倍即可
    
    # 计算每个 tau 对应的解决比例
    gnn_p = [np.mean(r_gnn <= tau) for tau in taus]
    scip_p = [np.mean(r_scip <= tau) for tau in taus]
    
    plt.figure(figsize=(8, 5))
    plt.step(taus, gnn_p, label='GNN (L2B)', where='post', linewidth=2)
    plt.step(taus, scip_p, label='SCIP (Strong Branching)', where='post', linewidth=2, linestyle='--')
    
    plt.xlabel(r'Performance Ratio $\tau$')
    plt.ylabel(r'Fraction of Instances $P(r_{p,s} \leq \tau)$')
    plt.title('Performance Profile')
    plt.grid(axis='both', alpha=0.3)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

def main(num_eval):
    if num_eval < 0:
        logger.error("Config Error")
        raise ValueError("Config Error")
    
    device = get_device()
    policy = GNNPolicy().to(device)
    state_dict = torch.load("src/trained_params.pkl", map_location= device)
    policy.load_state_dict(state_dict)
    policy.eval()

    instances = ecole.instance.SetCoverGenerator(n_rows=600, n_cols=1000, density=0.05)
    scip_parameters = vars(cfg.scip_parameters)
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nb_nodes": ecole.reward.NNodes().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
        },
        scip_params=scip_parameters,
    )
    default_env = ecole.environment.Configuring(
        observation_function=None,
        information_function={
            "nb_nodes": ecole.reward.NNodes().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
        },
        scip_params=scip_parameters,
    )

    scip_time = []
    gnn_time = []
    for instance_count, instance in zip(range(num_eval), instances):
        # Run the GNN brancher
        observation, action_set, _, done, info = env.reset(instance)
        while not done:
            with torch.no_grad():
                observation = (
                    torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                    torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                    torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                    torch.from_numpy(observation.variable_features.astype(np.float32)).to(device),
                )
                logits = policy(*observation)
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                observation, action_set, _, done, info = env.step(action)
        nb_nodes = info["nb_nodes"]
        time = info["time"]

        # Run SCIP's default brancher
        default_env.reset(instance)
        _, _, _, _, default_info = default_env.step({})

        print(f"Instance {instance_count: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")
        print(f"             | GNN  nb nodes    {int(nb_nodes): >4d}  | GNN  time   {time: >6.2f} ")
        print(f"             | Gain         {100*(1-nb_nodes/default_info['nb_nodes']): >8.2f}% | Gain      {100*(1-time/default_info['time']): >8.2f}%")
        gnn_time.append(time)
        scip_time.append(default_info['time'])

    gnn_time = np.array(gnn_time)
    scip_time = np.array(scip_time)
    plot_scatter(gnn_time,scip_time)
    plot_performance_profile(gnn_time,scip_time)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description= "Evaluate Script")
    parser.add_argument("--eval", type=int, default = cfg.nb_eval_instances, help = f"Evaluate Sample Number. Default: {cfg.nb_eval_instances}")
    args = parser.parse_args()
    main(args.eval)