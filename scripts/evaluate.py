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

        logger.info(f"Instance {instance_count: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")
        logger.info(f"             | GNN  nb nodes    {int(nb_nodes): >4d}  | GNN  time   {time: >6.2f} ")
        logger.info(f"             | Gain         {100*(1-nb_nodes/default_info['nb_nodes']): >8.2f}% | Gain      {100*(1-time/default_info['time']): >8.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description= "Evaluate Script")
    parser.add_argument("--eval", type=int, default = cfg.nb_eval_instances, help = f"Evaluate Sample Number. Default: {cfg.nb_eval_instances}")
    args = parser.parse_args()
    main(args.eval)