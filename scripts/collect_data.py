import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src import cfg,logger
from src.env_utils import make_env

def main(expert_probaility,max_samples):
    if not(0<expert_probaility<=1) or max_samples<=0:
        logger.error("Config Error")
        raise ValueError("Config Error")
    logger.info("Data Collection Begin")
    make_env(expert_probaility,max_samples)
    logger.info("Data Collection Finish")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description= "Data Collection Script")
    parser.add_argument("--expert", type=float, default = cfg.expert_probability, help = f"The proportion of strong branch (expert) used in the solution. The rest use the pseudocosts function. Default: {cfg.expert_probability}")
    parser.add_argument("--sample", type=int, default = cfg.max_samples, help = f"The number of saved expert samples. Default: {cfg.max_samples}")
    args = parser.parse_args()
    main(expert_probaility = args.expert,max_samples = args.sample)