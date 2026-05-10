import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src import cfg,logger,sample_dir,get_device
from src.dataset import *
from src.models import *
import torch.nn.functional as F

def main(lr,num_epoch,patience):
    if lr<=0 or num_epoch<=0:
        logger.error("Config Error")
        raise ValueError("Config Error")
    
    logger.info("Training Begin")
    sample_files = [str(path) for path in sample_dir.glob("sample_*.pkl")]
    train_files = sample_files[: int(0.8 * len(sample_files))]
    valid_files = sample_files[int(0.8 * len(sample_files)) :]
    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=128, shuffle=False)

    policy = GNNPolicy().to(get_device())
    observation = train_data[0].to(get_device())

    logits = policy(
        observation.constraint_features,
        observation.edge_index,
        observation.edge_attr,
        observation.variable_features,
    )
    action_distribution = F.softmax(logits[observation.candidates], dim=-1)
    print(action_distribution)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    best_loss = float('inf')
    counter = 0
    params_path = "src/trained_params.pkl"

    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}")

        train_loss, train_acc = process(policy, train_loader, get_device(), optimizer)
        valid_loss, valid_acc = process(policy, valid_loader, get_device(), None)

        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(policy.state_dict(),params_path)
            logger.info(f"Model improved. Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f} Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping triggered! Training finished. Total epoch: {epoch}")
                break
    logger.info("Training End")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description= "Training Script")
    parser.add_argument("--lr", type=float, default = cfg.learning_rate, help = f"Learning Rate. Default: {cfg.learning_rate}")
    parser.add_argument("--epoch", type=int, default = cfg.nb_epochs, help = f"The number of epoch. Default: {cfg.nb_epochs}")
    parser.add_argument("--patience", type=int, default = cfg.early_stop_patience, help = f"Early stop patience round. Default: {cfg.early_stop_patience}")
    args = parser.parse_args()
    main(lr = args.lr,num_epoch= args.epoch,patience = args.patience)