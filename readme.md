# Learn-to-Branch (L2B) Baseline

A GNN Learn-to-Branch baseline implemented using [ecole](https://github.com/ds4dm/ecole/tree/master). Thanks to the ecole team for creating such an excellent L2O library and example.

For my URIS Project only.

## Environment

[![Conda-Forge platforms](https://img.shields.io/conda/pn/conda-forge/ecole?logo=conda-forge)](https://anaconda.org/conda-forge)

```bash
conda env create -f environment.yml
conda activate ecole_env
```

## Run
This project is divided into three parts: Data Collection, Training, and Evaluation.

### Configuration
The project's hyperparameters are stored in `config/config.yaml`. For details, see the command of each part below.

### Data Collection
Use the MILP Generator `SetCoverGenerator` implemented in the ecole library to generate a Set Cover problem and save to the `samples` folder.

```bash
python scripts/collect_data.py --expert 0.05 --sample 1000
```

- `--expert`
    - Default: `config/config.yaml: expert_probability`(same file as bellow) = 0.05
    - Type: `float (0,1)`
    - Description: The proportion of strong branch (expert) used in the solution. The rest use the pseudocosts function. 

- `--sample`
    - Default: `max_samples` = 1000
    - Type: `int`
    - Description: The number of saved expert samples.

### Training
Train the GNN using the saved samples, and save the model parameters to `src/trained_params.pkl`.

```bash
python scripts/train.py --lr 0.001 --epoch 50
```

- `--lr`
    - Default: `learning_rate` = 0.001
    - Type: `float`
    - Description: Learning rate of GNN training
- `--epoch`
    - Default: `nb_epochs` = 50
    - Type: `int`
    - Description: Number of epochs. The epoch should not be set overly large. This baseline version does not include early stopping mechanism.
- `--patience`
    - Default: `early_stop_patience` = 10
    - Type: `int`
    - Description: Early stop patience round.

### Evaluation
Retrieve model parameters from `src/trained_params.pkl` and load the GNN model. Generate a new MILP problem using `SetCoverGenerator`. Compare the performance of SCIP’s traditional solver and the GNN solver.

```bash
python scripts/evaluate.py --eval 20
```

- `--eval`
    - Default: `nb_eval_instances` = 20
    - Type: `int`
    - Description: Evaluate Sample Number.
