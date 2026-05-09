import ecole
import numpy as np
from src import cfg,logger,sample_dir
import gzip
import pickle
from pathlib import Path

class ExploreThenStrongBranch:
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        # Only expert_probability that strong branch variable selection will be used.
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)
        
def make_env(expert_probaility,max_samples):
    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    env = ecole.environment.Branching(
        observation_function=(
            ExploreThenStrongBranch(expert_probability=expert_probaility),
            ecole.observation.NodeBipartite(),
        ),
        # Convert from SimpleNamesapce to DictionaryS
        scip_params=vars(cfg.scip_parameters),
    )
    env.seed(cfg.seed)
    episode_counter, sample_counter = 0, 0
    sample_dir.mkdir(parents= True,exist_ok= True)

    # We will solve problems (run episodes) until we have saved enough samples
    while sample_counter < max_samples:
        episode_counter += 1
        observation, action_set, _, done, _ = env.reset(next(instances))
        while not done:
            (scores, scores_are_expert), node_observation = observation
            action = action_set[scores[action_set].argmax()]

            # Only save samples if they are coming from the expert (strong branching)
            if scores_are_expert and (sample_counter < max_samples):
                sample_counter += 1
                data = [node_observation, action, action_set, scores]
                filename = sample_dir / f"sample_{sample_counter}.pkl"

                with gzip.open(filename, "wb") as f:
                    pickle.dump(data, f)

            observation, action_set, _, done, _ = env.step(action)

        logger.info(f"Episode {episode_counter}, {sample_counter} samples are collected")
