
import numpy as np
from omegaconf import DictConfig, OmegaConf


def sample_hyperparameters_optuna(
    trial: "optuna.trial.Trial",
    hyperprior: DictConfig,
    embedding_net: str
) -> DictConfig:
    """Sample hyperparameters from the hyperprior for Optuna and return a model config."""
    mcfg = {"embedding_net": embedding_net}

    # sample shared parameters
    mcfg['model'] = trial.suggest_categorical("model", hyperprior.shared.model)
    mcfg['hidden_features'] = trial.suggest_int(
        "hidden_features", *hyperprior.shared.hidden_features, log=True)
    mcfg['num_transforms'] = trial.suggest_int(
        "num_transforms", *hyperprior.shared.num_transforms)
    mcfg['batch_size'] = int(2**trial.suggest_int("log2_batch_size",
                                                 *hyperprior.shared.log2_batch_size))
    mcfg['learning_rate'] = trial.suggest_float(
        "learning_rate", *hyperprior.shared.learning_rate, log=True)
    mcfg['weight_decay'] = trial.suggest_float(
        "weight_decay", *hyperprior.shared.weight_decay, log=True)
    mcfg['lr_patience'] = trial.suggest_int(
        "lr_patience", *hyperprior.shared.lr_patience)
    mcfg['lr_decay_factor'] = trial.suggest_float(
        "lr_decay_factor", *hyperprior.shared.lr_decay_factor, log=True)

    # sample embedding-specific parameters
    if embedding_net == 'fcn':
        mcfg['fcn_depth'] = trial.suggest_int('fcn_depth',
                                              *hyperprior.fcn.fcn_depth)
        mcfg['fcn_width'] = trial.suggest_int('fcn_width',
                                              *hyperprior.fcn.fcn_width, log=True)
    elif embedding_net == 'cnn':
        mcfg['cnn_depth'] = trial.suggest_int('cnn_depth',
                                              *hyperprior.cnn.cnn_depth)
        mcfg['out_channels'] = trial.suggest_int(
            'out_channels', *hyperprior.cnn.out_channels, log=True)
        mcfg['kernel_size'] = trial.suggest_int('kernel_size',
                                                *hyperprior.cnn.kernel_size)

    return OmegaConf.create(mcfg)


def sample_hyperparameters_randomly(
    hyperprior: DictConfig,
    embedding_net: str,
    seed: int = None
) -> DictConfig:
    """Randomly sample hyperparameters from the hyperprior and return a model config."""
    mcfg = {"embedding_net": embedding_net}

    if seed is not None:
        np.random.seed(seed)

    # sample shared parameters
    mcfg['model'] = np.random.choice(hyperprior.shared.model)
    mcfg['hidden_features'] = int(np.exp(np.random.uniform(
        np.log(hyperprior.shared.hidden_features[0]),
        np.log(hyperprior.shared.hidden_features[1]))))
    mcfg['num_transforms'] = np.random.randint(
        hyperprior.shared.num_transforms[0], hyperprior.shared.num_transforms[1] + 1)
    mcfg['batch_size'] = int(2**np.random.randint(
        hyperprior.shared.log2_batch_size[0], hyperprior.shared.log2_batch_size[1] + 1))
    mcfg['learning_rate'] = np.exp(np.random.uniform(
        np.log(hyperprior.shared.learning_rate[0]),
        np.log(hyperprior.shared.learning_rate[1])))
    mcfg['weight_decay'] = np.exp(np.random.uniform(
        np.log(hyperprior.shared.weight_decay[0]),
        np.log(hyperprior.shared.weight_decay[1])))
    mcfg['lr_patience'] = np.random.randint(
        hyperprior.shared.lr_patience[0], hyperprior.shared.lr_patience[1] + 1)
    mcfg['lr_decay_factor'] = np.exp(np.random.uniform(
        np.log(hyperprior.shared.lr_decay_factor[0]),
        np.log(hyperprior.shared.lr_decay_factor[1])))

    # sample embedding-specific parameters
    if embedding_net == 'fcn':
        mcfg['fcn_depth'] = np.random.randint(
            hyperprior.fcn.fcn_depth[0], hyperprior.fcn.fcn_depth[1] + 1)
        mcfg['fcn_width'] = int(np.exp(np.random.uniform(
            np.log(hyperprior.fcn.fcn_width[0]),
            np.log(hyperprior.fcn.fcn_width[1]))))
    elif embedding_net == 'cnn':
        mcfg['cnn_depth'] = np.random.randint(
            hyperprior.cnn.cnn_depth[0], hyperprior.cnn.cnn_depth[1] + 1)
        mcfg['out_channels'] = int(np.exp(np.random.uniform(
            np.log(hyperprior.cnn.out_channels[0]),
            np.log(hyperprior.cnn.out_channels[1]))))
        mcfg['kernel_size'] = np.random.randint(
            hyperprior.cnn.kernel_size[0], hyperprior.cnn.kernel_size[1] + 1)
        
    # typecasting for OmegaConf
    for k, v in mcfg.items():
        if isinstance(v, np.int64):
            mcfg[k] = int(v)
        elif isinstance(v, np.float64):
            mcfg[k] = float(v)
        elif isinstance(v, np.str_):
            mcfg[k] = str(v)

    return OmegaConf.create(mcfg)
