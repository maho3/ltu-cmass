import numpy as np
from omegaconf import DictConfig, OmegaConf


def _get_or_sample_optuna(trial, name, value, sample_func, **kwargs):
    """Helper to sample a value for optuna or return it if it's fixed."""
    is_list_like = hasattr(value, '__len__') and not isinstance(value, str)
    if not is_list_like:
        return value

    if sample_func == 'categorical':
        return trial.suggest_categorical(name, value)
    elif sample_func == 'int':
        return trial.suggest_int(name, *value, **kwargs)
    elif sample_func == 'float':
        return trial.suggest_float(name, *value, **kwargs)
    raise ValueError(f"Unknown sample function: {sample_func}")


def _get_or_sample_random(value, sample_logic):
    """Helper to sample a value randomly or return it if it's fixed."""
    is_list_like = hasattr(value, '__len__') and not isinstance(value, str)
    if not is_list_like:
        return value

    if sample_logic == 'choice':
        return np.random.choice(value)
    elif sample_logic == 'randint':
        return np.random.randint(value[0], value[1] + 1)
    elif sample_logic == 'loguniform':
        return np.exp(np.random.uniform(np.log(value[0]), np.log(value[1])))
    elif sample_logic == 'uniform':
        return np.random.uniform(value[0], value[1])
    raise ValueError(f"Unknown sample logic: {sample_logic}")


def sample_hyperparameters_optuna(
    trial: "optuna.trial.Trial",
    hyperprior: DictConfig,
    embedding_net: str
) -> DictConfig:
    """Sample hyperparameters from the hyperprior for Optuna and return a model config."""
    mcfg = {"embedding_net": embedding_net}

    # sample shared parameters
    hp = hyperprior.shared
    mcfg['model'] = _get_or_sample_optuna(trial, "model", hp.model, 'categorical')
    mcfg['hidden_features'] = _get_or_sample_optuna(
        trial, "hidden_features", hp.hidden_features, 'int', log=True)
    mcfg['num_transforms'] = _get_or_sample_optuna(
        trial, "num_transforms", hp.num_transforms, 'int')
    log2_batch_size = _get_or_sample_optuna(
        trial, "log2_batch_size", hp.log2_batch_size, 'int')
    mcfg['batch_size'] = int(2**log2_batch_size)
    mcfg['learning_rate'] = _get_or_sample_optuna(
        trial, "learning_rate", hp.learning_rate, 'float', log=True)
    mcfg['weight_decay'] = _get_or_sample_optuna(
        trial, "weight_decay", hp.weight_decay, 'float', log=True)
    mcfg['lr_patience'] = _get_or_sample_optuna(
        trial, "lr_patience", hp.lr_patience, 'int')
    mcfg['lr_decay_factor'] = _get_or_sample_optuna(
        trial, "lr_decay_factor", hp.lr_decay_factor, 'float', log=True)
    mcfg['early_stopping'] = _get_or_sample_optuna(
        trial, "early_stopping", hp.early_stopping, 'categorical')
    mcfg['noise_percent'] = _get_or_sample_optuna(
        trial, "noise_percent", hp.noise_percent, 'float', log=True)
    mcfg['lr_scheduler'] = _get_or_sample_optuna(
        trial, "lr_scheduler", hp.lr_scheduler, 'categorical')
    mcfg['max_epochs'] = _get_or_sample_optuna(
        trial, "max_epochs", hp.max_epochs, 'int', log=True)
    mcfg['dropout'] = _get_or_sample_optuna(
        trial, "dropout", hp.dropout, 'float')

    # sample embedding-specific parameters
    hp_emb = hyperprior[embedding_net]
    if embedding_net == 'fcn':
        mcfg['fcn_depth'] = _get_or_sample_optuna(
            trial, 'fcn_depth', hp_emb.fcn_depth, 'int')
        mcfg['fcn_width'] = _get_or_sample_optuna(
            trial, 'fcn_width', hp_emb.fcn_width, 'int', log=True)
    elif embedding_net == 'cnn':
        mcfg['cnn_depth'] = _get_or_sample_optuna(
            trial, 'cnn_depth', hp_emb.cnn_depth, 'int')
        mcfg['out_channels'] = _get_or_sample_optuna(
            trial, 'out_channels', hp_emb.out_channels, 'int', log=True)
        mcfg['kernel_size'] = _get_or_sample_optuna(
            trial, 'kernel_size', hp_emb.kernel_size, 'int')
    elif embedding_net == 'mhe':
        mcfg['hidden_depth'] = _get_or_sample_optuna(
            trial, 'hidden_depth', hp_emb.hidden_depth, 'int')
        mcfg['hidden_width'] = _get_or_sample_optuna(
            trial, 'hidden_width', hp_emb.hidden_width, 'int', log=True)
        mcfg['out_features'] = _get_or_sample_optuna(
            trial, 'out_features', hp_emb.out_features, 'int', log=True)
    elif embedding_net in ['fun', 'mhf']:
        mcfg['hidden_depth'] = _get_or_sample_optuna(
            trial, 'hidden_depth', hp_emb.hidden_depth, 'int')
        mcfg['out_features'] = _get_or_sample_optuna(
            trial, 'out_features', hp_emb.out_features, 'int', log=True)
        mcfg['linear_dim'] = _get_or_sample_optuna(
            trial, 'linear_dim', hp_emb.linear_dim, 'int', log=True)
    else:
        raise ValueError(f"Unknown embedding net: {embedding_net}")

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
    hp = hyperprior.shared
    mcfg['model'] = _get_or_sample_random(hp.model, 'choice')
    mcfg['hidden_features'] = int(
        _get_or_sample_random(hp.hidden_features, 'loguniform'))
    mcfg['num_transforms'] = _get_or_sample_random(hp.num_transforms, 'randint')
    mcfg['batch_size'] = int(
        2**_get_or_sample_random(hp.log2_batch_size, 'randint'))
    mcfg['learning_rate'] = _get_or_sample_random(
        hp.learning_rate, 'loguniform')
    mcfg['weight_decay'] = _get_or_sample_random(hp.weight_decay, 'loguniform')
    mcfg['lr_patience'] = _get_or_sample_random(hp.lr_patience, 'randint')
    mcfg['lr_decay_factor'] = _get_or_sample_random(
        hp.lr_decay_factor, 'loguniform')
    mcfg['early_stopping'] = _get_or_sample_random(
        hp.early_stopping, 'choice')
    mcfg['noise_percent'] = _get_or_sample_random(
        hp.noise_percent, 'loguniform')
    mcfg['lr_scheduler'] = _get_or_sample_random(
        hp.lr_scheduler, 'choice')
    mcfg['max_epochs'] = int(_get_or_sample_random(
        hp.max_epochs, 'loguniform'))
    mcfg['dropout'] = _get_or_sample_random(
        hp.dropout, 'uniform')

    # sample embedding-specific parameters
    hp_emb = hyperprior[embedding_net]
    if embedding_net == 'fcn':
        mcfg['fcn_depth'] = _get_or_sample_random(hp_emb.fcn_depth, 'randint')
        mcfg['fcn_width'] = int(
            _get_or_sample_random(hp_emb.fcn_width, 'loguniform'))
    elif embedding_net == 'cnn':
        mcfg['cnn_depth'] = _get_or_sample_random(hp_emb.cnn_depth, 'randint')
        mcfg['out_channels'] = int(
            _get_or_sample_random(hp_emb.out_channels, 'loguniform'))
        mcfg['kernel_size'] = _get_or_sample_random(
            hp_emb.kernel_size, 'randint')
    elif embedding_net == 'mhe':
        mcfg['hidden_depth'] = _get_or_sample_random(
            hp_emb.hidden_depth, 'randint')
        mcfg['hidden_width'] = int(
            _get_or_sample_random(hp_emb.hidden_width, 'loguniform'))
        mcfg['out_features'] = int(
            _get_or_sample_random(hp_emb.out_features, 'loguniform'))
    elif embedding_net in ['fun', 'mhf']:
        mcfg['hidden_depth'] = _get_or_sample_random(
            hp_emb.hidden_depth, 'randint')
        mcfg['out_features'] = int(
            _get_or_sample_random(hp_emb.out_features, 'loguniform'))
        linear_dim_value = _get_or_sample_random(hp_emb.linear_dim, 'loguniform')
        mcfg['linear_dim'] = int(linear_dim_value) if linear_dim_value is not None else None
    else:
        raise ValueError(f"Unknown embedding net: {embedding_net}")

    # typecasting for OmegaConf
    for k, v in mcfg.items():
        if isinstance(v, np.int64):
            mcfg[k] = int(v)
        elif isinstance(v, np.float64):
            mcfg[k] = float(v)
        elif isinstance(v, np.str_):
            mcfg[k] = str(v)

    return OmegaConf.create(mcfg)
