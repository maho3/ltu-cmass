"""
A script to train ML models on existing suites of simulations.
"""
import time
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from os.path import join
import logging

from .preprocess import setup_optuna
from .train import (load_preprocessed_data,
                    run_training, run_training_with_precompression,
                    evaluate_posterior, plot_training_history)
from ..nbody.tools import parse_nbody_config
from ..utils import timing_decorator, clean_up
from .tools import split_experiments


def objective(trial, cfg: DictConfig,
              x_train, theta_train, x_val, theta_val, x_test, theta_test,
              hodprior, noiseprior, exp_path):

    trial_num = trial.number
    out_dir = join(exp_path, 'nets', f'net-{trial_num}')

    # Sample hyperparameters
    hyperprior = cfg.infer.hyperprior
    model = trial.suggest_categorical("model", hyperprior.model)
    hidden_features = trial.suggest_int(
        "hidden_features", *hyperprior.hidden_features, log=True)
    num_transforms = trial.suggest_int(
        "num_transforms", *hyperprior.num_transforms)
    fcn_width = trial.suggest_int("fcn_width", *hyperprior.fcn_width, log=True)
    fcn_depth = trial.suggest_int("fcn_depth", *hyperprior.fcn_depth)
    batch_size = int(2**trial.suggest_int(
        "log2_batch_size", *hyperprior.log2_batch_size))
    learning_rate = trial.suggest_float(
        "learning_rate", *hyperprior.learning_rate, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", *hyperprior.weight_decay, log=True)
    lr_patience = trial.suggest_int(
        "lr_patience", *hyperprior.lr_patience)
    lr_decay_factor = trial.suggest_float(
        "lr_decay_factor", *hyperprior.lr_decay_factor, log=True)
    mcfg = OmegaConf.create(dict(
        model=model,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        fcn_width=fcn_width,
        fcn_depth=fcn_depth,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_patience=lr_patience,
        lr_decay_factor=lr_decay_factor
    ))

    # run training
    start = time.time()
    train_fn = run_training_with_precompression if cfg.infer.precompress else run_training
    posterior, histories = train_fn(
        x_train, theta_train, x_val, theta_val, out_dir=out_dir,
        prior_name=cfg.infer.prior, mcfg=mcfg,
        batch_size=None,
        learning_rate=None,
        stop_after_epochs=cfg.infer.stop_after_epochs,
        val_frac=cfg.infer.val_frac,
        weight_decay=None,
        lr_decay_factor=None,
        lr_patience=None,
        backend=cfg.infer.backend, engine=cfg.infer.engine,
        device=cfg.infer.device,
        hodprior=hodprior, noiseprior=noiseprior, verbose=False
    )
    end = time.time()

    # Save the timing and metadata
    with open(join(out_dir, 'timing.txt'), 'w') as f:
        f.write(f'{end - start:.3f}')
    with open(join(out_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(OmegaConf.to_container(mcfg, resolve=True), f)

    # plot training history
    plot_training_history(histories, out_dir)

    # evaluate the posterior and save to file
    log_prob_test = evaluate_posterior(
        posterior, x_test, theta_test)
    with open(join(out_dir, 'log_prob_test.txt'), 'w') as f:
        f.write(f'{log_prob_test}\n')

    return log_prob_test


def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)

    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    for kmin in kmin_list:
        for kmax in kmax_list:
            logging.info(
                f'Running training for {name} with {kmin} <= k <= {kmax}')
            exp_path = join(model_path, f'kmin-{kmin}_kmax-{kmax}')

            # load training/test data
            (x_train, theta_train,
             x_val, theta_val,
             x_test, theta_test,
             hodprior, noiseprior) = load_preprocessed_data(exp_path)

            logging.info(
                f'Split: {len(x_train)} training, {len(x_val)} validation, '
                f'{len(x_test)} testing')

            # run hyperparameter optimization
            logging.info('Running hyperparameter optimization...')
            study = setup_optuna(
                exp_path, name, cfg.infer.n_startup_trials)
            study.optimize(
                lambda trial: objective(trial, cfg, x_train, theta_train,
                                        x_val, theta_val, x_test, theta_test,
                                        hodprior, noiseprior, exp_path),
                n_trials=cfg.infer.n_trials,
                n_jobs=1,
                timeout=60*60*4,  # 4 hours
                show_progress_bar=False,
                gc_after_trial=True
            )
            # NOTE: n_jobs>1 doesn't seem to speed things up much,
            # It seems processes are fighting for threads.
            # Instead, we parallelize via SLURM


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:

    cfg = parse_nbody_config(cfg)
    model_dir = join(cfg.meta.wdir, cfg.nbody.suite, cfg.sim, 'models')
    if cfg.infer.save_dir is not None:
        model_dir = cfg.infer.save_dir
    if cfg.infer.exp_index is not None:
        cfg.infer.experiments = split_experiments(cfg.infer.experiments)
        cfg.infer.experiments = [cfg.infer.experiments[cfg.infer.exp_index]]

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    for tracer in ['halo', 'galaxy',
                   'ngc_lightcone', 'sgc_lightcone', 'mtng_lightcone',
                   'simbig_lightcone']:
        if not getattr(cfg.infer, tracer):
            logging.info(f'Skipping {tracer} inference...')
            continue

        logging.info(f'Running {tracer} inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, tracer, '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path)


if __name__ == "__main__":
    main()
