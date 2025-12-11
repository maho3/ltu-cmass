"""
A script to train ML models on existing suites of simulations.
"""
import time
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from os.path import join
import logging

from sklearn.model_selection import KFold

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

def objective_cval(trial, cfg: DictConfig,
              x_train, theta_train, x_val, theta_val, x_test, theta_test,
              hodprior, noiseprior, exp_path, n_splits_in):

    trial_num = trial.number
    out_dir = join(exp_path, 'nets', f'net-{trial_num}')

    # Instantiate cross_validators
    # In ILI, we train with validation and train sets, and perform optuna based on the test set
    # We thus need nested cross validation
    cval_in = KFold(n_splits=n_splits_in, shuffle=False)

    # Aggregate again the splits, to implement the nested cross-val
    x_cvin, theta_cvin = np.vstack((x_train, x_val)), np.vstack((theta_train, theta_val))

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

    # Handle the cross val splits: this is a cross-validation inside the hyperparameter tuning/inner loop !
    # Careful about this confusion
    start = time.time()
    K=0
    scores_in = np.zeros((n_splits_in,))
    for fold_in, (train_idx, val_idx) in enumerate(cval_in.split(x_cvin,y = theta_cvin)):
        
        x_train_in = x_cvin[train_idx]
        theta_train_in = theta_cvin[train_ids]
        
        x_val_in= x_cvin[val_idx]
        theta_val_in = theta_cvin[val_idx]

        # run training
        train_fn = run_training_with_precompression if cfg.infer.precompress else run_training
        
        posterior, histories = train_fn(
            x_train_in, theta_train_in, x_val, theta_val_in, out_dir=out_dir,
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

        # Evaluate inner loop score
        scores_in[K] = evaluate_posterior(posterior, x_test, theta_test)
        K+=1

    end = time.time()

    # Save the timing and metadata
    with open(join(out_dir, 'timing.txt'), 'w') as f:
        f.write(f'{end - start:.3f}')
    with open(join(out_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(OmegaConf.to_container(mcfg, resolve=True), f)

    with open(join(out_dir, 'log_prob_test.txt'), 'w') as f:
        f.write(f'{scores_in.mean()}\n')

    return scores_in.mean()

# If the inner loop cross validation is enabled, the val_frac configuration is overwritten by
# the number of inner splits we ask for
def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)

    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    objective_fn = objective_cval if cfg.infer.cval_in else objective
    args_dict = {"n_splits":cfg.infer.n_splits} if cfg.infer.cval_in else {}
    
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
                lambda trial: objective_fn(trial, cfg, x_train, theta_train,
                                        x_val, theta_val, x_test, theta_test,
                                        hodprior, noiseprior, exp_path, **args_dict),
                n_trials=cfg.infer.n_trials,
                n_jobs=1,
                timeout=60*60*4,  # 4 hours
                show_progress_bar=False,
                gc_after_trial=True
            )
            # NOTE: n_jobs>1 doesn't seem to speed things up much,
            # It seems processes are fighting for threads.
            # Instead, we parallelize via SLURM

# If outer loop cross-val is enabled, we force the inner loop cross-val to avoid errors with val_frac
# and test_frac, since test_frac is overwritten by the number of splits in the outer loop
def run_experiment_cval(exp, cfg, model_path, n_splits_out):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)

    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    # Outer loop cross-val split
    cval_out = KFold(n_splits=n_splits_out, shuffle=False)
    
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

            total_ids = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
            val_frac = x_val.shape[0]/total_ids
            n_val = x_val.shape[0]
            test_frac = x_test.shape[0]/total_ids
            n_train = x_train.shape[0]
            
            x_all, theta_all = np.vstack((x_train, x_val), x_test), np.vstack((theta_train, theta_val, theta_test))

            K=0
            # Outer loop cross-validation
            for fold_out, (train_val_idx, test_idx) in enumerate(cval_out.split(x_all,y = theta_all)):

                # Each split is a separate optuna study --> create folder
                split_path = join(exp_path, "split%i"%K)
                os.makedirs(split_path, exist_ok=True)
        
                x_train_val = x_all[train_val_idx]
                theta_train_val = theta_all[train_val_ids]

                # To avoid writing duplicate code for objective_cval (inner loop), I rearrange val and train ensembles
                # which will be anyway split again according to the inner loop splitting parameter
                n_vt = half = int(x_train_val.shape[0]/2)
                x_train = x_train_val[:n_vt,:]
                x_val = x_train_val[n_vt:,:]
                theta_train = theta_train_val[:n_vt,:]
                theta_val = theta_train_val[n_vt:,:]

                # test set of the outer loop split
                x_test= x_all[test_idx]
                theta_test = theta_all[test_idx]

                logging.info(
                    f'Split: {n_train} training, {n_val} validation, '
                    f'{len(x_test)} testing')
    
                # run hyperparameter optimization
                logging.info('Running hyperparameter optimization...')
                study = setup_optuna(
                    exp_path, name, cfg.infer.n_startup_trials)
                study.optimize(
                    lambda trial: objective_cval(trial, cfg, x_train, theta_train,
                                            x_val, theta_val, x_test, theta_test,
                                            hodprior, noiseprior, split_path, cfg.infer.n_splits_in),
                    n_trials=cfg.infer.n_trials,
                    n_jobs=1,
                    timeout=60*60*4,  # 4 hours
                    show_progress_bar=False,
                    gc_after_trial=True
                )

                # TODO: when running validation metrics, with validation modules or not
                # , I must add routines to handle the ensemble of optuna studies if outer loop
                # cross-val was performed (ie different test sets)

@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:

    # New for outer loop of nested cross validation
    run_experiment_fn = run_experiment_cval if cfg.infer.cval_out else run_experiment
    args_dict = {"n_splits_out":cfg.infer.n_splits_out} if cfg.infer.cval_in else {}
    
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
            #save_path = join(model_dir, tracer, '+'.join(exp.summary))
            save_path = join(model_dir, tracer, cfg.sim, '+'.join(exp.summary)) # sim to compare pinocchio, fastpm...
            run_experiment_fn(exp, cfg, save_path, **args_dict)


if __name__ == "__main__":
    main()
