import torch
import numpy as np
import os
import argparse
import logging
import warnings
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from .train import run_training

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Notes from Matt:
    This is a very good hack script to test convergence vs time. However, there
    are some caveats to keep in mind when interpreting the results:
        - This uses a fixed neural architecture.
        - This uses the sbi backend, as opposed to our lampe implementation.
        - This mixes simulations from the same LHID in train and val, which
          may lead to data leakage and overfitting.
    TODO: These things should be fixed before we do any final analysis. But,
    this is good enough for a first pass to get a sense of the convergence
    behavior of the flows.
"""



def train_and_save_nested_models(
    X_all,
    theta_all,
    sizes,
    B,
    out_dir,
    base_seed=0,
):
    '''
    train and save nested NPE models

    X_all: all training data
    theta_all: all training parameters
    sizes: list of training sizes
    B: number of nested sequences
    out_dir: output directory to save the models
    base_seed: base random seed for reproducibility

    '''

    # a set of hyperparameters I found useful in other projects.
    # Should be re-tuned
    mcfg = OmegaConf.create({
        'model': 'maf',
        'hidden_features': 52,
        'num_transforms': 9,
        'learning_rate': 1e-4,
        'batch_size': 128,
        'embedding_net': 'fcn',
        'fcn_depth': 0,
    })
    cfg = OmegaConf.create({
        'infer': {
            'prior': 'uniform',
            'device': device,
            'backend': 'sbi',
            'engine': 'NPE',
            'stop_after_epochs': 30,
            'val_frac': 0.2,
            'weight_decay': 0.0,
            'lr_decay_factor': 1.0,
            'lr_patience': 5,
        }
    })

    N = len(X_all)

    for b in range(B):
        rng = np.random.default_rng(base_seed + b)

        # one random permutation = one nested sequence
        perm = rng.permutation(N)

        X_perm = X_all[perm]
        theta_perm = theta_all[perm]

        for n in sizes:
            assert n <= N

            X_sub = X_perm[:n]
            theta_sub = theta_perm[:n]

            logging.info(f"[seq {b}] training N={n}")

            x_train, x_val, theta_train, theta_val = train_test_split(
                X_sub, theta_sub, test_size=cfg.infer.val_frac, random_state=42+b)

            model, _ = run_training(
                x_train=x_train,
                theta_train=theta_train,
                x_val=x_val,
                theta_val=theta_val,
                out_dir=None,
                cfg=cfg,
                mcfg=mcfg,
                verbose=False
            )

            # save model
            fname = f"npe_seq{b}_N{n}.pkl"
            fpath = os.path.join(out_dir, fname)

            torch.save(model, fpath)
            logging.info(f"  saved -> {fpath}")


def eval_nested_models(models_dir, X_test, theta_test, sizes, B):
    """
    evaluate the nested models by computing the average log-probability on the
        test set
    models_dir: directory where the nested models are saved
    X_test: test data
    theta_test: test parameters
    sizes: list of training sizes
    B: number of nested sequences

    returns:
    all_mis: array of shape (B, len(sizes)) containing the average
        log-probabilities
    """

    seq_idx_all = np.arange(B)

    """
    NOTE:
    Here I use `log_prob_batched` from the sbi package to speed up the
    computation. I also set `norm_posterior=False` to obtain *unnormalized* log
    probabilities.

    The reason is that, for flat priors, sbi estimates the normalization
    constant by Monte Carlo sampling a large number of points at each
    log-probability evaluation. This procedure is very time-consuming and can
    occasionally stall.

    When the flow is well trained, the unnormalized log probabilities should be
    close to the normalized ones. However, this approximation should be kept in
    mind when interpreting the results.
    """

    all_mis = []
    for seq_idx in seq_idx_all:
        mi_seq = []
        for N_train in sizes:
            model = torch.load(f'{models_dir}/npe_seq{seq_idx}_N{N_train}.pkl')
            log_p = model.posteriors[0].log_prob_batched(np.float32(
                theta_test[np.newaxis]), np.float32(X_test), norm_posterior=False)
            log_p = log_p.numpy()
            # remove -inf values
            log_p = log_p[~np.isinf(log_p)]

            mi_seq.append(np.mean(log_p))

        all_mis.append(mi_seq)

    return np.array(all_mis)  # shape (B, len(sizes))


def main(summary_path, output_path=None):
    ''''
    main function to run data sufficiency test on a given summary
    inputs:
    summary_path: path to the summary directory containing x_train, theta_train,
        x_val, theta_val, x_test, theta_test
    output_path: optional path to save the results (if None, saves to
        summary_path)

    '''

    logging.info('Device: %s', device)
    logging.info(
        f'Summary used: {os.path.basename(os.path.dirname(summary_path))}')

    x_train = np.load(f'{summary_path}/x_train.npy')
    theta_train = np.load(f'{summary_path}/theta_train.npy')
    x_val = np.load(f'{summary_path}/x_val.npy')
    theta_val = np.load(f'{summary_path}/theta_val.npy')

    # fixed test set for calculating log-probabilities
    x_test = np.load(f'{summary_path}/x_test.npy')
    theta_test = np.load(f'{summary_path}/theta_test.npy')

    X_all = np.concatenate([x_train, x_val], axis=0)
    theta_all = np.concatenate([theta_train, theta_val], axis=0)

    N = len(X_all)
    k = 5  # number of nested sizes

    # nested sizes. in this case, double the size each time
    sizes = np.array([N/2**(k-i-1) for i in range(k)]).astype(int)
    sizes[-1] = N  # ensure max size = N

    # number of independent nested sequences
    # B sequences will be trained, each with different random permutations and
    # increasing sizes
    B = 5

    # output directory
    # if not provided, save to summary_path
    # models are saved to nested_npe_models/ subdirectory
    if output_path is None:
        model_out_dir = os.path.join(summary_path, 'nested_npe_models/')
        out_dir = summary_path
    else:
        model_out_dir = os.path.join(output_path, 'nested_npe_models/')
        out_dir = output_path

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_out_dir, exist_ok=True)

    train_and_save_nested_models(
        X_all=X_all,
        theta_all=theta_all,
        sizes=sizes,
        B=B,
        out_dir=model_out_dir,
    )

    test_log_probs = eval_nested_models(
        models_dir=model_out_dir,
        X_test=x_test,
        theta_test=theta_test,
        sizes=sizes,
        B=B,
    )

    # save test_log_probs
    np.save(os.path.join(out_dir, 'test_log_probs.npy'), test_log_probs)


def parse_args():
    '''
    get command line arguments

    args:
    --data_path: base path containing summary subdirectories
    --summaries: list of summaries to run
    --output_path: optional output path (if not provided, output_path=None)

    returns: parsed arguments
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base path containing summary subdirectories",
    )
    parser.add_argument(
        "--summaries",
        type=str,
        nargs="+",
        required=True,
        help="List of summaries to run",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output path (if not provided, output_path=None)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for summary in args.summaries:
        # summary path, assume kmin-0.0_kmax-0.4
        input_path = os.path.join(args.data_path, summary, 'kmin-0.0_kmax-0.4')

        main(
            summary_path=input_path,
            output_path=args.output_path,
        )
