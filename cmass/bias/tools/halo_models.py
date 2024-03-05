import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa, must go before jax
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'  # noqa, must go before jax
os.environ['JAX_ENABLE_X64'] = '1'  # noqa

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.optimize import minimize
import logging


class PowerLaw:
    @staticmethod
    @jax.jit
    def _get_mean_ngal(rho, params):
        lognmean, beta = params
        logd = jnp.log(1 + rho)
        logngal_mean = lognmean + beta * logd
        return jnp.exp(logngal_mean)

    def _loss(self, params, delta, count_field, scale=1):
        ngal_mean = self._get_mean_ngal(delta, params)

        loss = ngal_mean - count_field * jnp.log(ngal_mean)
        return jnp.mean(loss)/scale

    @staticmethod
    def _post(params):
        return params

    def get_initial_guess(self, count_field):
        hmean = count_field.mean()
        initial_guess = jnp.array([hmean, 10]).astype(jnp.float64)
        initial_guess = jnp.log(initial_guess)
        initial_guess += 1e-2 * np.random.randn(2)
        return initial_guess

    @staticmethod
    def get_default():
        return np.array([0, 1])

    def fit(self, delta, count_field, verbose=False, attempts=5):
        for i in range(attempts):
            initial_guess = self.get_initial_guess(count_field)
            if jnp.sum(count_field) == 0:
                return self.get_default(), None

            delta = delta.astype(jnp.float64)
            if i <= attempts//2:  # initial loss is not good sometimes
                scale = 10**i
            else:
                scale = 10**(attempts//2 - i)

            result = minimize(
                self._loss, initial_guess, args=(delta, count_field, scale),
                method='BFGS', tol=1e-4
            )
            params = result.x
            if result.success:
                break
            else:
                logging.warning(
                    f"Fit failed with status: {result.status}. "
                    f"Retrying {i+1}/{attempts}.")
        params = self._post(params)
        if verbose:
            logging.info(f"Power law bias fit params: {params}")
        return params, result

    def predict(self, delta, params):
        ngal_mean = self._get_mean_ngal(delta, params)
        return np.array(ngal_mean)

    def sample(self, delta, params):
        ngal_mean = self.predict(delta, params)
        return np.random.poisson(ngal_mean)


class TruncatedPowerLaw(PowerLaw):
    @staticmethod
    @jax.jit
    def _get_mean_ngal(rho, params):
        nmean, beta, epsilon_g, rho_g = jnp.exp(params)
        d = 1 + rho
        x = jnp.power(d / rho_g, -epsilon_g)
        ngal_mean = nmean * jnp.power(d, beta) * jnp.exp(-x)
        return ngal_mean + 1e-6

    def get_initial_guess(self, count_field):
        initial_guess = jnp.array([np.exp(-2), 1, 3, 3]).astype(jnp.float64)
        initial_guess = jnp.log(initial_guess)
        inital_guess = [-2, 0.01, 1.1, 1.1]
        initial_guess = jnp.array(inital_guess).astype(jnp.float64)
        initial_guess += 1e-6 * np.random.randn(4)
        return initial_guess

    @staticmethod
    def get_default():
        return np.array([-np.inf, 1, 1, 1])


class LogTPL(TruncatedPowerLaw):
    def get_initial_guess(self, count_field):
        logF = count_field.mean()
        initial_guess = jnp.array(
            [jnp.log(logF), 1, 1e-3, 1e-3]).astype(jnp.float64)
        initial_guess += 1e-2 * np.random.randn(4)
        return initial_guess

    @staticmethod
    @jax.jit
    def _get_mean_ngal(rho, params):
        # rho = jnp.where(rho <= -1, -1 + 1e-6, rho)  # Avoid log(0)
        logf, alpha, epsilon, Aexp = params
        Am = jnp.log(1 + rho)
        Ah = logf + alpha * Am + jnp.exp(-epsilon*(Am - Aexp))
        return jnp.exp(Ah)+1e-6

    @staticmethod
    def _post(params):
        return params

    @staticmethod
    def get_default():
        return np.array([0, 1, 1e-3, 1e-3])
