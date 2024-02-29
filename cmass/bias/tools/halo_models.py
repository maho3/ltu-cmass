import numpy as np
from scipy.optimize import minimize


# Simon's galaxy biasing

class TruncatedPowerLaw:
    @staticmethod
    def _get_mean_ngal(rho, nmean, beta, epsilon_g, rho_g):
        rho[rho <= -1] = -1 + 1e-6  # Avoid log(0)
        d = 1 + rho
        x = np.power(np.abs(d / rho_g), -epsilon_g)
        ngal_mean = nmean * np.power(d, beta) * np.exp(-x)
        return ngal_mean

    @staticmethod
    def _loss(params, delta, count_field):
        nmean, beta, epsilon_g, rho_g = params
        ngal_mean = TruncatedPowerLaw._get_mean_ngal(
            delta, nmean, beta, epsilon_g, rho_g)

        loss = ngal_mean - count_field * np.log(ngal_mean)
        return loss.mean()

    def fit(self, delta, count_field, verbose=False):
        if np.sum(count_field) == 0:
            return np.array([0, 0, 0, 1])

        initial_guess = np.array([1e-2, 1., 1e-3, 1]).astype(np.float64)
        delta = delta.astype(np.float64)

        bounds = [(0, None)] * 4  # Bounds for positive values
        result = minimize(
            self._loss, initial_guess, args=(delta, count_field),
            method='Nelder-Mead',
            bounds=bounds,
            options={'disp': verbose}
        )
        popt = result.x
        if verbose:
            print(f"Power law bias fit params: {popt}")
        return popt

    def predict(self, delta, popt):
        ngal_mean = TruncatedPowerLaw._get_mean_ngal(delta, *popt)
        return ngal_mean

    def sample(self, delta, popt):
        ngal_mean = self.predict(delta, popt)
        return np.random.poisson(ngal_mean)
