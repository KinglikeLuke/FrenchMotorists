import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, r2_score


def unit_dev(y, mu):
    return 2 * (xlogy(y, y / mu) - (y - mu))


def scoring(actual, pred, sample_weight=None):
    return {'mae': mean_absolute_error(actual, pred, sample_weight=sample_weight),
            'mpe': mean_poisson_deviance(actual, pred, sample_weight=sample_weight),
            'r2': r2_score(actual, pred, sample_weight=sample_weight)}


def print_score(score):
    print(f"MAE: {score['mae']:.3f}, MPD: {score['mpe']:.3f}, R2: {score['r2']:.3f}")


def potential_gains(pred, test, freq):
    """
    Compare to the possible improvement given the noisy Possion process underlying. Probably wrong rn.
    :param pred:
    :param test:
    :param freq:
    :return:
    """
    y = test["ClaimNb"].to_numpy()
    mu_base = freq * test["Exposure"].to_numpy()
    mu_gbm = pred.to_numpy()

    # Achievable improvement if the GBM were the true model
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(50):
        y_sim = rng.poisson(mu_gbm)
        gains.append(unit_dev(y_sim, mu_base).mean() - unit_dev(y_sim, mu_gbm).mean())
    print("ceiling gain:", np.mean(gains))
    print("observed gain:", unit_dev(y, mu_base).mean() - unit_dev(y, mu_gbm).mean())


def compare(y, mu_a, mu_b):
    if isinstance(mu_a, pd.Series):
        mu_a = mu_a.to_numpy()
    if isinstance(mu_b, pd.Series):
        mu_b = mu_b.to_numpy()
    d = unit_dev(y, mu_a) - unit_dev(y, mu_b)   # positive means b is better
    std = d.std(ddof=1) / np.sqrt(len(d))
    sigfig = int(np.ceil(-np.log10(std)))
    return np.round(d.mean(), sigfig), np.round(std, sigfig)

def compare_risk_groups(pred, test):
    df = test[["ClaimNb", "Exposure", "DrivAge"]].assign(pred=pred, rate=pred / test["Exposure"])
    df = df.sort_values("DrivAge")
    df["bin"] = pd.cut(df["Exposure"].cumsum() / df["Exposure"].sum(), 10, labels=False)
    g = df.groupby("bin").agg(actual=("ClaimNb", "sum"), pred=("pred", "sum"), age=("DrivAge", "min"))
    g.index = g["age"]
    g = g.drop(columns=["age"])
    g["ratio"] = np.round(g["actual"] / g["pred"], 2)
    print(g)

def lorenz_gini(y, pred, exposure):
    rate_pred = pred / exposure
    order = np.argsort(rate_pred)
    cum_exp = np.cumsum(exposure[order]) / exposure.sum()
    cum_clm = np.cumsum(y[order]) / y.sum()
    cum_exp = np.r_[0, cum_exp]
    cum_clm = np.r_[0, cum_clm]
    return cum_exp, cum_clm, 1 - 2 * np.trapezoid(cum_clm, cum_exp)

