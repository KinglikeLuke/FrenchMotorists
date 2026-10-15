import joblib

from evaluation import scoring, print_score, compare, compare_risk_groups, lorenz_gini

N_CORES = joblib.cpu_count(only_physical_cores=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

class ClaimNumberModel:
    def __init__(self, fitted_model):
        self.model = fitted_model

    def predict(self, X):
        exposure = X["Exposure"]

        return self.model.predict(
            X
        ) * exposure

def glm_model(data):
    """
    Returns a trained Poisson Generalized Linear model for the *claim frequency*
    :param data:
    :return: smf.glm()
    """
    model = smf.glm(
        formula=(
            "ClaimNb ~ C(VehPower) + VehAge + DrivAge + np.square(DrivAge) + BonusMalus"
            " + C(VehBrand) + C(VehGas) + C(Area) + C(Region)"
            " + np.log(Density)"
        ),
        data=data,
        family=sm.families.Poisson(),
        offset=np.log(data["Exposure"])
    ).fit()
    return ClaimNumberModel(model)

def gbm_model(data):
    """
    Returns a trained Gradient Boosted Tree model for the *claim frequency*
    :param data:
    :return:
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
             ["VehBrand", "VehGas", "Area", "Region"]),
        ],
        remainder="passthrough",
    )

    tree_model = make_pipeline(
        preprocessor,
        HistGradientBoostingRegressor(
            loss="poisson",
            categorical_features=[0,1,2,3],
            learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, early_stopping=True, l2_regularization=True
        ),
    )
    y = data["ClaimNb"] / data["Exposure"]
    tree_model.fit(data.drop(columns=["ClaimNb","Exposure"]), y, histgradientboostingregressor__sample_weight=data["Exposure"])
    return ClaimNumberModel(tree_model)


def main():
    data = pd.read_csv('freMTPL2freq.csv', sep=',', index_col=0)
    data["ClaimNb"] = data["ClaimNb"].clip(upper=4)
    data["Exposure"] = data["Exposure"].clip(upper=1)
    train, test = train_test_split(data, test_size=0.2, random_state=41)

    glm = glm_model(train)
    gbm = gbm_model(train)

    freq = train["ClaimNb"].sum() / train["Exposure"].sum()

    gbm_pred = gbm.predict(test.drop(columns=["ClaimNb"]))
    glm_pred = glm.predict(test.drop(columns=["ClaimNb"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    evaluate_model(freq, test["Exposure"] * freq, test, "baseline", ax)
    evaluate_model(freq, gbm_pred, test, "GBM", ax)
    evaluate_model(freq, glm_pred, test, "GLM", ax)

    print("GBM vs GLM", compare(test["ClaimNb"], gbm_pred, glm_pred))
    ax.set_title("Cumulative claims sorted by model")
    ax.legend()
    plt.show()

def evaluate_model(freq, pred, test, name, ax=None):
    print(f"{name} mean:", pred.mean())
    print_score(scoring(test["ClaimNb"], pred))
    if name != "baseline":
        print(f"{name} vs baseline", compare(test["ClaimNb"], freq * test["Exposure"], pred))
    compare_risk_groups(pred, test)
    cum_exposure, cum_claim, gini = lorenz_gini(test["ClaimNb"].to_numpy(), pred, test["Exposure"].to_numpy())
    print(f"{name} Gini:", gini)
    if ax:
        ax.plot(cum_exposure, cum_claim, label=name)


if __name__ == '__main__':
    main()
