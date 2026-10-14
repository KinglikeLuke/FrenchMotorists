import joblib

N_CORES = joblib.cpu_count(only_physical_cores=True)
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    r2_score
)
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

def scoring(actual, pred, sample_weight=None):
    return {'mae': mean_absolute_error(actual, pred, sample_weight=sample_weight),
            'mpe': mean_poisson_deviance(actual, pred, sample_weight=sample_weight),
            'r2': r2_score(actual, pred, sample_weight=sample_weight)}

def print_score(score):
    print(f"MAE: {score['mae']:.2f}, MPD: {score['mpe']:.2f}, R2: {score['r2']:.2f}")

def glm_model(data):
    """
    Returns a trained Poisson Generalized Linear model for the *claim frequency*
    :param data:
    :return:
    """
    model = smf.glm(
        formula=(
            "ClaimNb ~ C(VehPower) + VehAge + DrivAge + BonusMalus"
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
        ),
    )
    y = data["ClaimNb"] / data["Exposure"]
    tree_model.fit(data.drop(columns=["ClaimNb","Exposure"]), y, histgradientboostingregressor__sample_weight=data["Exposure"])
    return ClaimNumberModel(tree_model)

def main():
    data = pd.read_csv('freMTPL2freq.csv', sep=',', index_col=0)
    data["ClaimNb"] = data["ClaimNb"].clip(upper=4)
    data["Exposure"] = data["Exposure"].clip(upper=1)
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    glm = glm_model(train)
    gbm = gbm_model(train)

    print("Target mean:", test["ClaimNb"].mean())

    gbm_pred = gbm.predict(test.drop(columns=["ClaimNb"]))
    glm_pred = glm.predict(test.drop(columns=["ClaimNb"]))

    print("GBM mean:", gbm_pred.mean())
    print_score(scoring(test["ClaimNb"], gbm_pred))
    compare_risk_groups(gbm_pred, test)

    print("GLM mean:", glm_pred.mean())
    print_score(scoring(test["ClaimNb"], glm_pred))
    compare_risk_groups(glm_pred, test)

def compare_risk_groups(pred_count, test):
    comparison = test[["ClaimNb", "Exposure"]].copy()
    comparison["predicted_claims"] = pred_count
    comparison["risk_group"] = pd.qcut(pred_count / test["Exposure"], q=10, duplicates="drop")
    comparison["rate"] = pred_count / test["Exposure"]
    print(comparison.groupby("risk_group", observed=True).agg(
        actual_claims=("ClaimNb", "sum"),
        predicted_claims=("predicted_claims", "sum"),
        exposure=("Exposure", "sum")))

if __name__ == '__main__':
    main()
