import pandas as pd
from xgboost import XGBClassifier


def make_credit_features(borrowers: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the feature set small and interpretable for the paper.
    """
    return borrowers[["income", "thin_file", "group"]].copy()


def train_credit_model(X: pd.DataFrame, y: pd.Series, params: dict) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def predict_default_prob(model: XGBClassifier, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(X)[:, 1], name="p_default")
