import pandas as pd


def compute_joint_score(p_default: pd.Series, fraud_score: pd.Series, lambda_fraud: float) -> pd.Series:
    """
    Joint score: lower is better.
    Fraud risk is integrated into decisioning (not a hard filter).
    """
    return (p_default - lambda_fraud * fraud_score).rename("joint_score")


def approve(
    joint_score: pd.Series,
    threshold: float,
    thin_file: pd.Series,
    thin_file_relaxation: float,
) -> pd.Series:
    """
    Inclusion-aware thresholds:
    - allow slight relaxation for thin-file borrowers to reduce exclusion.
    """
    adj_threshold = threshold + thin_file_relaxation * thin_file
    return (joint_score < adj_threshold).astype(int).rename("approved")
