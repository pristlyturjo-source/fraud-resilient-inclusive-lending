import os
import yaml
import pandas as pd

from src.data_generation import generate_synthetic
from src.fraud_graph import build_hetero_graph, fraud_score_from_graph
from src.credit_model import make_credit_features, train_credit_model, predict_default_prob
from src.decision_engine import compute_joint_score, approve
from src.explainability import shap_reason_codes, neighborhood_explanation


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    exp = load_yaml("configs/experiment.yaml")
    credit_cfg = load_yaml("configs/credit.yaml")
    fraud_cfg = load_yaml("configs/fraud.yaml")
    decision_cfg = load_yaml("configs/decision.yaml")

    out_dir = "data/synthetic"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Generate synthetic data
    generate_synthetic(
        out_dir=out_dir,
        n_borrowers=exp["n_borrowers"],
        fraud_rate=exp["fraud_rate"],
        thin_file_rate=exp["thin_file_rate"],
        seed=exp["random_seed"],
    )

    borrowers = pd.read_csv(os.path.join(out_dir, "borrowers.csv"))
    borrower_devices = pd.read_csv(os.path.join(out_dir, "borrower_devices.csv"))
    transactions = pd.read_csv(os.path.join(out_dir, "transactions.csv"))

    # 2) Graph + fraud scores
    G = build_hetero_graph(borrowers, borrower_devices, transactions)
    fraud_score = fraud_score_from_graph(
        G, borrowers, shared_device_threshold=fraud_cfg["params"]["shared_device_threshold"]
    )

    # 3) Credit model
    X = make_credit_features(borrowers)
    y = borrowers["default"]

    n = len(borrowers)
    split = int(exp["train_ratio"] * n)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = train_credit_model(X_train, y_train, credit_cfg["params"])
    p_default = predict_default_prob(model, X_test)

    # 4) Joint decisioning
    fraud_test = fraud_score.iloc[split:].reset_index(drop=True)
    joint = compute_joint_score(p_default.reset_index(drop=True), fraud_test, decision_cfg["lambda_fraud"])
    approved = approve(
        joint_score=joint,
        threshold=decision_cfg["approval_threshold"],
        thin_file=borrowers["thin_file"].iloc[split:].reset_index(drop=True),
        thin_file_relaxation=decision_cfg["thin_file_relaxation"],
    )

    # 5) Explainability samples
    reason_codes = shap_reason_codes(model, X_test.head(10), top_k=5)
    graph_exp = neighborhood_explanation(G, int(borrowers["borrower_id"].iloc[split]), max_neighbors=15)

    os.makedirs("outputs", exist_ok=True)

    # Save minimal outputs for proof-of-run
    out = pd.DataFrame({
        "borrower_id": borrowers["borrower_id"].iloc[split:].reset_index(drop=True),
        "p_default": p_default.reset_index(drop=True),
        "fraud_score": fraud_test,
        "joint_score": joint,
        "approved": approved,
        "thin_file": borrowers["thin_file"].iloc[split:].reset_index(drop=True),
        "group": borrowers["group"].iloc[split:].reset_index(drop=True),
        "default": y_test.reset_index(drop=True),
        "fraud": borrowers["fraud"].iloc[split:].reset_index(drop=True),
    })
    out.to_csv("outputs/predictions.csv", index=False)

    # Save explanations
    pd.Series([str(rc) for rc in reason_codes]).to_csv("outputs/shap_reason_codes_sample.csv", index=False)
    with open("outputs/graph_explanation_sample.json", "w", encoding="utf-8") as f:
        f.write(str(graph_exp))

    print("✅ Done. Outputs written to outputs/")
    print(" - outputs/predictions.csv")
    print(" - outputs/shap_reason_codes_sample.csv")
    print(" - outputs/graph_explanation_sample.json")


if __name__ == "__main__":
    main()
