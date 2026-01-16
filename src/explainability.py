import shap
import pandas as pd
import networkx as nx


def shap_reason_codes(model, X: pd.DataFrame, top_k: int = 5):
    """
    Returns per-row top_k SHAP contributors as (feature, value) pairs.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    reason_codes = []
    for i in range(X.shape[0]):
        contrib = list(zip(X.columns.tolist(), shap_vals[i]))
        contrib = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        reason_codes.append([(f, float(v)) for f, v in contrib])
    return reason_codes


def neighborhood_explanation(G: nx.Graph, borrower_id: int, max_neighbors: int = 20):
    """
    Simple graph explanation: neighborhood listing for the borrower node.
    """
    node = f"b_{int(borrower_id)}"
    neigh = list(G.neighbors(node))[:max_neighbors]
    return {"borrower_node": node, "neighbors": neigh}
