import networkx as nx
import pandas as pd


def build_hetero_graph(
    borrowers: pd.DataFrame, borrower_devices: pd.DataFrame, transactions: pd.DataFrame
) -> nx.Graph:
    """
    Builds a heterogeneous graph:
    - borrower nodes: b_<id>
    - device nodes: d_<id>
    - merchant nodes: m_<id>
    Edges:
    - borrower-device: uses_device
    - borrower-merchant: transacts (with amount and time attributes)
    """
    G = nx.Graph()

    for bid in borrowers["borrower_id"].tolist():
        G.add_node(f"b_{int(bid)}", node_type="borrower")

    for _, r in borrower_devices.iterrows():
        b = f"b_{int(r['borrower_id'])}"
        d = f"d_{int(r['device_id'])}"
        G.add_node(d, node_type="device")
        G.add_edge(b, d, edge_type="uses_device")

    for _, r in transactions.iterrows():
        b = f"b_{int(r['borrower_id'])}"
        m = f"m_{int(r['merchant_id'])}"
        G.add_node(m, node_type="merchant")
        G.add_edge(b, m, edge_type="transacts", amount=float(r["amount"]), t=int(r["t"]))

    return G


def fraud_score_from_graph(
    G: nx.Graph, borrowers: pd.DataFrame, shared_device_threshold: int = 3
) -> pd.Series:
    """
    Simple, explainable fraud score based on shared-device neighborhood size.
    Higher shared device degree => higher fraud suspicion.
    Returns: Series indexed by borrower_id with values in [0, 1].
    """
    scores = {}
    for bid in borrowers["borrower_id"].tolist():
        node = f"b_{int(bid)}"
        neighbors = list(G.neighbors(node))
        device_neighbors = [n for n in neighbors if str(n).startswith("d_")]

        max_shared = 1
        for d in device_neighbors:
            b_on_device = [x for x in G.neighbors(d) if str(x).startswith("b_")]
            max_shared = max(max_shared, len(b_on_device))

        # Normalize: shared_device_threshold=3 means 1 shared -> low, 3+ -> high
        score = (max_shared - 1) / float(shared_device_threshold)
        scores[int(bid)] = float(min(1.0, max(0.0, score)))

    return pd.Series(scores, name="fraud_score")
