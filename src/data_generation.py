import os
import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def generate_synthetic(
    out_dir: str,
    n_borrowers: int,
    fraud_rate: float,
    thin_file_rate: float,
    seed: int = 42,
) -> None:
    """
    Generates privacy-safe synthetic data aligned with the paper:
    - borrowers.csv: borrower-level features + labels (default, fraud)
    - borrower_devices.csv: borrower-device links (supports shared-device fraud rings)
    - transactions.csv: borrower-merchant transactions with time index

    Files are written to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    borrower_id = np.arange(1, n_borrowers + 1)
    thin_file = (np.random.rand(n_borrowers) < thin_file_rate).astype(int)

    # Synthetic "group" proxy used for fairness diagnostics (no real sensitive attributes)
    group = (np.random.rand(n_borrowers) < 0.5).astype(int)

    income = np.where(
        thin_file == 1,
        np.random.lognormal(mean=7.1, sigma=0.5, size=n_borrowers),
        np.random.lognormal(mean=7.6, sigma=0.45, size=n_borrowers),
    )
    income = np.clip(income, 300, 20000).round(0)

    borrowers = pd.DataFrame(
        {
            "borrower_id": borrower_id,
            "thin_file": thin_file,
            "group": group,
            "income": income,
        }
    )

    # Devices (shared devices simulate synthetic identity rings)
    n_devices = max(1, int(n_borrowers * 0.7))
    device_ids = np.arange(1, n_devices + 1)
    device_map = np.random.choice(device_ids, size=n_borrowers, replace=True)

    fraud_label = (np.random.rand(n_borrowers) < fraud_rate).astype(int)

    # Inject rings: many fraud borrowers share a small set of devices
    ring_devices = np.random.choice(device_ids, size=min(5, len(device_ids)), replace=False)
    fraud_indices = np.where(fraud_label == 1)[0]
    if len(fraud_indices) > 0:
        device_map[fraud_indices] = np.random.choice(ring_devices, size=len(fraud_indices), replace=True)

    borrower_devices = pd.DataFrame({"borrower_id": borrower_id, "device_id": device_map})

    # Merchants + transactions
    n_merchants = max(1, int(n_borrowers * 0.2))
    merchant_ids = np.arange(1, n_merchants + 1)

    n_txn = max(1, int(n_borrowers * 8))
    txn_borrower = np.random.choice(borrower_id, size=n_txn, replace=True)
    txn_merchant = np.random.choice(merchant_ids, size=n_txn, replace=True)

    base_amt = np.random.gamma(shape=2.0, scale=40.0, size=n_txn)

    fraud_txn_mask = (np.isin(txn_borrower, borrower_id[fraud_label == 1]) & (np.random.rand(n_txn) < 0.3))
    base_amt[fraud_txn_mask] = np.random.gamma(shape=2.0, scale=120.0, size=fraud_txn_mask.sum())

    transactions = pd.DataFrame(
        {
            "txn_id": np.arange(1, n_txn + 1),
            "borrower_id": txn_borrower,
            "merchant_id": txn_merchant,
            "amount": base_amt.round(2),
            "t": np.random.randint(1, 31, size=n_txn),  # day index
        }
    )

    # Default label correlated with thin-file + fraud + income (synthetic but plausible)
    logits = (-2.2 + 0.9 * thin_file + 1.2 * fraud_label - 0.00008 * income + 0.2 * group)
    p_default = 1 / (1 + np.exp(-logits))
    default_label = (np.random.rand(n_borrowers) < p_default).astype(int)

    borrowers["fraud"] = fraud_label
    borrowers["default"] = default_label

    borrowers.to_csv(os.path.join(out_dir, "borrowers.csv"), index=False)
    borrower_devices.to_csv(os.path.join(out_dir, "borrower_devices.csv"), index=False)
    transactions.to_csv(os.path.join(out_dir, "transactions.csv"), index=False)
