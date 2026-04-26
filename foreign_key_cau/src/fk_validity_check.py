"""
FK→Causality Validity Framework

Validates when foreign key (FK) relationships can be interpreted causally via:
  1. Temporal ordering check (parent created before child)
  2. Domain knowledge alignment (learned τ signs + magnitudes match expert expectations)
  3. Effect consistency (τ stable across train/val splits)

Success criteria:
  - Temporal: ≥95% of parent rows precede child rows
  - Domain: ≥60% of attributes pass sign + magnitude checks
  - Consistency: ≥80% of attributes show 100% sign agreement across folds

Usage:
  validity_report = ValidateFK_Causality(
    tau_estimates={'age': 2.0, 'loyalty': 3.5},
    domain_knowledge={'age': (1, [0, 5]), 'loyalty': (1, [0, 10])}
  )

References:
  - Pearl (1993): Backdoor criterion and adjustment sets
  - RelFCI (2025): Temporal ordering in relational causal discovery
  - Kunzel et al. (2019): Effect consistency across folds (metalearners)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import warnings


def CheckTemporalOrdering(
    parent_table: pd.DataFrame,
    child_table: pd.DataFrame,
    fk_column: str,
    parent_created_col: str = "created_at",
    child_created_col: str = "created_at"
) -> Dict[str, any]:
    """
    Verify foreign key temporal directionality: parent created before child.

    Args:
        parent_table: Parent table (e.g., Customers)
        child_table: Child table (e.g., Orders)
        fk_column: Foreign key column name (e.g., 'customer_id')
        parent_created_col: Timestamp column in parent_table (default 'created_at')
        child_created_col: Timestamp column in child_table (default 'created_at')

    Returns:
        Dictionary with:
          - 'pct_valid': % of pairs with parent.created_at < child.created_at
          - 'status': 'PASS' if ≥95%, 'WARN' if 90-95%, 'FAIL' if <90%
          - 'n_valid': Number of valid pairs
          - 'n_total': Total pairs
          - 'violations': List of violated pair indices (first 10)
    """

    # Merge parent and child on FK
    merged = child_table.merge(
        parent_table,
        left_on=fk_column,
        right_on=fk_column,
        how="left",
        suffixes=("_child", "_parent")
    )

    if len(merged) == 0:
        raise ValueError(f"FK column '{fk_column}' not found in both tables")

    # Check temporal ordering
    if parent_created_col not in merged.columns or child_created_col not in merged.columns:
        raise ValueError(
            f"Timestamp columns '{parent_created_col}' or '{child_created_col}' not found. "
            f"Available columns: {merged.columns.tolist()}"
        )

    parent_created = merged[[col for col in merged.columns if parent_created_col in col]].iloc[:, 0]
    child_created = merged[[col for col in merged.columns if child_created_col in col]].iloc[:, 0]

    valid_order = pd.to_datetime(parent_created) < pd.to_datetime(child_created)
    pct_valid = valid_order.sum() / len(merged)

    violations = np.where(~valid_order)[0][:10]  # First 10 violations

    if pct_valid >= 0.95:
        status = "PASS"
    elif pct_valid >= 0.90:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "pct_valid": float(pct_valid),
        "status": status,
        "n_valid": int(valid_order.sum()),
        "n_total": len(merged),
        "violations": violations.tolist()
    }


def ValidateFK_Causality(
    tau_estimates: Dict[str, float],
    domain_knowledge: Dict[str, Tuple[int, Tuple[float, float]]]
) -> Dict[str, any]:
    """
    Validate learned causal effects against domain expert expectations.

    Args:
        tau_estimates: Dictionary mapping parent_attr → learned τ value
                      Example: {'age': 2.0, 'loyalty_score': 3.5}

        domain_knowledge: Dictionary mapping parent_attr → (expected_sign, (min_mag, max_mag))
                         Example: {
                            'age': (1, (0, 5)),           # positive, expect 0-5
                            'loyalty_score': (1, (0, 10)) # positive, expect 0-10
                         }

    Returns:
        Dictionary with:
          - 'credibility_report': Per-attribute credibility scores
          - 'num_sign_matches': Count of attributes where sign matches
          - 'num_magnitude_ok': Count of attributes in expected range
          - 'overall_credibility': (num_sign_matches + num_magnitude_ok) / (2 * num_attrs)
          - 'status': 'PASS' if ≥0.6, 'WARN' if 0.4-0.6, 'FAIL' if <0.4
    """

    credibility_report = {}
    num_sign_matches = 0
    num_magnitude_ok = 0
    num_attrs = len(tau_estimates)

    for attr, tau_value in tau_estimates.items():
        if attr not in domain_knowledge:
            warnings.warn(f"Attribute '{attr}' not in domain_knowledge. Skipping validation.")
            continue

        expected_sign, (min_mag, max_mag) = domain_knowledge[attr]

        # Check sign agreement
        learned_sign = 1 if tau_value > 0 else (-1 if tau_value < 0 else 0)
        sign_match = (learned_sign == expected_sign)
        if sign_match:
            num_sign_matches += 1

        # Check magnitude credibility
        magnitude_ok = (min_mag <= abs(tau_value) <= max_mag)
        if magnitude_ok:
            num_magnitude_ok += 1

        credibility_score = (float(sign_match) + float(magnitude_ok)) / 2.0

        credibility_report[attr] = {
            "learned_tau": float(tau_value),
            "expected_sign": expected_sign,
            "learned_sign": learned_sign,
            "sign_match": bool(sign_match),
            "expected_magnitude_range": (min_mag, max_mag),
            "magnitude_ok": bool(magnitude_ok),
            "credibility_score": credibility_score
        }

    # Overall credibility: percentage of attributes passing both checks
    overall_credibility = (num_sign_matches + num_magnitude_ok) / (2 * num_attrs) if num_attrs > 0 else 0.0

    if overall_credibility >= 0.6:
        status = "PASS"
    elif overall_credibility >= 0.4:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "credibility_report": credibility_report,
        "num_sign_matches": num_sign_matches,
        "num_magnitude_ok": num_magnitude_ok,
        "num_attributes": num_attrs,
        "overall_credibility": float(overall_credibility),
        "status": status
    }


def CheckEffectConsistency(
    parent_table: pd.DataFrame,
    child_table: pd.DataFrame,
    target: str,
    parent_columns: List[str],
    n_splits: int = 5,
    estimator_fn: callable = None
) -> Dict[str, any]:
    """
    Check τ stability across cross-validation folds.

    Refit τ estimates on k different train/val splits. If τ signs flip across
    folds, the effect is unstable (likely confounding or spurious correlation).

    Args:
        parent_table: Parent table
        child_table: Child table
        target: Target column name
        parent_columns: Parent attributes to estimate τ for
        n_splits: Number of k-fold splits (default 5)
        estimator_fn: Callable that estimates τ. Must return Dict[str, float].
                     If None, uses default linear regression.

    Returns:
        Dictionary with:
          - 'signs_by_col': Dict mapping attr → list of signs across folds
          - 'consistency_pct': % of attributes with 100% sign agreement
          - 'status': 'PASS' if ≥80%, 'WARN' if 60-80%, 'FAIL' if <60%
          - 'unstable_attrs': Attributes with sign flips
    """

    if estimator_fn is None:
        # Default: linear regression τ estimator
        from tau_estimation_linear import EstimateCausalEffect_LinearRegression
        estimator_fn = lambda pt, ct: EstimateCausalEffect_LinearRegression(
            pt, ct, target, parent_columns, method="ate"
        )

    # Merge tables for k-fold splitting
    merged = parent_table.merge(
        child_table,
        left_index=True,
        right_index=True,
        how="outer"
    )

    all_taus = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(merged)):
        train_data = merged.iloc[train_idx]
        parent_train = train_data[[col for col in parent_columns if col in train_data.columns]]
        child_train = train_data[[target]]

        if len(parent_train) == 0 or len(child_train) == 0:
            warnings.warn(f"Fold {fold_idx}: empty train split. Skipping.")
            continue

        tau_fold = estimator_fn(parent_train, child_train)
        all_taus.append(tau_fold)

    # Extract signs across folds
    signs_by_col = {}
    for col in parent_columns:
        signs = [
            1 if tau_dict.get(col, 0) > 0 else (-1 if tau_dict.get(col, 0) < 0 else 0)
            for tau_dict in all_taus
        ]
        signs_by_col[col] = signs

    # Check consistency
    consistency_scores = {}
    unstable_attrs = []

    for col, signs in signs_by_col.items():
        unique_signs = set(signs)
        is_consistent = len(unique_signs) == 1  # All folds agree
        consistency_scores[col] = float(is_consistent)

        if not is_consistent:
            unstable_attrs.append(col)

    consistency_pct = np.mean(list(consistency_scores.values()))

    if consistency_pct >= 0.8:
        status = "PASS"
    elif consistency_pct >= 0.6:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "signs_by_col": signs_by_col,
        "consistency_scores": consistency_scores,
        "consistency_pct": float(consistency_pct),
        "status": status,
        "unstable_attrs": unstable_attrs,
        "n_folds": n_splits
    }


def CheckFK_CausalityValidity(
    parent_table: pd.DataFrame,
    child_table: pd.DataFrame,
    fk_column: str,
    target: str,
    parent_columns: List[str],
    domain_knowledge: Dict[str, Tuple[int, Tuple[float, float]]],
    tau_estimates: Dict[str, float],
    parent_created_col: str = "created_at",
    child_created_col: str = "created_at"
) -> Dict[str, any]:
    """
    Complete FK→Causality validity check: all three criteria.

    Args:
        parent_table, child_table: Relational tables
        fk_column: Foreign key column name
        target: Target column name
        parent_columns: Parent attributes for τ estimation
        domain_knowledge: Expert expectations (see ValidateFK_Causality)
        tau_estimates: Already-estimated τ values
        parent_created_col, child_created_col: Timestamp columns

    Returns:
        Dictionary combining results from all three checks with overall verdict.
    """

    print("=" * 70)
    print("FK→CAUSALITY VALIDITY CHECK")
    print("=" * 70)

    # Check 1: Temporal ordering
    print("\n[1] Temporal Ordering Check")
    temporal_result = CheckTemporalOrdering(
        parent_table, child_table, fk_column,
        parent_created_col, child_created_col
    )
    print(f"  Status: {temporal_result['status']}")
    print(f"  Valid pairs: {temporal_result['n_valid']} / {temporal_result['n_total']} "
          f"({100*temporal_result['pct_valid']:.1f}%)")

    # Check 2: Domain knowledge alignment
    print("\n[2] Domain Knowledge Alignment")
    domain_result = ValidateFK_Causality(tau_estimates, domain_knowledge)
    print(f"  Status: {domain_result['status']}")
    print(f"  Sign matches: {domain_result['num_sign_matches']} / {domain_result['num_attributes']}")
    print(f"  Magnitude OK: {domain_result['num_magnitude_ok']} / {domain_result['num_attributes']}")
    print(f"  Overall credibility: {100*domain_result['overall_credibility']:.1f}%")

    # Check 3: Effect consistency
    print("\n[3] Effect Consistency (k-fold)")
    consistency_result = CheckEffectConsistency(
        parent_table, child_table, target, parent_columns, n_splits=5
    )
    print(f"  Status: {consistency_result['status']}")
    print(f"  Consistency: {100*consistency_result['consistency_pct']:.1f}%")
    if consistency_result['unstable_attrs']:
        print(f"  Unstable attributes: {', '.join(consistency_result['unstable_attrs'])}")

    # Overall verdict
    print("\n[OVERALL VERDICT]")
    temporal_pass = temporal_result['status'] == 'PASS'
    domain_pass = domain_result['status'] in ['PASS', 'WARN']
    consistency_pass = consistency_result['status'] == 'PASS'

    validity_valid = temporal_pass and domain_pass and consistency_pass

    if validity_valid:
        verdict = "✓ FK→CAUSALITY VALID"
    else:
        verdict = "✗ FK→CAUSALITY INVALID"

    print(f"  {verdict}")
    print(f"  Temporal (T1): {temporal_result['status']} {'✓' if temporal_pass else '✗'}")
    print(f"  Domain (D1+D2): {domain_result['status']} {'✓' if domain_pass else '✗'}")
    print(f"  Consistency (C1): {consistency_result['status']} {'✓' if consistency_pass else '✗'}")

    return {
        "temporal": temporal_result,
        "domain_knowledge": domain_result,
        "consistency": consistency_result,
        "overall_valid": validity_valid,
        "verdict": verdict
    }


if __name__ == "__main__":
    # Example: Synthetic e-commerce validation
    np.random.seed(42)

    n_customers = 100
    n_orders = 500

    # Create synthetic Customers
    customers = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "age": np.random.normal(40, 15, n_customers).clip(18, 80),
        "loyalty_score": np.random.uniform(0, 10, n_customers),
        "created_at": pd.date_range("2023-01-01", periods=n_customers, freq="D")
    })

    # Create synthetic Orders
    customer_ids = np.random.choice(n_customers, n_orders)
    orders = pd.DataFrame({
        "order_id": np.arange(n_orders),
        "customer_id": customer_ids,
        "order_amount": (
            50.0 +
            2.0 * customers.loc[customer_ids, "age"].values +
            3.5 * customers.loc[customer_ids, "loyalty_score"].values +
            np.random.normal(0, 10, n_orders)
        ),
        "created_at": pd.date_range("2023-04-01", periods=n_orders, freq="H")
    })

    print("Synthetic data created:")
    print(f"  Customers: {len(customers)} rows")
    print(f"  Orders: {len(orders)} rows")

    # Test 1: Temporal ordering
    print("\n" + "=" * 70)
    print("TEST 1: Temporal Ordering")
    print("=" * 70)
    temporal = CheckTemporalOrdering(customers, orders, "customer_id")
    print(f"Temporal ordering: {temporal['status']}")
    print(f"  {temporal['pct_valid']:.1%} of pairs valid")

    # Test 2: Domain knowledge validation
    print("\n" + "=" * 70)
    print("TEST 2: Domain Knowledge Alignment")
    print("=" * 70)
    tau_estimates = {"age": 2.0, "loyalty_score": 3.5}
    domain_knowledge = {
        "age": (1, (0, 5)),
        "loyalty_score": (1, (0, 10))
    }
    domain = ValidateFK_Causality(tau_estimates, domain_knowledge)
    print(f"Domain alignment: {domain['status']}")
    print(f"  Overall credibility: {domain['overall_credibility']:.1%}")

    # Test 3: Effect consistency (simplified, without refitting)
    print("\n" + "=" * 70)
    print("TEST 3: Effect Consistency")
    print("=" * 70)
    consistency = CheckEffectConsistency(
        customers, orders, "order_amount",
        ["age", "loyalty_score"], n_splits=3
    )
    print(f"Consistency: {consistency['status']}")
    print(f"  {consistency['consistency_pct']:.1%} of attributes stable")
