"""
Strategy A: Linear Regression Causal Effect (τ) Estimator

Implements empirical linear regression for estimating average treatment effects (ATE)
and conditional average treatment effects (CATE) in relational data.

Key assumptions:
  - Linear parent→child relationship: Y = β₀ + Σβᵢ × parent_attrᵢ + ε
  - No unmeasured confounders (causal sufficiency)
  - Parent precedes child temporally

Usage:
  tau_dict = EstimateCausalEffect_LinearRegression(
    customers, orders, 'order_amount',
    parent_columns=['age', 'loyalty_score'],
    method='ate'
  )
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import variance_inflation_factor


def EstimateCausalEffect_LinearRegression(
    parent_table: pd.DataFrame,
    child_table: pd.DataFrame,
    target: str,
    parent_columns: List[str],
    child_columns: Optional[List[str]] = None,
    method: str = "ate"
) -> Dict[str, float]:
    """
    Estimate causal effects (τ) using linear regression on relational data.

    Args:
        parent_table: DataFrame with parent entity attributes
        child_table: DataFrame with outcomes and child attributes
        target: Column name in child_table (outcome Y)
        parent_columns: Column names in parent_table (treatment X)
        child_columns: Column names in child_table for CATE (optional)
        method: One of 'ate' (Average), 'cate' (Conditional), 'ite' (Individual)

    Returns:
        Dictionary mapping parent_column → τ estimate (for ATE)
        or parent_column → {subgroup_id: τ} for CATE

    Raises:
        ValueError: If target not in child_table or parent_columns not found
    """

    if target not in child_table.columns:
        raise ValueError(f"Target '{target}' not in child_table columns")

    missing_cols = set(parent_columns) - set(parent_table.columns)
    if missing_cols:
        raise ValueError(f"Parent columns {missing_cols} not found in parent_table")

    # Extract features and target
    X = parent_table[parent_columns].values
    y = child_table[target].values

    if len(X) != len(y):
        raise ValueError(
            f"Dimension mismatch: parent_table ({len(X)}) "
            f"vs child_table ({len(y)})"
        )

    # Fit OLS model
    model = LinearRegression()
    model.fit(X, y)

    tau_dict = {col: model.coef_[i] for i, col in enumerate(parent_columns)}

    if method == "ate":
        return tau_dict

    elif method == "cate":
        if child_columns is None or len(child_columns) == 0:
            raise ValueError("CATE requires child_columns to be specified")

        # Augment X with interaction terms
        X_child = child_table[child_columns].values
        X_interactions = np.column_stack([
            X[:, i] * X_child[:, j]
            for i in range(X.shape[1])
            for j in range(X_child.shape[1])
        ])
        X_augmented = np.column_stack([X, X_interactions])

        model_cate = LinearRegression()
        model_cate.fit(X_augmented, y)

        # Extract interaction coefficients (CATE = β_interaction × child_attr)
        n_parent = X.shape[1]
        n_child = X_child.shape[1]
        cate_dict = {}

        for i, p_col in enumerate(parent_columns):
            for j, c_col in enumerate(child_columns):
                idx = n_parent + i * n_child + j
                cate_dict[f"{p_col}×{c_col}"] = model_cate.coef_[idx]

        return cate_dict

    elif method == "ite":
        # Per-sample ITE: τⱼ = model.predict(Xⱼ_intervened) - model.predict(Xⱼ)
        # Warning: ITE is high-variance and prone to overfitting unless n >> p
        if len(X) < 50000:
            print(
                "WARNING: ITE estimation with n < 50K is high-variance. "
                "Consider ATE instead."
            )

        # Placeholder: return average of predicted residuals as per-sample proxy
        y_pred = model.predict(X)
        ite_residuals = y - y_pred
        return {
            "ite_per_sample": ite_residuals,
            "ite_mean": np.mean(ite_residuals),
            "ite_std": np.std(ite_residuals)
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ate', 'cate', or 'ite'.")


def ValidateLinearModel(
    model: LinearRegression,
    X: np.ndarray,
    y: np.ndarray,
    X_col_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Validate linear model fit: R², residuals, VIF (multicollinearity).

    Args:
        model: Fitted LinearRegression model
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        X_col_names: Feature names for VIF reporting

    Returns:
        Dictionary with validation metrics:
          - r2: Coefficient of determination
          - rmse: Root mean squared error
          - residuals: Residual vector
          - vif: Dict of variance inflation factors per feature
          - issues: List of warnings (e.g., "High collinearity: age (VIF=12.5)")
    """

    y_pred = model.predict(X)
    residuals = y - y_pred

    # R² (coefficient of determination)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))

    # VIF (variance inflation factor) for multicollinearity
    vif_dict = {}
    issues = []

    if X_col_names is None:
        X_col_names = [f"X{i}" for i in range(X.shape[1])]

    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X, i)
        vif_dict[X_col_names[i]] = vif
        if vif > 5:
            issues.append(f"High collinearity: {X_col_names[i]} (VIF={vif:.2f})")

    return {
        "r2": r2,
        "rmse": rmse,
        "residuals": residuals,
        "vif": vif_dict,
        "issues": issues
    }


if __name__ == "__main__":
    # Example: Synthetic e-commerce data
    np.random.seed(42)

    # Generate synthetic Customers table
    n_customers = 100
    customers = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "age": np.random.normal(40, 15, n_customers).clip(18, 80),
        "loyalty_score": np.random.uniform(0, 10, n_customers),
        "account_age": np.random.exponential(2, n_customers)
    })

    # Generate synthetic Orders table (with causal relationship)
    n_orders = 500
    customer_ids = np.random.choice(n_customers, n_orders)
    orders = pd.DataFrame({
        "order_id": np.arange(n_orders),
        "customer_id": customer_ids,
        "order_amount": (
            50.0 +
            2.0 * customers.loc[customer_ids, "age"].values +
            3.5 * customers.loc[customer_ids, "loyalty_score"].values -
            0.5 * customers.loc[customer_ids, "account_age"].values +
            np.random.normal(0, 10, n_orders)
        )
    })

    print("=" * 70)
    print("EXAMPLE: Linear Regression τ Estimation")
    print("=" * 70)

    # Estimate ATE
    tau_ate = EstimateCausalEffect_LinearRegression(
        customers, orders, "order_amount",
        parent_columns=["age", "loyalty_score", "account_age"],
        method="ate"
    )
    print("\n[ATE Estimates]")
    for attr, tau in tau_ate.items():
        print(f"  τ({attr}) = {tau:.4f}")

    # Interpretation
    print("\n[Interpretation]")
    print(f"  Each year of age → +${tau_ate['age']:.2f} order amount")
    print(f"  Each loyalty unit → +${tau_ate['loyalty_score']:.2f}")
    print(f"  Each account year → ${tau_ate['account_age']:.2f} (negative saturation)")

    # Validate model
    X = customers[["age", "loyalty_score", "account_age"]].values
    y = orders["order_amount"].values

    model = LinearRegression()
    model.fit(X, y)

    validation = ValidateLinearModel(
        model, X, y,
        X_col_names=["age", "loyalty_score", "account_age"]
    )

    print("\n[Model Validation]")
    print(f"  R² = {validation['r2']:.4f}")
    print(f"  RMSE = ${validation['rmse']:.2f}")
    print(f"  VIF: {validation['vif']}")
    if validation['issues']:
        print(f"  Issues: {', '.join(validation['issues'])}")
    else:
        print(f"  Issues: None (model well-specified)")
