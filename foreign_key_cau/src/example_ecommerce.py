"""
Worked Example: E-commerce Customer Purchase Prediction with FK-Guided Causality

Full end-to-end demonstration:
  1. Generate synthetic Customers + Orders relational data
  2. Estimate causal effects (τ) via linear regression (Strategy A)
  3. Validate τ against domain knowledge
  4. Check temporal ordering and effect consistency
  5. Train neural network with interventional loss

Schema:
  Customers:
    - customer_id (PK)
    - age, region, account_age, loyalty_score
    - created_at (timestamp)

  Orders (FK: customer_id → Customers.customer_id):
    - order_id (PK)
    - customer_id (FK)
    - order_amount (outcome), num_items, days_since_last_order
    - created_at (timestamp)

Causal structure (presumed):
  customer attributes → order amount (customer properties influence purchase value)

References:
  See research_out.json for full literature synthesis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta

# Import our modules
from tau_estimation_linear import EstimateCausalEffect_LinearRegression, ValidateLinearModel
from fk_validity_check import (
    CheckTemporalOrdering,
    ValidateFK_Causality,
    CheckEffectConsistency
)
from interventional_loss import InterventionalLoss, InterventionalTrainingLoop


def GenerateSyntheticRelationalData(
    n_customers: int = 100,
    n_orders_per_customer: int = 5,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic Customers + Orders tables with causal relationship.

    True causal effects:
      - age: +$2.0 per year
      - loyalty_score: +$3.5 per unit
      - account_age: -$0.5 per year (saturation effect)

    Args:
        n_customers: Number of customer entities
        n_orders_per_customer: Average orders per customer
        seed: Random seed

    Returns:
        Tuple of (customers_df, orders_df)
    """

    np.random.seed(seed)

    # Generate Customers table
    customers = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "age": np.random.normal(40, 15, n_customers).clip(18, 80),
        "region": np.random.choice(["US_East", "US_West", "EU"], n_customers),
        "account_age": np.random.exponential(2, n_customers),
        "loyalty_score": np.random.uniform(0, 10, n_customers),
        "created_at": [
            datetime(2022, 1, 1) + timedelta(days=int(d))
            for d in np.random.uniform(0, 365, n_customers)
        ]
    })

    # Generate Orders table with causal relationship
    n_orders = int(n_customers * n_orders_per_customer)
    customer_ids = np.random.choice(n_customers, n_orders)

    # True causal model: order_amount = β₀ + β₁*age + β₂*loyalty + β₃*account_age + ε
    orders = pd.DataFrame({
        "order_id": np.arange(n_orders),
        "customer_id": customer_ids,
        "order_amount": (
            50.0 +  # Base amount
            2.0 * customers.loc[customer_ids, "age"].values +
            3.5 * customers.loc[customer_ids, "loyalty_score"].values -
            0.5 * customers.loc[customer_ids, "account_age"].values +
            np.random.normal(0, 10, n_orders)  # Observation noise
        ),
        "num_items": np.random.poisson(3, n_orders),
        "days_since_last_order": np.random.exponential(7, n_orders),
        "created_at": [
            customers.loc[cid, "created_at"] + timedelta(days=int(d))
            for cid, d in zip(customer_ids, np.random.uniform(0, 365, n_orders))
        ]
    })

    return customers, orders


def FullPipelineExample():
    """
    Complete FK→Causality pipeline: estimate, validate, and train.
    """

    print("=" * 70)
    print("FK-GUIDED CAUSAL EFFECT ESTIMATION: E-COMMERCE EXAMPLE")
    print("=" * 70)

    # ========================================================================
    # PHASE 1: Generate Synthetic Data
    # ========================================================================
    print("\n[PHASE 1] Generating Synthetic Relational Data")
    print("-" * 70)

    customers, orders = GenerateSyntheticRelationalData(
        n_customers=100,
        n_orders_per_customer=5,
        seed=42
    )

    print(f"  Customers table: {len(customers)} rows")
    print(f"  Orders table: {len(orders)} rows")
    print(f"\n  Sample Customers:")
    print(customers.head(3).to_string())
    print(f"\n  Sample Orders:")
    print(orders.head(3).to_string())

    # ========================================================================
    # PHASE 2: Estimate Causal Effects (τ)
    # ========================================================================
    print("\n[PHASE 2] Estimating Causal Effects (τ)")
    print("-" * 70)

    parent_columns = ["age", "loyalty_score", "account_age"]

    tau_estimates = EstimateCausalEffect_LinearRegression(
        parent_table=customers,
        child_table=orders,
        target="order_amount",
        parent_columns=parent_columns,
        method="ate"
    )

    print("  Estimated causal effects (ATE):")
    for attr, tau in tau_estimates.items():
        print(f"    τ({attr}) = ${tau:.4f}")

    # Interpretation
    print("\n  Interpretation:")
    print(f"    Each year of age → +${tau_estimates['age']:.2f} order amount")
    print(f"    Each loyalty unit → +${tau_estimates['loyalty_score']:.2f}")
    print(f"    Each account year → ${tau_estimates['account_age']:.2f}")

    # ========================================================================
    # PHASE 3: Validate Model Fit
    # ========================================================================
    print("\n[PHASE 3] Validating Model Fit")
    print("-" * 70)

    X = customers[parent_columns].values
    y = orders["order_amount"].values

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    validation = ValidateLinearModel(model, X, y, X_col_names=parent_columns)

    print(f"  R² = {validation['r2']:.4f}")
    print(f"  RMSE = ${validation['rmse']:.2f}")
    print(f"  VIF (variance inflation):")
    for attr, vif in validation['vif'].items():
        print(f"    {attr}: {vif:.2f}")

    if validation['issues']:
        print(f"  ⚠ Issues: {', '.join(validation['issues'])}")
    else:
        print(f"  ✓ Model well-specified (no multicollinearity)")

    # ========================================================================
    # PHASE 4: Temporal Ordering Check
    # ========================================================================
    print("\n[PHASE 4] FK→Causality Validity: Temporal Ordering")
    print("-" * 70)

    temporal_result = CheckTemporalOrdering(
        parent_table=customers,
        child_table=orders,
        fk_column="customer_id",
        parent_created_col="created_at",
        child_created_col="created_at"
    )

    print(f"  Status: {temporal_result['status']}")
    print(f"  Valid pairs: {temporal_result['n_valid']} / {temporal_result['n_total']} "
          f"({100*temporal_result['pct_valid']:.1f}%)")

    if temporal_result['status'] == 'PASS':
        print(f"  ✓ Temporal ordering satisfied (parent precedes child)")
    else:
        print(f"  ✗ Temporal ordering violated")

    # ========================================================================
    # PHASE 5: Domain Knowledge Validation
    # ========================================================================
    print("\n[PHASE 5] FK→Causality Validity: Domain Knowledge Alignment")
    print("-" * 70)

    # Expert expectations
    domain_knowledge = {
        "age": (1, (0, 5)),              # positive, expect 0-5
        "loyalty_score": (1, (0, 10)),   # positive, expect 0-10
        "account_age": (-1, (-2, 0))     # negative, expect -2-0
    }

    domain_result = ValidateFK_Causality(tau_estimates, domain_knowledge)

    print("  Domain knowledge alignment:")
    for attr, cred in domain_result['credibility_report'].items():
        match = "✓" if cred['sign_match'] else "✗"
        mag = "✓" if cred['magnitude_ok'] else "✗"
        print(f"    {attr}: τ={cred['learned_tau']:+.2f}, "
              f"sign {match}, magnitude {mag} "
              f"(credibility: {cred['credibility_score']:.2f})")

    print(f"\n  Overall credibility: {domain_result['overall_credibility']:.1%}")
    print(f"  Status: {domain_result['status']}")

    # ========================================================================
    # PHASE 6: Effect Consistency (k-fold validation)
    # ========================================================================
    print("\n[PHASE 6] FK→Causality Validity: Effect Consistency")
    print("-" * 70)

    consistency_result = CheckEffectConsistency(
        parent_table=customers,
        child_table=orders,
        target="order_amount",
        parent_columns=parent_columns,
        n_splits=5
    )

    print(f"  Consistency across 5 folds:")
    for attr, signs in consistency_result['signs_by_col'].items():
        consistent = len(set(signs)) == 1
        status = "✓" if consistent else "✗"
        print(f"    {attr}: {status} (signs: {signs})")

    print(f"\n  Overall consistency: {consistency_result['consistency_pct']:.1%}")
    print(f"  Status: {consistency_result['status']}")

    if consistency_result['unstable_attrs']:
        print(f"  ⚠ Unstable attributes: {', '.join(consistency_result['unstable_attrs'])}")

    # ========================================================================
    # PHASE 7: Interventional Training
    # ========================================================================
    print("\n[PHASE 7] Training Neural Network with Interventional Loss")
    print("-" * 70)

    # Prepare data as tensors
    X_tensor = torch.tensor(customers[parent_columns].values, dtype=torch.float32)
    y_tensor = torch.tensor(orders["order_amount"].values, dtype=torch.float32)

    # Normalize for stability
    X_mean = X_tensor.mean(dim=0)
    X_std = X_tensor.std(dim=0)
    X_tensor = (X_tensor - X_mean) / X_std

    # Define neural network
    class RelationalNet(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=16):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x.squeeze()

    model_nn = RelationalNet(input_dim=len(parent_columns), hidden_dim=16)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01)
    loss_fn = InterventionalLoss(lambda_weight=0.5)

    loop = InterventionalTrainingLoop(
        model=model_nn,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu"
    )

    print("  Training on 20 epochs...")
    print("  Intervention: age +0.5 std (expect Δŷ ≈ 2.0 / std_age)")

    num_epochs = 20
    batch_size = 32
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            loss = loop.training_step(
                X_batch, y_batch,
                intervention_attr_idx=0,           # intervene on age
                intervention_delta=0.5,            # +0.5 std
                tau_empirical=tau_estimates['age'] / X_std[0].item()  # normalize effect
            )
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

    print(f"  ✓ Training complete")

    # ========================================================================
    # PHASE 8: Final Validation
    # ========================================================================
    print("\n[PHASE 8] Final Evaluation")
    print("-" * 70)

    val_loss, metrics = loop.validation_step(
        X_tensor[:50], y_tensor[:50],
        intervention_attr_idx=0,
        intervention_delta=0.5,
        tau_empirical=tau_estimates['age'] / X_std[0].item()
    )

    print(f"  Validation loss: {val_loss:.6f}")
    print(f"  L_obs (observational): {metrics['L_obs']:.6f}")
    print(f"  L_int (interventional): {metrics['L_int']:.6f}")
    print(f"  Predicted Δŷ: {metrics['delta_y_pred_mean']:.4f} "
          f"(std: {metrics['delta_y_pred_std']:.4f})")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: FK→CAUSALITY VALIDITY")
    print("=" * 70)

    checks_pass = [
        ("Temporal Ordering", temporal_result['status'] == 'PASS'),
        ("Domain Knowledge", domain_result['status'] in ['PASS', 'WARN']),
        ("Effect Consistency", consistency_result['status'] == 'PASS')
    ]

    print("\nValidity Checks:")
    for check_name, passed in checks_pass:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name}: {status}")

    overall_valid = all(p for _, p in checks_pass)
    verdict = "✓ FK→CAUSALITY VALID" if overall_valid else "✗ FK→CAUSALITY INVALID"

    print(f"\nOverall Verdict: {verdict}")

    if overall_valid:
        print("\nInterpretation:")
        print(f"  The FK relationship Customers → Orders can be interpreted causally.")
        print(f"  Estimated effects:")
        for attr, tau in tau_estimates.items():
            print(f"    {attr}: τ = ${tau:.2f}")
    else:
        print("\nCaution: FK→causality interpretation may not be valid.")
        print("Check individual failures above for details.")

    print("\n" + "=" * 70)

    return {
        "customers": customers,
        "orders": orders,
        "tau_estimates": tau_estimates,
        "temporal_result": temporal_result,
        "domain_result": domain_result,
        "consistency_result": consistency_result,
        "model": model_nn,
        "losses": losses
    }


if __name__ == "__main__":
    results = FullPipelineExample()

    # Optional: Plot loss convergence
    print("\nNote: To visualize loss convergence, run:")
    print("  import matplotlib.pyplot as plt")
    print("  plt.plot(results['losses'])")
    print("  plt.xlabel('Epoch')")
    print("  plt.ylabel('Loss')")
    print("  plt.title('Interventional Loss during Training')")
    print("  plt.show()")
