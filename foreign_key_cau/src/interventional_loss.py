"""
Interventional Loss: Combining Observational + Causal Objectives

During relational deep learning training, combine two objectives:
  1. Observational loss: L_obs = MSE(ŷ_model − y_observed)
  2. Interventional loss: L_int = MSE(Δŷ_model − τ_empirical)

where Δŷ is the change in prediction under intervention (e.g., age +5 years),
and τ_empirical is the empirically-estimated causal effect from data.

This encourages the model to:
  - Predict accurate outcomes (L_obs)
  - Correctly encode causal effects (L_int)

Total loss: L_total = L_obs + λ × L_int

Usage (PyTorch):
  import torch
  from interventional_loss import InterventionalLoss, ComputeInterventionalGradient

  # Define intervened features
  X_intervened = X_observed.clone()
  X_intervened[:, age_idx] += 5.0  # Intervene: age +5

  # Compute expected intervention effect (from data)
  tau_empirical = EstimateCausalEffect_LinearRegression(...)

  # Define loss
  loss_fn = InterventionalLoss(lambda_weight=0.5)

  # During training:
  y_pred = model(X_observed)
  delta_y_pred, delta_y_intervened = ComputeInterventionalGradient(
    model, X_observed, X_intervened
  )

  loss = loss_fn(
    y_pred=y_pred,
    y_true=y_observed,
    delta_y_pred=delta_y_pred,
    tau_empirical=tau_empirical
  )

References:
  - Leskovec & Jure (2025): "Relational Transformer"
  - Chen et al. (2022): "CIGA - Causal Invariant Graph Representations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class InterventionalLoss(nn.Module):
    """
    Combined observational + interventional loss for relational models.

    Loss = L_obs + λ × L_int
    where:
      L_obs = MSE(ŷ_model - y_true)
      L_int = MSE(Δŷ_model - τ_empirical)
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        loss_type: str = "mse",
        reduction: str = "mean"
    ):
        """
        Args:
            lambda_weight: Weight on interventional loss (default 0.5)
            loss_type: Loss function ('mse', 'mae', 'huber')
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.lambda_weight = lambda_weight
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "mse":
            self.obs_loss = nn.MSELoss(reduction=reduction)
            self.int_loss = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            self.obs_loss = nn.L1Loss(reduction=reduction)
            self.int_loss = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            self.obs_loss = nn.HuberLoss(reduction=reduction)
            self.int_loss = nn.HuberLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        delta_y_pred: torch.Tensor,
        tau_empirical: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            y_pred: Model predictions (n_samples,)
            y_true: True outcomes (n_samples,)
            delta_y_pred: Model's predicted intervention effect (n_samples,)
            tau_empirical: Empirically-estimated causal effect (scalar or n_samples,)
            mask: Optional mask for valid samples (default: all True)

        Returns:
            Scalar loss value
        """

        # Observational loss
        L_obs = self.obs_loss(y_pred, y_true)

        # Interventional loss
        # tau_empirical may be scalar (ATE) or per-sample (CATE/ITE)
        if tau_empirical.dim() == 0:
            tau_empirical = tau_empirical.expand_as(delta_y_pred)

        L_int = self.int_loss(delta_y_pred, tau_empirical)

        # Combined loss
        L_total = L_obs + self.lambda_weight * L_int

        return L_total


def ComputeInterventionalGradient(
    model: nn.Module,
    X_observed: torch.Tensor,
    X_intervened: torch.Tensor,
    intervention_attr_idx: Optional[int] = None,
    intervention_magnitude: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute model's prediction change under intervention.

    Δŷ_model = ŷ(X_intervened) − ŷ(X_observed)

    Args:
        model: Neural network module (must be in eval mode or have no dropout)
        X_observed: Original features (n_samples, n_features)
        X_intervened: Intervened features (n_samples, n_features)
        intervention_attr_idx: Index of intervened attribute (for logging)
        intervention_magnitude: Magnitude of intervention (for logging)

    Returns:
        Tuple of:
          - delta_y_pred: Predicted effect (n_samples,)
          - y_observed: Model's prediction on original data (n_samples,)
          - y_intervened: Model's prediction under intervention (n_samples,)
    """

    model.eval()

    with torch.no_grad():
        y_observed = model(X_observed)
        y_intervened = model(X_intervened)

    delta_y_pred = y_intervened - y_observed

    return delta_y_pred, y_observed, y_intervened


def PrepareInterventionalBatch(
    X: torch.Tensor,
    y: torch.Tensor,
    intervention_attr_idx: int,
    intervention_delta: float,
    tau_empirical: float
) -> Dict[str, torch.Tensor]:
    """
    Prepare batch data for interventional loss computation.

    Creates intervened features and packages everything for loss function.

    Args:
        X: Features (n_samples, n_features)
        y: Outcomes (n_samples,)
        intervention_attr_idx: Column index to intervene on
        intervention_delta: Change magnitude (e.g., +5 for age +5 years)
        tau_empirical: Empirically-estimated effect from data

    Returns:
        Dictionary with:
          - 'X_observed': Original features
          - 'X_intervened': Intervened features
          - 'y_true': Outcomes
          - 'tau_empirical': Causal effect (tensor)
          - 'intervention_delta': Magnitude (for logging)
    """

    X_intervened = X.clone()
    X_intervened[:, intervention_attr_idx] += intervention_delta

    return {
        "X_observed": X,
        "X_intervened": X_intervened,
        "y_true": y,
        "tau_empirical": torch.tensor(tau_empirical, dtype=torch.float32),
        "intervention_delta": intervention_delta
    }


class InterventionalTrainingLoop:
    """
    Helper class for training models with interventional objectives.

    Example:
        loop = InterventionalTrainingLoop(
            model=my_model,
            optimizer=torch.optim.Adam(my_model.parameters(), lr=0.001),
            loss_fn=InterventionalLoss(lambda_weight=0.5),
            device='cuda'
        )

        for epoch in range(num_epochs):
            for X_batch, y_batch in dataloader:
                loss = loop.training_step(
                    X_batch, y_batch,
                    intervention_attr_idx=0,  # intervene on first feature
                    intervention_delta=5.0,   # intervention magnitude
                    tau_empirical=2.0         # expected effect
                )
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: InterventionalLoss,
        device: str = "cpu"
    ):
        """
        Args:
            model: Neural network module to train
            optimizer: PyTorch optimizer
            loss_fn: InterventionalLoss instance
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def training_step(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        intervention_attr_idx: int,
        intervention_delta: float,
        tau_empirical: float
    ) -> float:
        """
        Single training step with interventional loss.

        Args:
            X_batch: Feature batch (n_batch, n_features)
            y_batch: Target batch (n_batch,)
            intervention_attr_idx: Which attribute to intervene
            intervention_delta: Intervention magnitude
            tau_empirical: Expected causal effect

        Returns:
            Scalar loss value
        """

        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Forward pass on observed data
        y_pred = self.model(X_batch).squeeze()

        # Prepare intervened batch
        batch_data = PrepareInterventionalBatch(
            X_batch, y_batch,
            intervention_attr_idx, intervention_delta, tau_empirical
        )

        # Compute intervention effect
        delta_y_pred, _, _ = ComputeInterventionalGradient(
            self.model,
            batch_data["X_observed"],
            batch_data["X_intervened"],
            intervention_attr_idx, intervention_delta
        )

        # Compute loss
        loss = self.loss_fn(
            y_pred=y_pred,
            y_true=y_batch,
            delta_y_pred=delta_y_pred,
            tau_empirical=batch_data["tau_empirical"].to(self.device)
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        intervention_attr_idx: int,
        intervention_delta: float,
        tau_empirical: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validation step (no gradient updates).

        Returns:
            Tuple of (loss, metrics_dict) where metrics_dict contains:
              - 'L_obs': Observational loss only
              - 'L_int': Interventional loss only
              - 'delta_y_pred_mean': Mean predicted intervention effect
              - 'delta_y_pred_std': Std of predicted intervention effect
        """

        self.model.eval()
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        with torch.no_grad():
            y_pred = self.model(X_batch).squeeze()

            batch_data = PrepareInterventionalBatch(
                X_batch, y_batch,
                intervention_attr_idx, intervention_delta, tau_empirical
            )

            delta_y_pred, _, _ = ComputeInterventionalGradient(
                self.model,
                batch_data["X_observed"],
                batch_data["X_intervened"],
                intervention_attr_idx, intervention_delta
            )

            loss = self.loss_fn(
                y_pred=y_pred,
                y_true=y_batch,
                delta_y_pred=delta_y_pred,
                tau_empirical=batch_data["tau_empirical"].to(self.device)
            )

            # Compute individual loss components (for diagnostic)
            L_obs = self.loss_fn.obs_loss(y_pred, y_batch)
            L_int = self.loss_fn.int_loss(
                delta_y_pred,
                batch_data["tau_empirical"].to(self.device)
            )

            metrics = {
                "L_obs": L_obs.item(),
                "L_int": L_int.item(),
                "delta_y_pred_mean": delta_y_pred.mean().item(),
                "delta_y_pred_std": delta_y_pred.std().item()
            }

        return loss.item(), metrics


if __name__ == "__main__":
    # Example: Synthetic relational data with interventional training
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)

    # Generate synthetic data
    n_samples = 200
    X_true = torch.randn(n_samples, 3)
    # y = 2*X0 + 0.5*X1 - X2 + noise
    y_true = (
        2.0 * X_true[:, 0] +
        0.5 * X_true[:, 1] -
        1.0 * X_true[:, 2] +
        0.1 * torch.randn(n_samples)
    )

    print("=" * 70)
    print("EXAMPLE: Interventional Loss Training")
    print("=" * 70)
    print(f"\nSynthetic data: {n_samples} samples, 3 features")
    print(f"True causal effect (τ) for feature 0: 2.0")

    # Define simple model
    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = InterventionalLoss(lambda_weight=0.5)

    # Training loop
    loop = InterventionalTrainingLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu"
    )

    print("\nTraining with interventional loss...")
    print("Intervening on feature 0: +1.0 (expect Δŷ ≈ 2.0)")

    num_epochs = 20
    batch_size = 32

    dataset = TensorDataset(X_true, y_true)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            loss = loop.training_step(
                X_batch, y_batch,
                intervention_attr_idx=0,      # intervene on feature 0
                intervention_delta=1.0,       # +1.0
                tau_empirical=2.0             # expected effect 2.0
            )
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

    # Evaluate
    print("\n[Final Evaluation]")
    val_loss, metrics = loop.validation_step(
        X_true[:50], y_true[:50],
        intervention_attr_idx=0,
        intervention_delta=1.0,
        tau_empirical=2.0
    )

    print(f"  Validation loss: {val_loss:.6f}")
    print(f"  L_obs: {metrics['L_obs']:.6f}")
    print(f"  L_int: {metrics['L_int']:.6f}")
    print(f"  Predicted Δŷ: {metrics['delta_y_pred_mean']:.4f} "
          f"(expected: 2.0, std: {metrics['delta_y_pred_std']:.4f})")
