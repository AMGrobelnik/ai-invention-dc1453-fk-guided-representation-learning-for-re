"""
Strategy C: Kernel Ridge Regression τ Estimator

Implements nonparametric causal effect estimation via kernel ridge regression.
Handles nonlinear parent→child relationships without model specification.

Key assumptions:
  - Parent attributes as continuous "treatments"; outcome as response
  - Causal effect τ can be nonlinear (captured by RKHS norms)
  - Sufficient data: n > p² for numerical stability

Methods:
  - RBF kernel: exp(-γ × ||x - x'||²) — defaults γ = 1/n_features
  - Linear kernel: <x, x'>
  - Matérn kernel: (√5|d|/l) exp(-√5|d|/l) + 5|d|²/(3l²)

References:
  Sinha & Gretton (2020): "Kernel Methods for Causal Functions"
  https://arxiv.org/abs/2010.04855

Usage:
  tau_curve = EstimateCausalEffect_KernelRidgeRegression(
    parent_features, child_outcomes,
    kernel_type='rbf', lambda_reg=0.01
  )
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import solve
import warnings


def RBFKernel(X1: np.ndarray, X2: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix: exp(-γ × ||x - x'||²).

    Args:
        X1: Array (n, p) or (m, p)
        X2: Array (n, p) or (m, p)
        gamma: Kernel bandwidth. Defaults to 1/p

    Returns:
        Kernel matrix (n, m) or (m, n)
    """
    if gamma is None:
        gamma = 1.0 / X1.shape[1]

    distances = cdist(X1, X2, metric='euclidean')
    return np.exp(-gamma * (distances ** 2))


def LinearKernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute linear kernel matrix: <x, x'>."""
    return np.dot(X1, X2.T)


def MaternKernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: float = 1.0,
    nu: float = 2.5
) -> np.ndarray:
    """
    Compute Matérn kernel: (√(2ν)|d|/l)^ν × (2^(1-ν)/Γ(ν)) × K_ν(√(2ν)|d|/l).

    For ν=2.5 (common choice): K(d) = (√5|d|/l)(1 + √5|d|/l) exp(-√5|d|/l)

    Args:
        X1, X2: Feature arrays
        length_scale: l parameter
        nu: Smoothness parameter (default 2.5 for twice differentiability)

    Returns:
        Kernel matrix
    """
    distances = cdist(X1, X2, metric='euclidean')

    if nu == 2.5:
        # Closed form for ν=2.5
        term = np.sqrt(5) * distances / length_scale
        kernel = (1.0 + term) * np.exp(-term)
    else:
        from scipy.special import kv, gamma as gamma_fn
        sqrt_term = np.sqrt(2 * nu) * distances / length_scale
        # Avoid division by zero
        kernel = np.zeros_like(distances)
        nonzero = sqrt_term > 0
        kernel[nonzero] = (
            (2 ** (1 - nu)) / gamma_fn(nu) *
            (sqrt_term[nonzero] ** nu) *
            kv(nu, sqrt_term[nonzero])
        )
        # At d=0, K(0) = 1
        kernel[~nonzero] = 1.0

    return kernel


def EstimateCausalEffect_KernelRidgeRegression(
    parent_features: np.ndarray,
    child_outcomes: np.ndarray,
    kernel_type: str = "rbf",
    lambda_reg: float = 0.01,
    kernel_params: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Estimate nonparametric causal effects via kernel ridge regression.

    The model solves: α = (K + λI)⁻¹ @ y
    where K is the kernel matrix, y is outcomes, λ is regularization.

    Args:
        parent_features: Feature matrix (n_samples, n_features)
        child_outcomes: Target vector (n_samples,)
        kernel_type: One of 'rbf', 'linear', 'matern'
        lambda_reg: L2 regularization parameter (default 0.01)
        kernel_params: Dict of kernel-specific params (e.g., {'gamma': 0.1})

    Returns:
        Dictionary with:
          - 'model': Fitted model (callable)
          - 'alpha': Regularization coefficients
          - 'K_train': Training kernel matrix
          - 'X_train': Training features (for inference)
          - 'kernel_type': Type of kernel used
          - 'kernel_params': Kernel hyperparameters
    """

    if len(parent_features) < 10:
        warnings.warn("Very small sample size (n < 10). Results may be unreliable.")

    if len(parent_features) < parent_features.shape[1] ** 2:
        warnings.warn(
            f"Sample size {len(parent_features)} < p² {parent_features.shape[1]**2}. "
            f"Kernel ridge regression may be numerically unstable. Consider regularization."
        )

    if kernel_params is None:
        kernel_params = {}

    # Compute kernel matrix
    if kernel_type == "rbf":
        K = RBFKernel(parent_features, parent_features, **kernel_params)
    elif kernel_type == "linear":
        K = LinearKernel(parent_features, parent_features)
    elif kernel_type == "matern":
        K = MaternKernel(parent_features, parent_features, **kernel_params)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    # Solve: α = (K + λI)⁻¹ @ y
    try:
        alpha = solve(K + lambda_reg * np.eye(len(parent_features)), child_outcomes)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"Kernel matrix singular. Try increasing lambda_reg or reducing features. Error: {e}"
        )

    # Define prediction function
    def predict(X_test):
        if kernel_type == "rbf":
            K_test = RBFKernel(X_test, parent_features, **kernel_params)
        elif kernel_type == "linear":
            K_test = LinearKernel(X_test, parent_features)
        elif kernel_type == "matern":
            K_test = MaternKernel(X_test, parent_features, **kernel_params)

        return np.dot(K_test, alpha)

    return {
        "model": predict,
        "alpha": alpha,
        "K_train": K,
        "X_train": parent_features,
        "kernel_type": kernel_type,
        "kernel_params": kernel_params,
        "lambda_reg": lambda_reg
    }


def EstimateDosetResponse_KernelRidgeRegression(
    parent_features: np.ndarray,
    child_outcomes: np.ndarray,
    treatment_col_idx: int,
    kernel_type: str = "rbf",
    lambda_reg: float = 0.01,
    dose_grid_size: int = 100
) -> Dict[str, any]:
    """
    Estimate dose-response curve (τ as function of treatment dose).

    Creates a grid of treatment values and estimates outcome at each dose,
    holding other features constant at training means.

    Args:
        parent_features: Feature matrix (n_samples, n_features)
        child_outcomes: Target vector
        treatment_col_idx: Column index of treatment variable
        kernel_type: Kernel type ('rbf', 'linear', 'matern')
        lambda_reg: Regularization
        dose_grid_size: Number of doses to evaluate (default 100)

    Returns:
        Dictionary with:
          - 'dose_grid': Grid of treatment values
          - 'tau_predictions': τ at each dose
          - 'dose_response_curve': Callable dose-response function
    """

    krr = EstimateCausalEffect_KernelRidgeRegression(
        parent_features, child_outcomes,
        kernel_type=kernel_type, lambda_reg=lambda_reg
    )

    # Create dose grid: treatment range
    treatment_min = parent_features[:, treatment_col_idx].min()
    treatment_max = parent_features[:, treatment_col_idx].max()
    dose_grid = np.linspace(treatment_min, treatment_max, dose_grid_size)

    # Construct test features: dose grid with other features at training mean
    X_test = np.tile(parent_features.mean(axis=0), (dose_grid_size, 1))
    X_test[:, treatment_col_idx] = dose_grid

    # Predict at each dose
    tau_predictions = krr["model"](X_test)

    return {
        "dose_grid": dose_grid,
        "tau_predictions": tau_predictions,
        "dose_response_curve": krr["model"],
        "treatment_col_idx": treatment_col_idx,
        "feature_means": parent_features.mean(axis=0)
    }


if __name__ == "__main__":
    # Example: Nonlinear causal relationship
    np.random.seed(42)

    n_samples = 200
    parent_features = np.random.uniform(-5, 5, (n_samples, 2))

    # Nonlinear outcome: Y = X₁² + 0.5*X₂ + ε
    child_outcomes = (
        parent_features[:, 0] ** 2 +
        0.5 * parent_features[:, 1] +
        np.random.normal(0, 0.5, n_samples)
    )

    print("=" * 70)
    print("EXAMPLE: Kernel Ridge Regression τ Estimation")
    print("=" * 70)

    # Fit model
    krr = EstimateCausalEffect_KernelRidgeRegression(
        parent_features, child_outcomes,
        kernel_type="rbf", lambda_reg=0.01
    )

    print("\n[Model Fitted]")
    print(f"  Kernel: {krr['kernel_type']}")
    print(f"  Regularization (λ): {krr['lambda_reg']}")
    print(f"  Training samples: {len(parent_features)}")

    # Estimate dose-response for first feature
    dose_response = EstimateDosetResponse_KernelRidgeRegression(
        parent_features, child_outcomes,
        treatment_col_idx=0,
        kernel_type="rbf",
        lambda_reg=0.01,
        dose_grid_size=50
    )

    print("\n[Dose-Response Curve (Feature 0)]")
    print(f"  Dose range: [{dose_response['dose_grid'].min():.2f}, "
          f"{dose_response['dose_grid'].max():.2f}]")
    print(f"  τ range: [{dose_response['tau_predictions'].min():.2f}, "
          f"{dose_response['tau_predictions'].max():.2f}]")

    # Estimate derivative (marginal effect at mean dose)
    mean_dose_idx = len(dose_response['dose_grid']) // 2
    epsilon = 0.01
    tau_before = dose_response['tau_predictions'][max(0, mean_dose_idx - 1)]
    tau_after = dose_response['tau_predictions'][min(len(dose_response['dose_grid'])-1, mean_dose_idx + 1)]
    marginal_effect = (tau_after - tau_before) / (2 * epsilon)

    print(f"\n[Marginal Effect at Mean]")
    print(f"  dτ/dtreatment ≈ {marginal_effect:.4f}")

    # Prediction on test set
    X_test = np.array([[-2.0, 1.0], [0.0, 0.0], [2.0, -1.0]])
    tau_test = krr["model"](X_test)

    print(f"\n[Predictions on Test Set]")
    for i, x in enumerate(X_test):
        print(f"  X={x} → τ={tau_test[i]:.4f}")
