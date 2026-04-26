# FK-Guided Causal Effect Estimation for Relational Deep Learning

## Overview

This research synthesis operationalizes foreign key (FK)-guided causal effect (τ) estimation for relational deep learning. The framework grounds relational learning in causal theory by exploiting the semantic directionality that FKs encode (parent → child relationships).

**Key Deliverables:**
- Comprehensive literature synthesis (research_out.json)
- Technical report (TECHNICAL_REPORT.md, ~1800 words)
- Three τ estimation strategies with pseudocode and implementations
- FK→causality validity framework (3 operationalizable checks)
- Five Python reference implementations
- Full worked example (e-commerce customer orders)

---

## File Structure

### Core Research Artifacts

| File | Purpose |
|------|---------|
| **research_out.json** | Structured research findings with 19 citations. Includes executive summary, literature synthesis (SCM, causal GNNs, RelBench), three τ strategies, worked example, validity framework, limitations. |
| **TECHNICAL_REPORT.md** | Comprehensive technical report (~1800 words). Discusses FK-causality gap, distinction from causal GNNs, Phase 1-7 of research (synthesis, operationalization, worked example, validity, implementation). Audience: DL researchers familiar with GNNs and causal inference. |

### Python Reference Implementations

| File | Strategy | Purpose |
|------|----------|---------|
| **tau_estimation_linear.py** | Strategy A | Linear regression τ estimator. ATE, CATE, ITE methods. Includes validation (R², RMSE, VIF). Example with synthetic e-commerce data. |
| **tau_estimation_kernel.py** | Strategy C | Kernel ridge regression τ estimator. RBF, linear, Matérn kernels. Dose-response curve estimation. Example with nonlinear synthetic data. |
| **fk_validity_check.py** | Framework | Three checks: temporal ordering, domain knowledge alignment, effect consistency (k-fold). Full operationalization with pseudocode. |
| **interventional_loss.py** | Training | PyTorch interventional loss combining observational + causal objectives. `InterventionalTrainingLoop` class for end-to-end training. |
| **example_ecommerce.py** | Full Pipeline | Complete worked example (Customers + Orders tables). Demonstrates all 7 phases: data generation, τ estimation, validation, temporal check, domain alignment, consistency, interventional training. |

---

## Quick Start

### 1. Read the Research Summary

Start with **research_out.json** for structured findings or **TECHNICAL_REPORT.md** for prose narrative:

```bash
# View high-level summary
head -100 research_out.json | jq '.summary'

# Read technical report
less TECHNICAL_REPORT.md
```

### 2. Run the Worked Example

```bash
cd /path/to/workspace
python example_ecommerce.py
```

**Output:**
- Synthetic Customers + Orders data
- Estimated causal effects (τ)
- Model validation (R², RMSE, VIF)
- Temporal ordering check
- Domain knowledge alignment
- Effect consistency (5-fold CV)
- Interventional training (20 epochs)
- Final validity verdict

### 3. Understand the Three Strategies

**Choose based on your problem:**

```python
# Strategy A: Fast, interpretable
from tau_estimation_linear import EstimateCausalEffect_LinearRegression
tau = EstimateCausalEffect_LinearRegression(customers, orders, 'order_amount', ['age', 'loyalty'], method='ate')

# Strategy C: Flexible, nonlinear
from tau_estimation_kernel import EstimateCausalEffect_KernelRidgeRegression
krr = EstimateCausalEffect_KernelRidgeRegression(parent_features, outcomes, kernel_type='rbf', lambda_reg=0.01)
tau_predictions = krr['model'](test_features)
```

### 4. Validate FK→Causality

```python
from fk_validity_check import CheckTemporalOrdering, ValidateFK_Causality, CheckEffectConsistency

# Check 1: Temporal ordering
temporal = CheckTemporalOrdering(customers, orders, 'customer_id')
print(f"Temporal: {temporal['status']} ({100*temporal['pct_valid']:.1f}% valid)")

# Check 2: Domain alignment
domain_knowledge = {'age': (1, (0, 5)), 'loyalty_score': (1, (0, 10))}
domain = ValidateFK_Causality(tau_estimates, domain_knowledge)
print(f"Domain: {domain['status']} (credibility: {domain['overall_credibility']:.1%})")

# Check 3: Effect consistency
consistency = CheckEffectConsistency(customers, orders, 'order_amount', ['age', 'loyalty'], n_splits=5)
print(f"Consistency: {consistency['status']} ({100*consistency['consistency_pct']:.1%} stable)")
```

### 5. Train with Interventional Loss

```python
import torch
from interventional_loss import InterventionalLoss, InterventionalTrainingLoop

model = YourRDLModel(...)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = InterventionalLoss(lambda_weight=0.5)  # 50% obs, 50% interventional

loop = InterventionalTrainingLoop(model, optimizer, loss_fn, device='cuda')

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        loss = loop.training_step(
            X_batch, y_batch,
            intervention_attr_idx=0,      # intervene on feature 0 (e.g., age)
            intervention_delta=5.0,       # +5 units
            tau_empirical=2.0             # expected effect
        )
```

---

## Key Concepts

### Three τ Estimation Strategies

| Strategy | When | Speed | Accuracy | Interpretability |
|----------|------|-------|----------|------------------|
| **A: Linear** | Large n, linear assumed | Fast | Moderate | High |
| **B: CARL-Hybrid** | Small schema, confounders | Slow | High | High |
| **C: Kernel** | Nonlinear, large n | Slow | High | Low |

**ITE vs. ATE:** Favor **ATE** by default (lower variance). Use CATE if heterogeneity significant. Avoid ITE unless n >> 50K.

### FK→Causality Validity Framework

Three checks (all required):

1. **Temporal Ordering (T1):** parent.created_at < child.created_at for ≥95% of pairs
2. **Domain Alignment (D1+D2):** ≥60% of attributes pass sign + magnitude validation
3. **Consistency (C1):** ≥80% of attributes show 100% sign agreement across k-folds

**Success Criterion:** T1=PASS AND (D1=PASS OR D2=PASS) AND C1=PASS

### Distinction: FK-Causality vs. Causal GNNs

| Aspect | FK-Causality | Causal GNNs |
|--------|--------------|------------|
| **Mechanism** | Exploit pre-declared FK structure | Learn causal structure from shifts |
| **Assumption** | Temporal ordering, no latent confounders | Distribution shifts exist |
| **Cost** | Low (d-separation or regression) | High (optimization) |
| **Applicability** | Single-domain effect estimation | OOD generalization |
| **Complementarity** | FK structure seeds GNN initialization | GNN validates FK-derived effects |

---

## References & Citations

All claims are grounded in cited literature:

- **CARL [1]:** Declarative causal inference via Datalog (SIGMOD 2020)
- **RelFCI [9]:** Causal discovery with latent confounders (JMLR 2025)
- **CIGA [5]:** Causal GNNs for OOD robustness (NeurIPS 2022)
- **DIR-GNN [11]:** Invariant rationales for GNNs (ICLR 2022)
- **RelBench [2, 13]:** Relational deep learning benchmark (NeurIPS 2024, ICML 2025)
- **RelGNN [14]:** Composite message passing via FKs (ICML 2025)
- **Kernel Methods [15]:** Nonparametric causal dose-response (arXiv 2020)
- **Metalearners [16]:** Heterogeneous treatment effects (PNAS 2019)

See research_out.json and TECHNICAL_REPORT.md for full references.

---

## Limitations & Future Work

### Known Limitations

1. **Confounding:** Unmeasured confounders bias τ estimates (untestable assumption)
2. **Nonlinearity:** Linear τ fails for nonlinear relationships (use Strategy C)
3. **Scalability:** CARL exponential in schema size (use sampling for large schemas)
4. **Causal Sufficiency:** All methods except RelFCI assume no latent confounders
5. **FK Semantics:** Foreign keys encode directionality, not always causality (validation required)

### Future Directions

1. **FK Priors for GNNs:** Seed CIGA/DIR with RelFCI-discovered confounders
2. **Multi-Hop Causality:** Extend to grandparent→parent→child paths
3. **Automated Domain Knowledge:** LLM-based domain expectation generation
4. **Interventional Pretraining:** Augment contrastive learning with τ estimates
5. **Temporal Relational Data:** Dynamic causal models for time-series
6. **Causal Abstraction:** Align embeddings with causal variables (CAL framework)

---

## Dependencies

### Python Packages

```
numpy >= 1.20
pandas >= 1.3
scikit-learn >= 1.0
scipy >= 1.7
torch >= 1.10        # for interventional_loss.py
pytorch-lightning    # optional, for distributed training
```

### Installation

```bash
pip install numpy pandas scikit-learn scipy torch
# Optional:
pip install pytorch-lightning matplotlib seaborn
```

---

## Testing

### Run All Examples

```bash
# Example 1: Linear regression strategy
python tau_estimation_linear.py

# Example 2: Kernel ridge regression strategy
python tau_estimation_kernel.py

# Example 3: Validity framework
python fk_validity_check.py

# Example 4: Interventional loss
python interventional_loss.py

# Example 5: Full pipeline
python example_ecommerce.py
```

### Expected Output

Each script outputs:
- Method explanation
- Synthetic data generation
- Key results (τ estimates, validation metrics, validity checks)
- Interpretation and caveats

---

## Citation

If using this research, cite:

```bibtex
@article{aiiInventor2026,
  title={FK-Guided Causal Effect Estimation for Relational Deep Learning},
  author={Claude (AI Inventor)},
  year={2026},
  month={April},
  note={Research synthesis: 19 citations, 3 strategies, 5 implementations}
}
```

---

## Contact & Questions

For questions about:
- **Research synthesis:** See research_out.json
- **Technical details:** See TECHNICAL_REPORT.md
- **Implementation:** See specific .py files with docstrings
- **Examples:** Run example_ecommerce.py with modifications

---

## Changelog

- **2026-04-26:** Initial release
  - research_out.json (structured findings, 19 citations)
  - TECHNICAL_REPORT.md (comprehensive technical report)
  - 5 Python implementations (tau_estimation_linear.py, tau_estimation_kernel.py, fk_validity_check.py, interventional_loss.py, example_ecommerce.py)
  - README.md (this file)

---

**Last Updated:** 2026-04-26  
**Status:** Complete  
**Word Count:** Report ~1800 words + implementations ~1500 lines of code

