# Foreign Key Causal Effect Estimation in Relational Databases

## Summary

This research synthesizes relational causal discovery (CARL [1], RelFCI [5]), causal GNNs (CIGA [3]), and relational deep learning (RelBench [4], RelGNN [7]) to operationalize foreign key (FK)-guided causal effect (τ) estimation in multi-table databases. Relational FK causality exploits schema-declared directionality (parent→child) semantically; causal GNNs learn invariance from distribution shifts topologically—complementary approaches. Three τ estimation strategies: (1) linear regression for data-efficient empirical estimation [12]; (2) CARL integration for confounder-aware symbolic discovery via d-separation [1]; (3) kernel ridge regression for nonparametric dose-response curves [14]. A validity framework checks temporal ordering (parent.created_at < child.created_at for ≥95% rows), domain knowledge alignment (≥60% of attributes pass sign/magnitude checks), and effect consistency across folds. Worked example: e-commerce customer orders demonstrate τ calculation and interventional loss integration. Key limitations: unmeasured confounding [17, 19], nonlinearity [15], CARL scalability [5], and untestable causal sufficiency. Recommendations: default to ATE over ITE for interventional loss; use CATE when heterogeneous effects expected; conduct sensitivity analysis for unmeasured confounders. Temporal validation essential; FK structure without temporal precedence is insufficient for causal interpretation.

## Research Findings

**Foreign Key-Guided Causal Effect Estimation for Relational Deep Learning**

**EXECUTIVE SUMMARY**

Foreign key relationships in relational databases encode parent→child directionality. This research operationalizes causal effect (τ) estimation exploiting this structure, grounded in structural causal models (SCMs), with three empirically-grounded strategies and a practical validity framework.

**CORE FINDING 1: DISTINCTION BETWEEN FK CAUSALITY AND CAUSAL GNNs**

CIGA [3] and DIR [8] learn causal features via **invariance across distribution shifts** (topological causality). They require or generate multiple training environments and identify features robust to distributional changes.

Relational FK causality exploits **pre-declared schema directionality** (semantic causality). FKs encode parent→child links where temporal precedence is typically assumed. This is fundamentally different: causal GNNs assume distributional heterogeneity; FK approaches assume temporal ordering and schema semantics. **Complementarity:** FK structure could seed causal GNN initialization, improving sample efficiency. GNN invariance could validate FK-derived causal edges under distribution shift.

**CORE FINDING 2: THREE τ ESTIMATION STRATEGIES WITH TRADE-OFFS**

**Strategy A: Linear Regression** [6, 12] — Fits Y = β₀ + Σ βᵢ × parent_attrᵢ + ε. Direct coefficient interpretation as τ. Strength: interpretable, fast (OLS), works with small n. Limitation: assumes linearity.

**Strategy B: CARL-Hybrid** [1, 5] — Two-stage: (1) CARL/FCI d-separation discovers confounding set Z; (2) fit Y ~ parent_attrs[Z]. More accurate, removes confounders via graphical criteria [2, 3]. Limitation: computationally expensive on large schemas [5].

**Strategy C: Kernel Ridge Regression** [14, 15] — Nonparametric ŷ = K(K + λI)⁻¹y handles nonlinearity without functional form. Limitation: requires n > p²; O(n³) complexity.

**CORE FINDING 3: ATE vs. CATE vs. ITE**

ATE (Average Treatment Effect): single scalar τ_avg. Low variance, default choice. CATE (Conditional ATE): τ(z) varies by subgroup; use when child attributes predict heterogeneity. ITE (Individual Treatment Effect): per-row τⱼ; high variance, overfits unless n > 50K [16]. **Recommendation:** Default ATE for interventional loss; CATE as secondary objective; avoid ITE unless massive data.

**CORE FINDING 4: FK→CAUSALITY VALIDITY FRAMEWORK**

Three essential checks determine when FK-based reasoning is sound:

1. **Temporal Ordering** [12]: parent.created_at < child.created_at for ≥95% of rows. Temporal precedence validates directionality.

2. **Domain Knowledge Alignment** [3]: Learned τ sign/magnitude match expert expectations. Compute credibility = (sign_match + magnitude_ok) / 2. Target: ≥60% of parent columns pass.

3. **Effect Consistency** [16]: Refit on k-fold splits. Verify τ signs remain stable. If signs flip across folds, effect is spurious (confounding). Target: ≥80% attributes show 100% consistency.

**CORE FINDING 5: LITERATURE SYNTHESIS**

CARL [1] provides d-separation framework for relational causal discovery. RelFCI [5] extends to latent confounders with theoretical guarantees (sound, complete). RelBench [4] provides 11 datasets (22M+ rows, 66 tasks) as empirical testbed. RelGNN [7] demonstrates FK directionality naturally encodes in GNN architecture via atomic routes. Linear regression [12] is asymptotically efficient for ATE; kernel methods [14, 15] handle nonlinearity; double ML [17] extends inference to high-dimensional covariates while preserving validity. Pearl's backdoor criterion [2, 3] grounds confounder adjustment. Sensitivity analysis bounds bias from unmeasured confounders [17, 19].

**CORE FINDING 6: CRITICAL LIMITATIONS**

**Unmeasured Confounding** [17, 19]: Cannot identify confounders not in database. Untestable assumption. Mitigation: RelFCI marks latent confounders; sensitivity analysis; instrumental variables.

**Nonlinearity** [15]: Linear τ assumes E[Y|Parent] linear. Mitigation: residual diagnostics; kernel methods/causal forests if nonlinearity evident.

**Scalability** [5, 18]: CARL d-separation exponential in hops. Large schemas may need sampling-based approximations.

**FK Not Always Causal**: Many-to-many links (order→item) have ambiguous causality. Domain validation essential; FK structure ≠ automatic causality.

**Missing Temporal Data**: Without timestamps, cannot verify temporal ordering. Conservative: assume schema DAG order; document assumption; sensitivity analysis.

**WORKED EXAMPLE: E-COMMERCE CUSTOMER ORDERS**

Schema: Customers (age, loyalty_score) → Orders (order_amount) via FK.

Linear fit: order_amount = 50 + 2.0×age + 3.5×loyalty_score − 0.5×account_age + ε

Intervention: age + 5 years → τ = 2.0 × 5 = $10 average increase.

Interventional loss: L_total = MSE(ŷ − y) + 0.5 × MSE(Δŷ − $10)

Validation: ✓ Temporal (95%+ customers before orders); ✓ Domain (expert expects +$0–$20 range).

**KEY RECOMMENDATIONS**

1. Start with Strategy A (linear) for interpretability; check residuals for linearity.
2. Adopt validity framework (temporal, domain, consistency) before claiming causal effects.
3. Use ATE for interventional loss; CATE only if heterogeneity demonstrated.
4. Conduct sensitivity analysis for unmeasured confounding; document assumptions transparently.
5. Integrate with causal GNNs: FK priors could improve CIGA/DIR initialization.

## Sources

[1] [Causal Relational Learning (CARL)](https://arxiv.org/abs/2004.03644) — Declarative language for causal inference on relational data; enables d-separation queries to identify confounders in multi-table schemas.

[2] [Backdoor Criterion for Causal Effect Identification](https://datascience.oneoffcoder.com/backdoor-criterion.html) — Pearl's graphical criterion for confounder adjustment; identifies when variable set Z blocks non-causal paths between treatment and outcome.

[3] [Learning Causally Invariant Representations for OOD Generalization on Graphs (CIGA)](https://proceedings.neurips.cc/paper_files/paper/2022/file/8b21a7ea42cbcd1c29a7a88c444cce45-Paper-Conference.pdf) — SCM-based framework for learning invariant subgraphs; causal GNNs learn via invariance (topological), distinct from FK-declared directionality (semantic).

[4] [RelBench: A Benchmark for Deep Learning on Relational Databases](https://arxiv.org/abs/2407.20060) — 11 large-scale relational databases (22M+ rows, 66 tasks) across medical, e-commerce, ERP domains; empirical testbed for FK→causality validation.

[5] [Relational Causal Discovery with Latent Confounders (RelFCI)](https://arxiv.org/abs/2507.01700) — Sound and complete causal discovery for relational data with latent confounders; introduces LRCMs, MAAGGs, PAAGGs for handling unmeasured confounders.

[6] [The Unreasonable Effectiveness of Linear Regression for Causal Inference](https://matheusfacure.github.io/python-causality-handbook/05-The-Unreasonable-Effectiveness-of-Linear-Regression.html) — Linear regression asymptotically efficient for ATE estimation; provides interpretable coefficients and practical foundation for τ estimation.

[7] [RelGNN: Composite Message Passing for Relational Deep Learning](https://arxiv.org/abs/2502.06784) — Introduces atomic routes (FK-derived paths) and composite message passing; demonstrates FK directionality naturally encodes in GNN architecture.

[8] [Discovering Invariant Rationales for GNNs (DIR-GNN)](https://hexiangnan.github.io/papers/iclr22-invariant-gnn.pdf) — Identifies invariant features via interventions on training distributions; causal GNN approach fundamentally differs from FK-declared structural causality.

[12] [Regression-based Estimation of Treatment Effects](https://sites.stat.columbia.edu/gelman/arm/chap9.pdf) — Practical guide to linear regression for causal effect estimation; conditional expectation framework for treatment effects.

[13] [PC Algorithm and Constraint-based Causal Discovery](https://www.pywhy.org/dodiscover/dev/constraint_causal_discovery.html) — Constraint-based discovery via d-separation and conditional independence tests; basis for CARL/RelFCI methods.

[14] [Kernel Methods for Causal Dose-Response Curves](https://arxiv.org/abs/2010.04855) — Kernel ridge regression for nonparametric causal dose-response; closed-form RKHS solutions with consistency guarantees.

[15] [Causal Inference with Continuous Treatments and Heterogeneous Effects](https://arxiv.org/html/2604.13410) — Two-stage kernel ridge regression for continuous treatment effects; practical guidance for nonparametric methods necessity.

[16] [Metalearners for Heterogeneous Treatment Effects](https://www.pnas.org/doi/10.1073/pnas.1804597116) — Metalearner framework for CATE and ITE; guidance on ATE vs. ITE trade-offs (sample efficiency, overfitting risk).

[17] [Confounding Bias in Observational Studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC8786092/) — Treatment of confounding; identifies unmeasured confounders as fundamental limitation; methods for sensitivity analysis.

[18] [Scalable Causal Discovery Algorithms](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2019.00524/full) — Scalable variants of FCI and PC algorithms; sampling-based approximations for large graphs; computational scalability solutions.

[19] [Causal Sufficiency Assumption and Its Violations](https://www.tandfonline.com/doi/full/10.1080/00273171.2025.2507742) — Formal treatment of causal sufficiency (no latent confounders); shows assumption untestable and often violated in practice.

## Follow-up Questions

- How sensitive are τ estimates to linearity violations? Characterize when kernel methods become necessary across RelBench tasks.
- For multi-hop FK paths (customer→order→product), does causal composition hold? Can we estimate multi-hop τ via path decomposition?
- Can FK priors improve causal GNN sample efficiency? Validate whether RelFCI-discovered confounders reduce CIGA/DIR sample complexity.
- How do temporal dynamics affect FK causality validity? Do methods extend to temporal relational data with evolving relationships?
- Can domain knowledge bounds be automatically inferred from τ distributions? Can sensitivity analysis inform domain_knowledge dict?

---
*Generated by AI Inventor Pipeline*
