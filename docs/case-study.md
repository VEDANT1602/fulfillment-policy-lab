# Before asking whether an offer wins, ask whether the log can answer

## Topic: marketplace decision evaluation

Fulfillment is a sequence of decisions under constraints. This project focuses on one decision: offer selection. A higher incentive may improve completion but costs utility; the best action depends on the order context. A test of this decision should compare policies on the same target population, rather than compare the particular orders each historical action happened to receive.

## The usual shortcut and its failure

Looking at mean completion by incentive ignores why each action was chosen. Even in this randomized simulator, exploration probabilities depend on backlog: expensive offers appear more frequently during difficult conditions. A direct action-by-action comparison therefore answers a different question from evaluating a new policy across all incoming contexts.

IPS reweights logged outcomes by target probability divided by logging probability. It can be noisy when target actions were rarely explored. Clipping bounds observed contributions but introduces bias. Self-normalization uses the sum of weights as denominator; it is a separate finite-sample estimator, not a claim that exploration was balanced.

## How I structured the investigation

I generated 60 days, four zones, and 40 offers per zone-day. Within each group, a latent demand shock correlates outcomes. Days 0-29 train the nuisance model, 30-44 support selection, and 45-59 provide the untouched test set. Logging probabilities are fixed, known context-dependent randomization probabilities. The fourth possible action always has zero support.

I defined five candidate policies before scoring test outcomes: no incentive, modest offer, high offer, load-sensitive, and distance-sensitive. Validation reward predictions use four `GroupKFold` folds: each held-out zone-day is excluded from its own nuisance fit, while the earlier training period is always available. After selecting the highest validation DR point estimate, I fit one final reward model on training plus validation logs and use it for test predictions. The final reward model sees no test outcomes.

For a target policy pi and known logging policy p:

```text
w_i = pi(a_i | x_i) / p(a_i | x_i)
IPS = mean(w_i * r_i)
SNIPS = sum(w_i * r_i) / sum(w_i)
clipped IPS = mean(min(w_i, 5) * r_i)
DR score_i = sum_a pi(a | x_i) * qhat(x_i, a)
             + w_i * (r_i - qhat(x_i, a_i))
```

The DR estimator uses the untruncated weights. The reward model is not assumed correct. Identification here rests on the known randomized logger and positive support, not on a promise that gradient boosting recovers the simulator.

The gate checks positive target mass against every logging probability, including actions not sampled in this particular log. A theoretical importance ratio above 20 fails the gate. Observed effective sample size must be at least 30. Those are explicit demonstration thresholds, not universal sufficiency guarantees. Zero observed matching actions also fails even when a caller requests a zero ESS threshold.

## What I kept in the result

The frozen `modest_offer` policy has test DR utility 0.6554, with a zone-day bootstrap interval [0.6129, 0.6967]. Its paired difference from no incentive is +0.1533, interval [+0.1071, +0.1946]. The paired interval resamples whole zone-days and evaluates the difference within the same resampled groups. It uses 1,000 seeded percentile bootstrap replicates and keeps the trained nuisance model fixed.

This does not make the selection optimal. The held-out simulator oracle prefers `high_offer`, whose expected utility is 0.7056 versus 0.6718 for `modest_offer`. The selected policy's DR estimate is 0.0163 away from its oracle expected value. I report the oracle after selection as a diagnostic, and retain the miss as evidence of uncertainty in policy selection.

The support failure is equally important. `unsupported_premium` cannot be evaluated because the target requests an action with zero historical probability. The evaluator returns a refusal rather than a number. A reward model could still emit a prediction for that unseen action, but extrapolation is not counterfactual evidence.

## Impact and what this changes

The useful output is a repeatable decision record: the logging contract, the frozen candidate selection, estimator disagreement, support diagnostics, cluster uncertainty, and an explicit boundary on what cannot be estimated. A production review could use that record to decide whether there is enough evidence to design an online trial.

This is not a delivery simulation with capacity coupling, an experiment on human shoppers, or a deployed service. Marketplace interference and carryover require a different design; cluster resampling alone cannot remove those effects. Statistical intervals do not certify rollout safety. I would next define experiment units, washout windows, minimum detectable effect, stopping rules, and service guardrails before collecting trial data.
