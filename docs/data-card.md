# Data card

- **Origin:** original NumPy simulator, seed 1602. No external or employer dataset.
- **Population:** fictional grocery offer events; 60 days x four zones x 40 events = 9,600 rows.
- **Decision-time features:** distance (km), basket item count, normalized backlog, hour sine/cosine, normalized store friction.
- **Actions:** 0=no incentive, 1=modest, 2=high, 3=premium. Action 3 is never logged; its presence is a deliberate unsupported-policy test.
- **Logging:** p=[0.35,0.45,0.20,0] above backlog 0.55; otherwise p=[0.70,0.23,0.07,0]. Actions are independently drawn conditional on context.
- **Outcome:** Bernoulli completion with logistic probability, minus action costs [0,0.035,0.09,0.20]. Costs are arbitrary normalized utility units. They do not map to real currency or worker incentives.
- **Dependence:** an independent latent shock per zone-day shifts completion probability. This motivates cluster resampling; it is not interference or capacity coupling.
- **Split:** 4,800 training events / 120 groups; 2,400 validation / 60 groups; 2,400 test / 60 groups. Entire zone-days remain within one split.
- **Oracle:** a separate diagnostic file records expected reward for all actions. Training, predefined policies, and validation selection do not read it.
- **Restrictions:** synthetic fixture evidence only. No real population generalization, learned fairness, causal workplace claim, or calibrated business forecast is supported.
- **Privacy:** no real people, locations, orders, phone numbers, emails, or confidential records.

`scripts/run.py` regenerates both CSVs with explicit LF row endings and records their SHA-256 hashes. The same synthetic environment must not be described as an independent real-world validation dataset.
