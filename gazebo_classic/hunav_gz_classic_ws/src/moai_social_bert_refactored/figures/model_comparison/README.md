# Model comparison

Both models use the same Gazebo episode split, architecture, optimizer,
learning rate, batch size, seed, and maximum epoch count.

| Test metric | Gazebo-only | ETH/UCY pretrain → Gazebo | Change |
|---|---:|---:|---:|
| ADE | 1.0208 m | 0.9513 m | 6.8% better |
| FDE | 2.1463 m | 2.0063 m | 6.5% better |
| GDE | 1.4466 m | 1.6598 m | 14.7% worse |
| Safe candidate rate | 95.8% | 90.9% | -4.9 percentage points |
| Selected-goal valid rate | 98.2% | 94.7% | -3.6 percentage points |
| Trajectory map-safe rate | 91.6% | 95.6% | +4.0 percentage points |
| Execution-valid rate | 89.8% | 90.2% | +0.4 percentage points |

ETH/UCY pretraining improves trajectory-shape prediction and map-safe TGP
paths, but does not directly teach the Gazebo guidance-conditioned MGP goal.
The next experiment should preserve the pretrained trajectory encoder while
increasing Gazebo adaptation of the MGP goal head, for example by using a
smaller encoder learning rate and a larger MGP-head learning rate.

