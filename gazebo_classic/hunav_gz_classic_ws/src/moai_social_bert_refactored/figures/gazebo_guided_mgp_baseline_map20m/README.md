# Gazebo-only guided MGP baseline

## Dataset

- Source: 7 valid Gazebo recordings
- Quality-filtered samples: 1,493
- Episode-safe split: train 1,044 / validation 224 / test 225
- Episode groups: 122
- Split leakage: 0
- Local map: 20 m × 20 m, 32 × 32 cells, 4 patches
- Guidance point: final-goal direction, maximum radius 8 m

All 1,493 samples passed tensor-shape and finite-value validation. Both the
8 m guidance point and the 12-step trajectory endpoint are inside the local
map for every sample.

## Training

- Gazebo-only training from scratch
- Hidden size 256, 4 encoder layers, 4 attention heads, dropout 0.1
- Batch size 32
- AdamW learning rate 5e-5
- Maximum 30 epochs, early-stopping patience 7
- Best validation checkpoint: epoch 29

## Independent test result

| Metric | Result |
|---|---:|
| ADE | 1.0208 m |
| FDE | 2.1463 m |
| GDE | 1.4466 m |
| Safe candidate rate | 95.8% |
| Selected-goal valid rate | 98.2% |
| Trajectory map-safe rate | 91.6% |
| Execution-valid rate | 89.8% |
| Samples with no valid MGP candidate | 4 / 225 |

The previous 8 m × 8 m map covered only ±4 m from the robot. Most 12-step
endpoints and all 8 m guidance points were outside that map, so the runtime
unknown/out-of-bounds filter rejected otherwise plausible MGP candidates.
Expanding the map to the original SPU-BERT scene range of ±10 m raised the
execution-valid rate from 6.7% to 89.8%.

## Reproduction

```bash
cd /home/kistmnl/social_nav/hunavsim_containers

./scripts/prepare_dataset.sh \
  gazebo_classic/hunav_gz_classic_ws/moai_recordings/corridor_low.pkl \
  training_corridor \
  corridor_low

./scripts/train.sh
```

`prepare_dataset.sh` merges every processed run already present under
`data/processed/gazebo/runs`, recreates the episode-safe split, and validates
the model input contract.

