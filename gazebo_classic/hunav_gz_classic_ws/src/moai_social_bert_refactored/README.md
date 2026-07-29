# moai_social_bert_refactored

This is a standalone training and evaluation project. The root layout has been simplified around `train.py`, `test.py`, `configs/`, `data/`, `output/`, `src/`, and `visualization/`, while the framework-specific implementations are split into `Social-BERT/` and `SPU-BERT/`.

For the Gazebo guidance-conditioned MGP pipeline, start with
[`GAZEBO_GUIDED_MGP.md`](GAZEBO_GUIDED_MGP.md).

## Structure

- `configs/`: Unified nested YAML runtime configuration
- `configs/sbert/`: Social-BERT canonical config
- `configs/spubert/`: SPU-BERT canonical config
- `data/raw/`: Raw dataset location
- `data/processed/`: Preprocessed dataset artifacts
- `output/`: Checkpoints and outputs organized by framework / dataset / split
- `configs/loader.py`: Nested YAML config loader
- `src/data_loader.py`: Framework-specific dataset and DataLoader construction
- `src/model.py`: Shared BERT / ViT backbones, encoder adapters, shared layers, and framework model exports
- `src/loss.py`: Shared loss and metric utilities
- `src/trainer.py`: Config-only CLI, training loop, evaluation loop, and token contract output
- `src/utils.py`: Checkpoint, scheduler, trainer base, and shared utilities
- `visualization/`: Sample trajectory visualization scripts
- `Social-BERT/sbert/`: Social-BERT implementation
- `SPU-BERT/spubert/`: SPU-BERT implementation

## Run

```bash
cd /path/to/moai_social_bert_refactored
python3 train.py --config configs/sbert/ethucy_sbert.yaml
python3 train.py --config configs/sbert/jrdb.yaml --dry_run
python3 train.py --config configs/spubert/ethucy_sbert.yaml
python3 test.py --config configs/spubert/ethucy_sbert.yaml
```

## Dry Run

```bash
python3 train.py --config configs/spubert/ethucy_sbert.yaml --dry_run
```

`--dry_run` prints the effective config and the current token contract.

## Notes

- Runtime parameters are loaded only from config YAML files.
- The CLI accepts only `--config` and `--dry_run`.
- Social-BERT encodes trajectory tokens with HF `BertModel(inputs_embeds=...)`.
- In the Gazebo config, SPU-BERT splits a 32x32 occupancy map into four
  16x16 flattened patch tokens, applies a linear scene-patch embedding, and
  encodes scene and trajectory tokens with the shared HF `BertModel`.
- The simplified `test.py` currently focuses on `finetune` evaluation.
