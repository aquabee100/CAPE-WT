# Reproducibility

## Primary replay

From `src/wt_cm/`:

```bash
python run_online.py --fresh
```

This regenerates the strict-online filter, adaptive GBM, thermal, episode, availability, and headline outputs from the authorised primary SCADA archive.

## Causality checks

```bash
python verify_freeze_integrity.py
python verify_online_causality.py
python verify_deep_gbm_causality.py --help
python test_causality.py
```

The protocol is in `docs/protocol/ONLINE_PROTOCOL.md`.

## Results

Corrected manuscript evidence is in `results/authoritative/`. Derived manuscript support is in `results/manuscript_support/`. Regenerated outputs are written to `results/runtime/`. See `results/MANUSCRIPT_EVIDENCE.csv`.

## Verification

```bash
python verify.py
```

Without restricted data, data-dependent checks are skipped.
