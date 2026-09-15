# CAPE-WT

**Causal online condition monitoring for wind-turbine gearboxes.**

CAPE-WT separates early warning from maintenance-facing escalation using a filter-pressure channel, an adaptive thermal-residual model, and grey-box thermal corroboration.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd src/wt_cm
python run_online.py
```

Windows activation: `.venv\\Scripts\\activate`.

The primary replay requires the restricted archive at:

`src/wt_cm/data/MANF2017-2022.pkl`

## Repository

- `src/wt_cm/` — CAPE-WT analyses and audits
- `src/kelmarsh/` — independent transfer analysis
- `results/authoritative/` — manuscript evidence tables
- `results/manuscript_support/` — derived manuscript support
- `results/runtime/` — regenerated outputs (ignored by Git)
- `docs/protocol/` — strict-online protocol

See `REPRODUCIBILITY.md` and `DATA_ACCESS.md`.

Evidence index: `results/MANUSCRIPT_EVIDENCE.csv`.
