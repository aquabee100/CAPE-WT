# Strict-online protocol

CAPE-WT is evaluated by chronological block-prequential replay. At decision time `t`, outputs use only observations available at or before `t`.

## Decision cycle

1. Use trailing preprocessing only.
2. Check channel history availability.
3. Score the trailing 45-day window.
4. Train adaptive GBM on preceding history only.
5. Update warning and physical evidence.
6. Update NORMAL/WARNING/CONFIRMED state.
7. Never revise past decisions.

The decision step is 15 days. Adaptive GBM uses a 365-day lookback ending before the scoring window. The physical model uses the same trailing 45-day score window.

## Forbidden

- centered windows
- centered derivatives
- backward fill
- future-dependent interpolation decisions
- whole-record scaling
- future-informed threshold selection during replay
- cross-episode warning/confirmation pairing

## Verification

`verify_freeze_integrity.py` checks stored-artifact consistency.

`verify_online_causality.py` tests raw-data future invariance under hostile future continuations.

`test_causality.py` includes known-positive leakage controls.
