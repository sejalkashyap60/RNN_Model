# Changelog

## 2026-10-27

- Added fixed-window tokenization, vocabulary encoding, and the original sample corpus.
- Implemented a Vanilla RNN forward pass, stable softmax, cross-entropy gradients, BPTT, global norm clipping, and SGD in NumPy.
- Added per-epoch training callbacks and Streamlit training, prediction, and intermediate-state views.
- Added JSON and NPZ checkpoint save/load with metadata, array-key, and shape validation.
- Added finite-difference gradient, text processing, training, and checkpoint tests.
- Added setup, mathematics, architecture, and cloud deployment notes.
- Added deterministic train/validation/test token splits with windows isolated within each split.
- Added independent what-if runs, per-epoch validation tracking, held-out evaluation, and local experiment summary history.
- Added Experiment Lab controls, fair-comparison filtering, and tests for split boundaries, independent models, evaluation metrics, and history persistence.
- Numerical gradient checks now cover both bias vectors in addition to the three weight matrices.
- Added pipeline, RNN-cell, and shared-weight unrolled architecture views, token-synchronized controls, prediction autopsy, sequence and memory comparisons.
- Added tensor/parameter/gradient inspection and per-step logits and probability distributions.
- Added held-out top-1/top-5 accuracy, mean cross-entropy, perplexity, unknown-token counts, and an add-one bigram baseline.
- Added dataset inspection, per-experiment checkpoints, and the requested eleven-page navigation.
