# scVI – Minimal PyTorch Reproduction

A clean, from-scratch PyTorch implementation of **scVI**
(*Single-cell Variational Inference*, Lopez et al.).

This repository reproduces the **core modeling ideas** of scVI:
- Separate latent variables for biology (`z`) and library size (`ℓ`)
- ZINB likelihood
- ELBO optimization
- End-to-end training in PyTorch

No scvi-tools. No abstractions.

---

## What’s implemented

- Encoder: `q(z, ℓ | x)`
- Decoder: `p(x | z, ℓ)` with **ZINB**
- Reparameterization trick
- KL regularization
- Minibatch training loop

This repo is intentionally minimal and pedagogical.

---

## Files

- `model.py` — encoder & decoder
- `ZINB.py` — ZINB log-likelihood
- `loss.py` — ELBO
- `train.py` — training loop
- `data.py` — synthetic count data
- `utils.py` — reparameterization

---

## Run

```bash
python train.py
