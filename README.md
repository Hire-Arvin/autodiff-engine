# Autodifferentiation Engine from Scratch

Built a fully functional reverse-mode autodifferentiation engine in Python — analogous to PyTorch's autograd — without using any automatic differentiation library.

**[View Report →](https://hire-arvin.github.io/autodiff-engine/autodifferentiation_engine.html)**

## What's Inside

- **`BearTensor`** — a tensor class that tracks operations and builds a dynamic computation graph at runtime
- **Backpropagation** implemented via topological sort of the computation graph, computing exact gradients for arbitrary expressions
- **Four optimizers** implemented from scratch: SGD, Momentum, Adam, and Muon (the 2024 optimizer behind NanoGPT speedrun records)
- **End-to-end training** on the Wine Quality dataset — a two-layer MLP cuts test MSE 37% vs. a mean-prediction baseline (0.65 → 0.41), confirming the full autodiff + optimizer + training loop pipeline

## Tech Stack

Python · NumPy · Matplotlib · NetworkX

## Files

| File | Description |
|------|-------------|
| `autodifferentiation_engine_2026-10-26.qmd` | Quarto source — full write-up with embedded code |
| `autodifferentiation_engine.html` | Self-contained rendered report (open in browser) |
| `utils.py` | Graph visualization helpers |
