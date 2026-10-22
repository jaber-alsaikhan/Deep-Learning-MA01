# Lab 8 — PyTorch: Writing the Training Loop Again, On Purpose

**Week 8 · Deep Learning Frameworks**
Building and training neural networks using PyTorch.

| | |
|---|---|
| **Prerequisites** | Lab 5 (the hand-written training loop) and Lab 7 (Keras) |
| **Reference** | Ekman, *Learning Deep Learning*, Appendix J · official [PyTorch](https://pytorch.org/docs/stable/index.html) documentation |

## Files

| File | Description |
|---|---|
| [arti402_Lab8_2240006539.ipynb](./arti402_Lab8_2240006539.ipynb) | The completed lab notebook, with all outputs |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook (same version as Labs 6–7: `draw_overfitting_curves`, `draw_decision_boundaries`) |

Keep both files in the same folder. `arti402_figures.py` is imported directly by the notebook.
The notebook needs PyTorch (`pip install torch`). It was run with PyTorch 2.14.0 (CPU) on
Python 3.13.

## What this lab covers

Keras gave us layers, automatic gradients **and** a training loop. PyTorch gives the first two
and leaves the loop to you, so this lab is Lab 5's loop written again with framework parts.
The data is the same as Labs 6–7: **40 spiral samples per class** for training and 100 per class
for validation. The network is `2 → 128 (ReLU) → 3`, trained with Adam (lr 0.02).

| Lab 5 (NumPy) | Lab 7 (Keras) | Lab 8 (PyTorch) |
|---|---|---|
| `Layer_Dense(2, 128)` | `layers.Dense(128)` | `nn.Linear(2, 128)` |
| your cross-entropy | `loss="sparse_categorical_crossentropy"` | `F.cross_entropy` (takes logits) |
| every `backward()` | automatic | `loss.backward()` |
| `get_batches` | `fit(batch_size=32)` | `DataLoader(..., batch_size=32)` |
| your training loop | `model.fit(...)` | **your training loop, again** |

* **Exercise 1: autograd vs a numerical gradient.** The central-difference check from Lab 5, run
  against `loss.backward()` in float64. Max difference: **3.98e-11**.
* **Exercise 2: `nn.Module`.** The Lab 6 network as a `Net` class with optional dropout. It has
  **771** parameters, the same as Keras. `nn.Linear` stores weights transposed, as `(out, in)`.
  `forward` returns logits with no softmax, because `F.cross_entropy` applies it internally.
* **Exercise 3: the training step.** `zero_grad` → forward + loss → `backward` → `step`, with a
  manual shuffle. The loss goes from 1.201 to 1.003 in 10 epochs.
* **Exercise 4: `evaluate`.** `model.eval()` plus `torch.no_grad()`, with accuracy from `argmax`.
* **Exercise 5: `DataLoader`.** The same loop over `DataLoader(TensorDataset(...))`, giving 4
  batches per epoch. Loss after 10 epochs: 0.996.
* **Exercise 6: `run()`.** One reusable function that trains with weight decay (an optimizer
  argument) and dropout (a layer), and records history every 5 epochs.
* **Checkpoints 1–2.** Written answers in the notebook: the chain rule, gradient accumulation and
  `zero_grad`, `eval()` vs `no_grad()`, and why `argmax` gives the same answer with or without
  softmax.

## Assessment and results

The handout says the graded Assessment is at the end, but no assessment cell was included. I added
one in the style of Lab 7's: the same Lab 6 regularization comparison, using `run()`. Each model is
trained for 1,500 epochs, except early stopping, which is the baseline retrained only up to its
best validation epoch.

| Model | Train acc | Val acc | Val loss | Gap |
|---|---|---|---|---|
| baseline | 0.958 | **0.740** | 1.559 | +0.218 |
| weight decay (1e-4) | 0.942 | 0.727 | 1.047 | +0.215 |
| dropout (0.2) | 0.858 | 0.643 | 1.959 | +0.215 |
| early stopping (151 epochs) | 0.742 | 0.560 | **0.986** | +0.182 |

* **Baseline:** validation loss is lowest at epoch 150 and then climbs to about 1.56 while training
  loss heads to about 0.1. This is classic overfitting, with a jagged decision boundary.
* **Weight decay** gives the best balance. It cuts validation loss by a third (1.56 → 1.05) with
  almost no loss of accuracy. PyTorch applies it in the optimizer, so the biases are decayed too.
* **Dropout (0.2)** alone did not help here. Validation loss bottoms out at epoch 230 and ends
  higher than the baseline's. On 120 points over 1,500 epochs it slows overfitting but doesn't
  stop it.
* **Early stopping** has the lowest validation loss at a tenth of the cost, but also the lowest
  accuracy. That's a reminder that lower cross-entropy and higher accuracy aren't the same thing.

The conclusion matches Labs 6 and 7: regularizers give up training fit to generalise better. The
difference this time is that the loop doing it was written by hand.
