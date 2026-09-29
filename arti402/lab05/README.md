# Lab 5 — Optimizers: SGD, Mini-batches, Momentum, RMSProp and Adam

**Week 5 · Optimization Algorithms**
Gradient descent variants: stochastic gradient descent (SGD), mini-batch SGD; adaptive learning
rate methods: Adam, RMSProp.

| | |
|---|---|
| **Marks** | 1 mark (graded) |
| **Estimated time** | 100–120 minutes |
| **Prerequisites** | Lab 2 — gradients, `numerical_gradient`, the chain rule |
| **Reading** | Kinsley & Kukieła, *Neural Networks from Scratch in Python*, Ch. 9–10 |

## Files

| File | Description |
|---|---|
| [arti402_Lab5_2240006539.ipynb](./arti402_Lab5_2240006539.ipynb) | The completed lab notebook, with all outputs |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook — the updated version that adds the Lab 5 figures (spiral data, optimizer family, valley paths, loss curves, decision boundaries) |

Keep both files in the same folder — `arti402_figures.py` is imported directly by the notebook.

## What this lab covers

The lab replaces Lab 2's slow `numerical_gradient` with real backpropagation, then asks:
*given the gradient, how exactly should we step?* It runs in four stages: **see it → open it →
build it → race it.**

* **Exercise 1** — `SpiralNet.forward` / `backward`: dense(2→64) → ReLU → dense(64→3) →
  softmax + cross-entropy, with backprop checked against the numerical gradient
  (max difference ≈ 1e-10, about 290× faster).
* **Exercise 2** — `Optimizer_SGD`: `weights -= lr * dweights`, with learning-rate decay in the
  base class.
* **Exercise 3** — `get_batches`: shuffle every epoch, then cut into mini-batches.
* **Exercise 4** — `Optimizer_SGD_Momentum`: `update = momentum * previous_update - lr * gradient`.
* **Exercise 5** — `Optimizer_RMSprop`: a per-parameter step size from a running average of
  squared gradients.
* **Exercise 6** — `Optimizer_Adam`: momentum + RMSProp + the start-up (bias) correction.
* **Checkpoints 1–2** — epochs vs iterations, batch-size trade-off, how momentum and RMSProp
  fix the zig-zag, and why Adam uses `iterations + 1`.
* **Q1–Q2 Assessment** — an optimizer race on the spiral dataset, and a learning-rate sweep for Adam.

## Results

**Batch size** (plain SGD, lr 1.0, 1,000 epochs):

| Batch size | Steps | Train accuracy | Loss |
|---|---|---|---|
| 300 (full) | 1,001 | 0.427 | 1.066 |
| 100 | 3,003 | 0.583 | 0.810 |
| 32 | 10,010 | 0.797 | 0.565 |
| 8 | 38,038 | 0.570 | 0.804 |

**Q1 — The race** (10,000 epochs, full batch):

| Optimizer | Train accuracy | Train loss | Test accuracy |
|---|---|---|---|
| SGD | 0.703 | 0.639 | 0.637 |
| SGD + decay | 0.717 | 0.729 | 0.567 |
| SGD + momentum | 0.970 | 0.082 | 0.777 |
| RMSProp | 0.890 | 0.269 | 0.807 |
| Adam | 0.967 | 0.091 | 0.820 |

**Q2 — Is Adam tuning-free?** (2,000 epochs, no decay):

| Learning rate | Train accuracy | Worst loss seen |
|---|---|---|
| 0.0005 | 0.670 | 1.10 |
| 0.005 | 0.927 | 1.10 |
| 0.05 | 0.950 | 1.77 |
| 0.5 | 0.503 | 2.73 |
| 2.0 | 0.340 | 10.45 |

Momentum and Adam fit the training spiral almost perfectly, while plain SGD is still struggling
after 10,000 epochs. Adam is not tuning-free: too small a learning rate crawls, too large a one
blows up to chance. And the best network scores about 0.97 on the training data but only about
0.82 on new data — it is overfitting, which is where Lab 6 picks up.

## By the end of this lab you should be able to

1. Run a full forward and backward pass, and verify backprop against numerical gradients.
2. Implement SGD, and explain the difference between an **epoch** and an **iteration**.
3. Train with **mini-batches**, and explain the trade-off in batch size.
4. Implement **momentum**, **RMSProp** and **Adam**, and say what problem each one fixes.
5. Compare optimizers fairly, and recognise when a learning rate is too small or too large.

## References

* [TensorFlow Playground](https://playground.tensorflow.org)
* Andrew Ng — *Mini Batch Gradient Descent*, *Gradient Descent With Momentum*, *RMSProp*, *Adam Optimization Algorithm*
* Gabriel Goh — [*Why Momentum Really Works*](https://distill.pub/2017/momentum/) (Distill)
