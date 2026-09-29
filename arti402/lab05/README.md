# Lab 5 — Optimizers: SGD, Mini-batches, Momentum, RMSProp and Adam

**Week 5 · Optimization Algorithms**
Gradient descent variants: stochastic gradient descent (SGD), mini-batch SGD; adaptive learning
rate methods: Adam, RMSProp.

| | |
|---|---|
| **Marks** | 1 mark (graded) |
| **Estimated time** | 100–120 minutes |
| **Prerequisites** | Lab 2 — gradients, `numerical_gradient`, the chain rule |
| **Reference** | Kinsley & Kukieła, *Neural Networks from Scratch in Python*, Ch. 9–10 |

## Files

| File | Description |
|---|---|
| [arti402_Lab5_2240006539.ipynb](./arti402_Lab5_2240006539.ipynb) | The completed lab notebook |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook — the updated version that adds the Lab 5 figures (spiral data, optimizer family, valley paths, loss curves, decision boundaries) |

Keep both files in the same folder — `arti402_figures.py` is imported directly by the notebook.

## What this lab covers

Sections are labelled **Idea** (read and run), **Exercise** (you write code), and **Checkpoint**
(a short answer). Run cells in order, top to bottom — later cells depend on earlier ones.

The lab runs in four stages: **see it → open it → build it → race it.** The network is NNFS's
`dense(2 → 64) → ReLU → dense(64 → 3) → softmax` trained on the 3-class spiral dataset.

* **Exercise 1** — `SpiralNet`: wire up the forward and backward pass, and check backprop against
  Lab 2's `numerical_gradient` (differences ~1e-10, about 225× faster).
* **Exercise 2** — `Optimizer_SGD`: plain gradient descent.
* **Exercise 3** — `get_batches`: shuffle and split the data into mini-batches.
* **Exercise 4** — `Optimizer_SGD_Momentum`: add a memory of past updates to damp zig-zagging.
* **Exercise 5** — `Optimizer_RMSprop`: a per-parameter step size from the running average of
  squared gradients.
* **Exercise 6** — `Optimizer_Adam`: momentum + RMSProp, with bias correction.
* **Checkpoints 1–2** — short questions on epochs vs. iterations, batch size, and how momentum,
  RMSProp and Adam's start-up correction work.
* **Q1–Q2 Assessment** — race all optimizers on the spiral, then test Adam across learning rates.

## Results

**Batch size** (plain SGD, lr = 1.0, 1,000 epochs):

| Batch size | Steps | Accuracy |
|---|---|---|
| 300 (full) | 1,001 | 0.427 |
| 100 | 3,003 | 0.583 |
| 32 | 10,010 | 0.797 |
| 8 | 38,038 | 0.450 |

**The optimizer race** (10,000 epochs, full batch):

| Optimizer | Train acc | Train loss | Test acc |
|---|---|---|---|
| SGD | 0.743 | 0.624 | 0.657 |
| SGD + decay | 0.717 | 0.729 | 0.567 |
| SGD + momentum | 0.967 | 0.104 | 0.783 |
| RMSProp | 0.897 | 0.273 | 0.783 |
| Adam | 0.970 | 0.077 | 0.820 |

**Adam learning rate** (2,000 epochs): 0.0005 → 0.663, 0.005 → 0.940, 0.05 → 0.950,
0.5 → 0.470, 2.0 → 0.340. Adam is not tuning-free — too large a learning rate still diverges.

Adam and momentum win the race, but the gap between train (~0.97) and test (~0.82) accuracy, and
the small islands in their decision boundaries, show the network is starting to overfit — the
topic of the next lab.

## By the end of this lab you should be able to

1. Run a full forward and backward pass, and verify backprop against numerical gradients.
2. Implement SGD, and explain the difference between an **epoch** and an **iteration**.
3. Train with **mini-batches**, and explain the trade-off in batch size.
4. Implement **momentum**, **RMSProp** and **Adam**, and say what problem each one fixes.
5. Compare optimizers fairly, and recognise when a learning rate is too small or too large.

## References

* Kinsley & Kukieła — *Neural Networks from Scratch in Python*, Chapters 9–10
* [TensorFlow Playground](https://playground.tensorflow.org)
* Andrew Ng — [*Mini Batch Gradient Descent*](https://www.youtube.com/watch?v=4qJaSmvhxi8), [*Gradient Descent With Momentum*](https://www.youtube.com/watch?v=k8fTYJPd3_I), *RMSProp*, [*Adam Optimization Algorithm*](https://www.youtube.com/watch?v=JXQT_vxqwIs)
* Distill — [*Why Momentum Really Works*](https://distill.pub/2017/momentum/)
