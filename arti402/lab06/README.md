# Lab 6 — Regularization: Closing the Gap

**Week 6 · Optimization Algorithms (cont.)**
Regularization techniques: dropout, L1 and L2 regularization; batch normalization and weight
initialization strategies.

| | |
|---|---|
| **Prerequisites** | Lab 5 — backprop, `Layer_Dense`, Adam, the training loop |
| **Reference** | Kinsley & Kukieła, *Neural Networks from Scratch in Python*, Ch. 14–15 |

## Files

| File | Description |
|---|---|
| [arti402_Lab6_2240006539.ipynb](./arti402_Lab6_2240006539.ipynb) | The completed lab notebook |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook — the updated version that adds the Lab 6 figures (overfitting curves, weight histograms, initialisation depth, batch norm effect) |

Keep both files in the same folder — `arti402_figures.py` is imported directly by the notebook.

## What this lab covers

Sections are labelled **Idea** (read and run), **Exercise** (you write code), and **Checkpoint**
(a short answer). Run cells in order, top to bottom — later cells depend on earlier ones.

The lab runs in four stages: **see the problem → measure it properly → build the tools → prove it.**
The setup is deliberately harsh: a `dense(2 → 128) → ReLU → dense(128 → 3)` network trained with
Adam on only **40 spiral samples per class**, with separate training, validation and test sets.

* **Exercise 1** — `evaluate`: loss and accuracy with dropout switched off.
* **Exercise 2** — `regularization_loss` and `add_regularizer_gradients`: L1 and L2 penalties in
  both the loss and the gradient.
* **Exercise 3** — `Layer_Dropout`: inverted dropout, on in training and off at test time.
* **Exercise 4** — `activation_sizes`: push data through a 10-layer network with small, He and
  large initialisation.
* **Exercise 5** — `batchnorm_forward`: normalise each feature across the batch, then scale and
  shift with `gamma` and `beta`.
* **Checkpoints 1–2** — short questions on dropout's keep rate and scaling, He initialisation, and
  what `gamma` and `beta` are for.
* **Q1–Q2 Assessment** — train four networks that differ only in regularization, then compare
  their decision boundaries and training curves.

## Results

**Baseline overfitting** (no regularization, 4,000 epochs): training accuracy 0.983 vs validation
0.693. Validation loss bottoms out at **epoch 500** and climbs for the rest of training.

**What each penalty does to the first layer's weights:**

| Penalty | Weights ≈ 0 | Largest weight |
|---|---|---|
| none | 4% | 19.5 |
| L1 (1e-3) | 69% | 4.7 |
| L2 (1e-3) | 9% | 1.8 |

**Weight initialisation** (activation std after 10 layers): 0.01 → 1.9e-11 (vanishes),
He `sqrt(2/n)` → 1.78 (stable), 1.0 → 1.9e+9 (explodes).

**Batch norm forward:** mean +7.04 / std 2.97 → mean 0.00 / std 1.00; with `gamma=2, beta=5`
→ mean 5.00 / std 2.00.

**The showdown** (Adam, 4,000 epochs):

| Model | Train acc | Val acc | Val loss | Gap |
|---|---|---|---|---|
| baseline | 0.983 | 0.693 | 1.877 | +0.290 |
| L2 (5e-4) | 0.958 | 0.743 | 1.029 | +0.215 |
| dropout (0.2) | 0.883 | 0.650 | 1.446 | +0.233 |
| L2 + dropout | 0.875 | 0.647 | 1.126 | +0.228 |

Every regularized model fits the training data worse but shrinks the gap and lowers validation
loss. L2 is the clear winner here — it also gives the best validation accuracy and a much smoother
decision boundary. Dropout has little room to help on a network this small.

## By the end of this lab you should be able to

1. Split data properly into **training, validation and test**, and say what each is for.
2. Recognise overfitting from a loss curve, not just from a final number.
3. Implement **L1 and L2** penalties in both the loss and the gradient.
4. Implement **dropout**, and explain why it must be switched off at test time.
5. Explain why weight initialisation matters, and what **batch normalisation** does.
6. Accept the central trade: a regularised model fits the training data **worse** on purpose.

## References

* Kinsley & Kukieła — *Neural Networks from Scratch in Python*, Chapters 14–15
* [TensorFlow Playground](https://playground.tensorflow.org)
* Ioffe & Szegedy (2015) — [*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*](https://arxiv.org/abs/1502.03167)
* He et al. (2015) — [*Delving Deep into Rectifiers*](https://arxiv.org/abs/1502.01852)
* Srivastava et al. (2014) — [*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/v15/srivastava14a.html)
