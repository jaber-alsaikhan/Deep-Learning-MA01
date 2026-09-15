# Lab 2 — Activations, Loss, and How a Network Learns

**Week 2 · Introduction to Deep Learning**
Activation functions and feedforward neural networks; gradient-based optimisation and back-propagation.

| | |
|---|---|
| **Marks** | 1 mark (graded) |
| **Estimated time** | 100–120 minutes |
| **Prerequisites** | Lab 1 — comfortable with `np.dot`, shapes, and the forward pass |

## Files

| File | Description |
|---|---|
| [arti402_lab2_2240006539.ipynb](./arti402_lab2_2240006539.ipynb) | The completed lab notebook |

## What this lab covers

Sections are labelled **Idea** (read and run), **Exercise** (you write code), and **Checkpoint**
(a short answer). Run cells in order, top to bottom — later cells depend on earlier ones.

* **Exercise 1** — collapse a stack of linear layers into one, numerically, to show why
  non-linearity is needed.
* **Exercise 2** — implement **ReLU** and **sigmoid** from scratch.
* **Exercise 3** — build a reusable `Layer_Dense` class and stack it into a feedforward network.
* **Exercise 4** — implement **softmax** for the output layer.
* **Exercise 5** — implement **categorical cross-entropy** loss.
* **Exercise 6** — estimate a derivative numerically (measure a slope).
* **Exercise 7** — descend the loss curve with **gradient descent**.
* **Exercise 8** — hand-compute a **backward pass** through a single neuron using the chain rule.
* **Q1–Q2 Assessment** — build the full network, then measure loss and accuracy before training.

## By the end of this lab you should be able to

1. Demonstrate, numerically, that stacked linear layers collapse into a single layer.
2. Implement **ReLU**, **sigmoid** and **softmax**, and choose the right one for a given layer.
3. Build a reusable `Layer_Dense` class and stack it into a feedforward network.
4. Measure how wrong a network is using **categorical cross-entropy** loss.
5. Estimate a derivative numerically and use it to run **gradient descent**.
6. Hand-compute a **backward pass** through a single neuron using the chain rule.

## Reference

Kinsley, H. & Kukieła, D. — *Neural Networks from Scratch in Python*
