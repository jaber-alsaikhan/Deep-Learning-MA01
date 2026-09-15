# Lab 1 — From a Single Neuron to a Layer

**Week 1 · Introduction to Deep Learning**
History, motivations and applications of deep learning; neural-network architecture and terminology.

| | |
|---|---|
| **Marks** | 1 mark (graded) |
| **Estimated time** | 90–120 minutes |
| **Prerequisites** | Basic Python (lists, loops, functions) |

## Files

| File | Description |
|---|---|
| [arti402_lab1_2240006539.ipynb](./arti402_lab1_2240006539.ipynb) | The completed lab notebook |

## What this lab covers

Sections are labelled **Idea** (read and run), **Exercise** (you write code), and **Checkpoint**
(a short answer). Run cells in order, top to bottom — later cells depend on earlier ones.

* **Exercises 1–3** — vocabulary and a first neuron: putting a table of inputs into arrays,
  computing a neuron with four inputs by hand, and experimenting with your own weights and bias.
* **Exercises 4–6** — the step function, a whole layer built with plain Python loops, and
  predicting array shapes before running the code.
* **Exercises 7–9** — the same layer the NumPy way (`np.dot`), a full batched layer, and counting
  the trainable parameters of a fully connected network.
* **Checkpoints 1–3** and the graded **Q1–Q5 Assessment** — short-answer questions that check the
  vocabulary (feature, sample, batch, label, weight, bias, layer), shape reasoning, and parameter
  counting.

## By the end of this lab you should be able to

1. Use the vocabulary of deep learning correctly: *feature, sample, batch, label, weight, bias, layer*.
2. Compute the output of a single neuron in plain Python.
3. Explain what a **weight** does and what a **bias** does, and show it on a graph.
4. Compute the output of a whole layer — first with loops, then with NumPy.
5. Predict the **shape** of every array in a forward pass before you run it.
6. Count the trainable parameters of a fully connected network.

## Reference

Kinsley, H. & Kukieła, D. — *Neural Networks from Scratch in Python*
