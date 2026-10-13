# Lab 7 — TensorFlow and Keras: Everything You Built, in One Line Each

**Week 7 · Deep Learning Frameworks**
Overview of TensorFlow and PyTorch; building and training neural networks using TensorFlow.

| | |
|---|---|
| **Prerequisites** | Labs 1–6 — every piece used here was built by hand in NumPy first |
| **Reference** | Official [TensorFlow](https://www.tensorflow.org) and [Keras](https://keras.io) documentation |

## Files

| File | Description |
|---|---|
| [arti402_Lab7_2240006539.ipynb](./arti402_Lab7_2240006539.ipynb) | The completed lab notebook, with all outputs |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook (same version as Lab 6 — `draw_overfitting_curves`, `draw_decision_boundaries`) |
| [spiral_model.keras](./spiral_model.keras) | The early-stopped model saved by Idea 6 (`model.save` / `keras.models.load_model`) |

Keep the notebook and `arti402_figures.py` in the same folder — the figures file is imported
directly. The notebook needs TensorFlow (`pip install tensorflow`); it was run with
TensorFlow 2.21.0 / Keras 3.15.1 on CPU.

## What this lab covers

Sections are labelled **Idea** (read and run), **Exercise** (you write code), and the graded
**Assessment** at the end. The data is Lab 6's harsh setup: **40 spiral samples per class** for
training, with separate validation and test sets, and a `Dense(128, relu) → Dense(3, softmax)`
network trained with Adam.

The whole point of the lab is the translation from Labs 1–6 to Keras:

| Built by hand | TensorFlow / Keras |
|---|---|
| `Layer_Dense` + `Activation_ReLU` | `layers.Dense(128, activation="relu")` |
| softmax + cross-entropy | `loss="sparse_categorical_crossentropy"` |
| every `backward()` | automatic differentiation |
| `Optimizer_Adam(0.02)` | `keras.optimizers.Adam(0.02)` |
| the training loop | `model.fit(...)` |
| L2 penalty, `Layer_Dropout` | `kernel_regularizer=regularizers.l2(5e-4)`, `layers.Dropout(0.2)` |
| "you could stop at the best epoch" | `EarlyStopping(restore_best_weights=True)` |

* **Exercise 1** — `build_model`: the Lab 6 network as a `keras.Sequential`, with optional L2 and
  dropout. Keras's parameter count matches `n_in * n_out + n_out`: 384 + 387 = **771**.
* **Exercise 2** — `compile`, `fit` (100 epochs, batch 32, with validation data) and `evaluate`.
* **Exercise 3** — a model with `kernel_regularizer=l2(5e-4)`; the penalty lives in `model.losses`.
* **Exercise 4** — an `EarlyStopping` callback on `val_loss` with patience 25.
* **Idea 5** — `tf.data.Dataset` with `.repeat()` and `steps_per_epoch`, so 2,000 gradient
  steps cost 100 epochs of Keras bookkeeping instead of 2,000.
* **Idea 6** — saving and reloading a model as `.keras`.
* **Q1–Q2 Assessment** — re-run Lab 6's regularization showdown in Keras and plot it.

## Results

**Exercise 2 — baseline, 100 epochs (batch 32):** training accuracy 0.817 vs validation 0.607
(gap +0.210). Validation loss levels off and starts to rise, the same picture as Lab 6.

**Exercise 3 — L2 (5e-4):** train 0.808 / val 0.600; the penalty adds 0.0324 to the loss and adds
no parameters (still 771).

**Exercise 4 — early stopping:** asked for 200 epochs, stopped after **158**; the best epoch was
133 with validation loss **0.939**, and `restore_best_weights` rewound to exactly that.

**Idea 6 — save / reload:** predictions identical after reloading (`spiral_model.keras`, 30 KB).

**Q1 — the showdown in Keras** (Adam, 2,000 full-batch updates = 100 epochs × 20 steps):

| Model | Train acc | Val acc | Val loss | Gap |
|---|---|---|---|---|
| baseline | 0.983 | 0.730 | 2.228 | +0.253 |
| L2 (5e-4) | 0.950 | 0.723 | 1.152 | +0.227 |
| dropout (0.2) | 0.867 | 0.667 | 1.780 | +0.200 |
| early stopping | 0.917 | 0.647 | **0.948** | +0.270 |

**Q2:** the baseline's validation loss bottoms out at epoch 7 (about 150 updates) and then climbs
to 2.2 while training loss goes to almost zero. Its decision boundary is jagged and memorises
individual points; L2's is smoother, and dropout's is the smoothest.

The conclusions match Lab 6: every regularizer fits the training data worse on purpose and lowers
validation loss. L2 nearly halves the baseline's validation loss while keeping its validation
accuracy, and early stopping gives the lowest validation loss of all for the least training.

## By the end of this lab you should be able to

1. Build, compile, fit and evaluate a network with the four-step Keras workflow.
2. Map every Keras argument back to the NumPy code from Labs 1–6.
3. Choose between `sparse_categorical_crossentropy` and `categorical_crossentropy` from the label format.
4. Add L2, dropout and batch normalisation as layer arguments and layers.
5. Use callbacks — especially `EarlyStopping(restore_best_weights=True)`.
6. Feed data with `tf.data`, and save and reload a trained model.

## References

* [TensorFlow 2 quickstart for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner)
* [Keras — Getting started](https://keras.io/getting_started/)
* [`keras.layers`](https://keras.io/api/layers/) · [`keras.optimizers`](https://keras.io/api/optimizers/) · [`keras.callbacks`](https://keras.io/api/callbacks/)
* [`tf.data` guide](https://www.tensorflow.org/guide/data)
* [Keras Sequential model guide](https://keras.io/guides/sequential_model/)
