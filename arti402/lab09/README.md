# Lab 9 — Autoencoders: Learning Without Labels

**Week 9 · Autoencoders and Representation Learning**
Autoencoders, the bottleneck, denoising, and multitask learning.

| | |
|---|---|
| **Prerequisites** | Lab 8. The PyTorch training loop is reused almost unchanged |
| **Reference** | Ekman, *Learning Deep Learning*, Chapter 17 · official [PyTorch](https://pytorch.org/docs/stable/index.html) documentation |

## Files

| File | Description |
|---|---|
| [arti402_Lab9_2240006539.ipynb](./arti402_Lab9_2240006539.ipynb) | The completed lab notebook, with all outputs |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook (this version adds the Lab 9 figures: `draw_autoencoder_architecture`, `draw_digit_grid`, `draw_latent_space`, `draw_bottleneck_sweep`) |

Keep both files in the same folder. `arti402_figures.py` is imported directly by the notebook.
The notebook needs PyTorch and scikit-learn. No download is needed, because the digits come
bundled with scikit-learn. It was run with PyTorch 2.14.0 (CPU) and scikit-learn 1.9.1 on
Python 3.13. The handout's two `pip install` cells were removed, since both packages were already
installed and the saved output came from a different machine.

## What this lab covers

Every network so far was trained against labels. An autoencoder drops the labels and uses the
**input as its own target**: squeeze a 64-pixel image through a much smaller **bottleneck**, then
rebuild it, with loss `MSE(output, input)`. The data is scikit-learn's **8 × 8 hand-written
digits**, split into 1,347 training and 450 test images. The model is `64 → 32 → code → 32 → 64`
with ReLU, and a Sigmoid on the output, trained with Adam (lr 1e-3) for 60 epochs.

| Lab 8 (classification) | Lab 9 (autoencoder) |
|---|---|
| `loss = F.cross_entropy(model(xb), yb)` | `loss = F.mse_loss(model(xb), xb)` |
| target is the **label** | target is the **input** |

* **Exercise 1: build the hourglass.** `encoder` and `decoder` as two `nn.Sequential` blocks.
  Five images go in as `(5, 64)`, come out as codes of `(5, 8)`, and are rebuilt as `(5, 64)`.
* **Exercise 2: train it.** The Lab 8 loop with MSE against the input. The loss falls from
  0.178 to 0.022, and test reconstruction error is **0.0219**. The digits are blurrier but all
  readable, rebuilt from 8 numbers each.
* **Exercise 3: sweep the bottleneck.** Code sizes 2 to 32:

  | code size | compression | train MSE | test MSE |
  |---|---|---|---|
  | 2 | 32× | 0.0467 | 0.0479 |
  | 4 | 16× | 0.0346 | 0.0368 |
  | 8 | 8× | 0.0213 | 0.0219 |
  | 16 | 4× | 0.0193 | 0.0203 |
  | 32 | 2× | 0.0179 | 0.0190 |

  Predicting the average image scores 0.0740. The curve flattens after 8, and train and test
  errors stay almost equal, so there's no overfitting at this model and data size.
* **Exercise 4: look at a 2-D code.** The 0s and 4s get their own regions, without any labels.
  The 1s, 7s and 8s pile on top of each other, and so do the 5s and 9s.
* **Exercise 5: measure the representation.** Logistic regression trained on the codes:

  | features | test accuracy |
  |---|---|
  | 2-number code | 0.622 |
  | 8-number code | **0.924** |
  | all 64 pixels | 0.969 |

  Eight unsupervised numbers recover most of what the 64 pixels give.
* **Exercise 6: denoising.** Corrupt the input (`+ 0.5 · noise`, clamped), keep the target clean:

  | | clean input | noisy input |
  |---|---|---|
  | plain autoencoder | **0.0219** | 0.0534 |
  | denoising autoencoder | 0.0315 | **0.0436** |
  | do nothing | – | 0.1157 |

  The denoiser is better on noisy input and worse on clean input, so the robustness has a cost,
  just as dropout did in Lab 6.
* **Checkpoints 1–2.** Written answers in the notebook: why the target must be the input, why a
  64-wide code can learn the identity, why "unsupervised" still fits, how codes become useful for
  classification, the 50,000 unlabelled / 200 labelled recipe (keep the encoder), and which digits
  overlap in 2-D and why.

## Assessment and results

The handout says the graded Assessment is at the end, but no assessment cell was included. The
closing summary lists **anomaly detection** ("train on normal, alarm on whatever rebuilds badly")
without trying it, so I added that as the assessment. The same `Autoencoder(latent=8)` is trained
with one digit left out entirely. Each test image is then scored by its reconstruction error, and
the alarm threshold is the 95th percentile of the training errors.

**Held-out 4s:** mean error 0.0479 for the unseen 4s against 0.0211 for normal digits. **84.4%** of
the 4s (38 of 45) raise the alarm, with 5.4% false alarms and ROC AUC **0.964**. The decoder can
only draw shapes it knows, so it rebuilds each 4 as something like a 9 or an 8.

**Every digit held out in turn:**

| held-out digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| ROC AUC | 0.954 | **0.803** | 0.936 | 0.882 | **0.964** | 0.955 | 0.917 | 0.957 | 0.904 | 0.888 |
| detected | 73% | 50% | 77% | 35% | 84% | 74% | 56% | 73% | 65% | 33% |

* The mean AUC is **0.916**, so the error separates anomalies well for every digit.
* **4, 7, 5 and 0** are easiest to flag, because no other digit has a similar shape.
* **1** is the hardest. Its single stroke is part of almost every other digit, so it rebuilds well.
* **3 and 9** rank reasonably well (AUC about 0.89) but are caught only about a third of the time.
  They look like 5/8 and 4/8, so many land just under the threshold. Lowering the threshold would
  catch more of them but raise more false alarms.

The conclusion: an autoencoder flags "new" only when new means *far from everything it was trained
on*. The loop is still the same `zero_grad → forward → backward → step` from Lab 8, with a
different loss and a different target.
