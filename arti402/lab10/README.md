# Lab 10 — Attention: Letting the Network Look Back

**Week 10 · Attention Mechanisms**
Scaled dot-product attention, self-attention, masking, and the road to transformers.

| | |
|---|---|
| **Prerequisites** | Lab 4 (the problem) and Lab 9 (the bottleneck). Lab 2's softmax returns |
| **Reference** | Ekman, *Learning Deep Learning*, Chapter 15 · Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), Section 3.2 |

## Files

| File | Description |
|---|---|
| [arti402_Lab10_2240006539.ipynb](./arti402_Lab10_2240006539.ipynb) | The completed lab notebook, with all outputs |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook (this version adds the Lab 10 figures: `draw_bottleneck_vs_attention`, `draw_qkv_mechanism`, `draw_attention_heatmap`, `draw_length_sweep`) |

Keep both files in the same folder. `arti402_figures.py` is imported directly by the notebook.
The whole lab is plain NumPy and Matplotlib. Nothing is trained and no data is downloaded. It was
run with NumPy 2.5.2 and Matplotlib 3.11.1 on Python 3.13.

## What this lab covers

Lab 4's LSTM had to squeeze a whole sequence into **one hidden state**, and it lost accuracy as the
sequence grew. Attention keeps every state and, at each step, computes a weighted average of the
ones it wants to look at:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

* **Exercise 1: write it.** `attention(Q, K, V, mask=None)` in four lines: `Q @ K.T / sqrt(d_k)`,
  an optional mask (−10⁹ where blocked), `softmax`, then `weights @ V`. With a query matching key 1,
  the weights are `[0.67, 0.33]` and the output is 13.3, leaning towards the value 10.
* **Exercise 2: a lookup table you can check by hand.** Four one-hot keys:

  | query | weights | output |
  |---|---|---|
  | exact (slot 2) | 0.215 · **0.355** · 0.215 · 0.215 | 24.3 |
  | 5× sharper | 0.066 · **0.802** · 0.066 · 0.066 | 21.3 |
  | torn between 1 and 2 | **0.409** · **0.409** · 0.091 · 0.091 | 18.6 |

  Attention is soft: the right slot wins but never takes everything. A larger query makes the
  lookup more decisive, and a torn query returns a blend.
* **Exercise 3: why divide by √d_k.** Largest softmax weight over 6 random keys:

  | d_k | raw score range | max weight, unscaled | max weight, scaled |
  |---|---|---|---|
  | 4 | −5.06 to 3.49 | 0.332 | 0.272 |
  | 64 | −11.37 to 4.99 | 0.969 | 0.320 |
  | 512 | −24.36 to 11.33 | **1.000** | 0.300 |

  Unscaled, the softmax saturates onto one key and the other gradients vanish. The divide keeps it
  soft.
* **Exercise 4: attention against the bottleneck.** Store `T` random key–value pairs and ask for one
  by its key. The bottleneck is simulated by averaging every value into one vector (400 trials each):

  | T | 2 | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|
  | attention | 1.000 | 0.993 | 0.990 | 0.990 | 0.980 | **0.958** |
  | one summary vector | 0.430 | 0.225 | 0.135 | 0.050 | 0.035 | 0.015 |
  | chance | 0.500 | 0.250 | 0.125 | 0.062 | 0.031 | 0.016 |

  Attention barely cares about length. The single vector sits on the chance line from the start.
* **Exercise 5: self-attention.** `attention(X, X, X)` on "the cat sat on the mat" with random
  embeddings. Each word attends mostly to itself, and the two "the"s get **identical** rows
  (`[0.353, 0.167, 0.021, 0.099, 0.353, 0.006]`). Attention has no idea of order, which is why
  transformers add **positional encoding**.
* **Idea 6: multi-head attention.** Several copies of the same layer, each with its own learned
  projections, run in parallel and concatenated.

## Assessment and results

**Q1 — causal masking.** `causal = np.tril(np.ones((T, T), dtype=bool))` lets position *i* see only
positions 0..*i*. Run through `attention(Xc, Xc, Xc, mask=causal)`, every weight above the diagonal
is exactly 0 and every row still sums to 1. The first row is `[1, 0, 0, 0, 0]`: position 0 can see
nothing except itself, so after the softmax all its weight has to land there.

**Q2 — what the mask costs.**

| position | keys visible | weight on itself, unmasked | weight on itself, masked |
|---|---|---|---|
| 0 | 1 | 0.585 | 1.000 |
| 1 | 2 | 0.556 | 0.577 |
| 2 | 3 | 0.412 | 0.480 |
| 3 | 4 | 0.695 | 0.758 |
| 4 | 5 | 0.358 | 0.358 |

Early positions lose most of their context, and with fewer keys competing the weight on itself
rises. The cost shrinks along the sequence. The last position sees every key, so masked and
unmasked match exactly. That loss of context is the price a GPT-style decoder pays to generate text
one word at a time without looking at the future.
