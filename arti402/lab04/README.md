# Lab 4 — Recurrent Networks: RNN, LSTM and GRU

**Week 4 · Neural Network Architectures (cont.)**
Recurrent neural networks (RNNs) for sequential data; Long Short-Term Memory (LSTM) and Gated
Recurrent Units (GRUs).

| | |
|---|---|
| **Marks** | 1 mark (graded) |
| **Estimated time** | 100–120 minutes |
| **Prerequisites** | Labs 1–3 — `Layer_Dense`, softmax, `train_head`, the idea of weight sharing |
| **Reading** | Chris Olah — [*Understanding LSTM Networks*](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) |

## Files

| File | Description |
|---|---|
| [arti402_Lab4_2240006539.ipynb](./arti402_Lab4_2240006539.ipynb) | The completed lab notebook |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook — the updated version that adds the Lab 4 figures (unrolled RNN, repeated multiplication, LSTM cell, recurrent parameter counts, assessment pipeline) |

Keep both files in the same folder — `arti402_figures.py` is imported directly by the notebook.

## What this lab covers

Sections are labelled **Idea** (read and run), **Exercise** (you write code), and **Checkpoint**
(a short answer). Run cells in order, top to bottom — later cells depend on earlier ones.

The central idea: a CNN shares **one filter across space**; an RNN shares **one cell across time**.

* **Exercise 1** — `rnn_step`: one step of an RNN, `h = tanh(x·Wx + h_prev·Wh + b)`.
* **Exercise 2** — `rnn_forward`: unroll the cell over a sequence of any length.
* **Exercise 3** — count recurrent parameters: RNN = 1 block, GRU = 3, LSTM = 4
  (204 / 612 / 816 for 4 inputs and 12 hidden units).
* **Exercise 4** — `lstm_step`: forget gate, input gate + candidate, output gate, and the
  additive cell-state update `c = f·c_prev + i·g`.
* **Checkpoints 1–2** — short questions on shapes, parameter sharing, the forget gate and GRUs.
* **Q1–Q4 Assessment** — a memory task where only step 1 carries the class. Frozen RNN and
  LSTM cells feed a trained dense head on their final hidden state.

## Results

| T | RNN accuracy | LSTM accuracy |
|---|---|---|
| 2 | 1.000 | 1.000 |
| 5 | 0.608 | 0.992 |
| 10 | 0.292 | 1.000 |
| 20 | 0.317 | 0.983 |
| 40 | 0.275 | 0.942 |
| 80 | 0.333 | 0.792 |

The plain RNN falls to chance (≈ 0.33) by T = 10. The LSTM holds on to step 1 up to T = 40 and
only starts to slip at T = 80. Turning the forget-gate bias from −2 to +4 at T = 20 raises
accuracy from 0.28 to 0.98, because the memory kept after 20 steps goes from almost none to
`0.98^20 ≈ 0.67`.

## By the end of this lab you should be able to

1. Explain why a dense network cannot handle word order or variable-length input.
2. Implement an RNN cell and unroll it over a sequence.
3. Count recurrent parameters, and explain why they do not depend on sequence length.
4. Show why a plain RNN forgets, using the idea of repeated multiplication.
5. Implement an LSTM cell and explain its three stages and its two kinds of memory.
6. Compare RNN, LSTM and GRU, and show experimentally that the LSTM remembers longer.

## References

* StatQuest — *Recurrent Neural Networks, Clearly Explained* and *Long Short-Term Memory (LSTM), Clearly Explained*
* Andrej Karpathy — *The Unreasonable Effectiveness of Recurrent Neural Networks*
* Michael Phi — *Illustrated Guide to LSTM's and GRU's*
