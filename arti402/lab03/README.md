# Lab 3 — CNN Architecture: The Four Building Blocks

**Week 3 · Neural Network Architectures**
Convolutional neural networks (CNNs) for image analysis; transfer learning and fine-tuning
pre-trained models.

| | |
|---|---|
| **Marks** | 1 mark (graded) |
| **Estimated time** | 100–120 minutes |
| **Prerequisites** | Labs 1–2 — shapes, `Layer_Dense`, ReLU, softmax, cross-entropy |
| **Reference** | Ekman, *Learning Deep Learning*, Ch. 7–8 (terminology) |

## Files

| File | Description |
|---|---|
| [arti402_lab3_2240006539.ipynb](./arti402_lab3_2240006539.ipynb) | The completed lab notebook |
| [arti402_figures.py](./arti402_figures.py) | Diagram helper functions used by the notebook — draws the CNN architecture, convolution step, max-pooling and shape-journey figures from the same numbers the notebook computes |
| [Zebra_image.jpeg](./Zebra_image.jpeg) | Input photo used in the architecture diagram |

Keep all three files in the same folder — `arti402_figures.py` is imported directly by the
notebook, and it looks for `Zebra_image.jpeg` next to itself (falling back to a grey placeholder
if the photo is missing).

## What this lab covers

Sections are labelled **Idea** (read and run), **Exercise** (you write code), and **Checkpoint**
(a short answer). Run cells in order, top to bottom — later cells depend on earlier ones.

The lab builds all four CNN building blocks from scratch in NumPy:

1. **Filtering** (convolution) — learnable
2. **Max pooling** — fixed, no parameters
3. **Flatten** — fixed, no parameters
4. **Fully connected** (`Layer_Dense`, from Lab 2) — learnable

* **Exercise 1** — implement `convolve2d`: slide a kernel over an image and sum.
* **Exercise 2** — the filtering layer: apply a bank of filters, stack into a 3-D tensor, add ReLU.
* **Exercise 3** — max pooling: shrink feature maps while keeping the strongest signal.
* **Exercise 4** — flatten a 3-D tensor into a 1-D vector, and trace the shape journey end to end.
* **Exercise 5** — count the learnable parameters of a two-conv-layer network.
* **Checkpoint 1** — short questions on output depth, which blocks learn, and filter shape.
* **Q1–Q4 Assessment** — use the four blocks as a frozen feature extractor (transfer-learning
  pattern), compare raw pixels vs. the CNN pipeline on a shifted test set, measure how pooling
  tile size trades off spatial precision for shift tolerance, and answer short questions on the
  results.

## By the end of this lab you should be able to

1. Read any CNN diagram as a sequence of the four blocks.
2. Implement convolution, max pooling and flatten from scratch.
3. Predict the shape of a tensor at every stage of a CNN.
4. Say which blocks have learnable parameters, and count them.
5. Show experimentally why a CNN handles a shifted image when a dense network cannot.

## Reference

Ekman, M. — *Learning Deep Learning*, Ch. 7–8
