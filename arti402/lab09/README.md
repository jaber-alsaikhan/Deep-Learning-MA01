# Lab 9 — Named Entity Recognition (NER) with spaCy

**Week 9 · Natural Language Processing**
Finding and classifying named entities (people, organisations, places, dates, money and more) in
text, and fine-tuning a spaCy NER model on custom annotations.

| | |
|---|---|
| **Prerequisites** | Python basics; `pip install spacy` and `python -m spacy download en_core_web_sm` |
| **Reference** | spaCy documentation: [Named Entities](https://spacy.io/usage/linguistic-features#named-entities) and [Training](https://spacy.io/usage/training#ner) |

## Files

| File | Description |
|---|---|
| [arti402_Lab9_2240006539.ipynb](./arti402_Lab9_2240006539.ipynb) | The completed lab notebook, with all outputs |
| [custom_ner_model/](./custom_ner_model) | The fine-tuned spaCy pipeline saved by Task 3 (`nlp.to_disk` / `spacy.load`) |
| [arti402_figures.py](./arti402_figures.py) | The course's diagram helper file (same version as Lab 7). This lab does not use it. It is kept so every lab folder has the same files |

The notebook was run with spaCy 3.8.16 and `en_core_web_sm` 3.8.0 on Python 3.13.

## What this lab covers

NER is a two-step process: **detect** the entity spans in the text, then **classify** each one
into a category (`PERSON`, `ORG`, `GPE`, `DATE`, `MONEY`, `PRODUCT`, …).

* **Pre-trained NER**: `nlp = spacy.load("en_core_web_sm")`, then `doc.ents` gives each entity's
  text, label, token start/end and `spacy.explain(label)`.
* **Annotating custom entities**: character-offset annotations, checked against spaCy's tokens
  with `offsets_to_biluo_tags`. BILUO stands for **B**egin, **I**nside, **L**ast, **U**nit and
  **O**utside. A `-` tag would mean a span doesn't line up with token boundaries.
* **Visualisation**: `displacy.render(doc, style="ent")`.

The handout's visualisation cell loads `custom_ner_model`, which is only created in Task 3. It now
falls back to the pre-trained model if that folder doesn't exist, so the notebook runs top to bottom.

## Tasks and results

**Task 1: entities in a user's sentence.** The cell asks for a sentence with `input()` and prints
each entity's details. It uses a default sentence when no input is available, for example when the
notebook runs headless. On the default sentence it found *Elon Musk* (PERSON), *Monday* (DATE),
*Tesla* (ORG), *Berlin* (GPE) and *$5 billion* (MONEY).

**Task 2: the agreement paragraph.**

1. The pre-trained model tags **"Bcd LLC" as `WORK_OF_ART`**, so the company is missed. I added it
   as an `ORG` span with `doc.char_span(...)`. `filter_spans` then keeps the new span and drops the
   wrong one.
2. Counts after the fix: **PERSON = 1** (*Samuel P. Jackson*) and **ORG = 2** (*Abc & Co.*, *Bcd LLC*).
3. The corrected entities are drawn with displaCy.

**Task 3: training a custom NER model.** The training sentence is the Microsoft one. It is annotated
with ORG (*Microsoft*), PERSON (*Bill Gates*, *Paul Allen*), GPE (*Albuquerque*, *New Mexico*) and
PRODUCT (*Windows*, *Office*), and wrapped with `Example.from_dict()`. The pre-trained NER component
is then fine-tuned for **20 iterations** with `resume_training()`, updating only `ner`. The result
is saved to `custom_ner_model/` and reloaded.

| | Windows | Office | Others |
|---|---|---|---|
| Before training | `ORG` ✗ | missed ✗ | all correct |
| After 20 iterations | `PRODUCT` ✓ | `PRODUCT` ✓ | all correct |

The loss fell from 2.56 to about 0 by iteration 5. The reloaded model also works on a sentence it
never saw: *"Satya Nadella leads Microsoft from Redmond, Washington, and Office is still popular."*
gives Satya Nadella (PERSON), Microsoft (ORG), Redmond and Washington (GPE), and Office (PRODUCT).

I first tried a blank `spacy.blank("en")` model. After 20 iterations on this one example it only
recognised "Paul Allen". Fine-tuning the pre-trained model keeps what it already knows and only
corrects the labels it got wrong.
