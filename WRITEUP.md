# DS-GA 1011 — Assignment 4

- **GitHub repository:** https://github.com/Abhinav0710rajput/nyu-dsga1011-a4
- **Google Drive folder (model checkpoint for Q7):** https://drive.google.com/drive/folders/1khH96hK15rpBGpVrHhVMVAB5F9UblvN7?usp=sharing

---

## Q2a. Design a data transformation

**Transformation: WordNet synonym replacement + QWERTY keyboard typos.**

The review is processed word by word using the regex `[A-Za-z]+` over the raw text, so
punctuation, whitespace and HTML tags are left unchanged. A word is never modified if:

- it has fewer than 3 letters;
- it is a negation or function word (*not, no, nor, never, none, nothing, neither, nobody,
  nowhere, without, hardly, barely, scarcely, cannot, but, the, and, for, are, was, were, this,
  that, with, you, his, her, its, they, them, she, him, has, had, have, been, who, what, which,
  than, then, there, their, from*), or a preposition/particle (*about, above, across, after,
  against, along, among, around, before, behind, below, beneath, beside, between, beyond, down,
  during, into, near, off, onto, out, over, past, since, through, throughout, till, toward,
  towards, under, until, upon, within*);
- it is directly adjacent to an apostrophe (part of a contraction, e.g. *didn't*, *movie's*).

Every other word is processed independently, in this order:

1. **Synonym replacement (probability 0.5).** Applied only if the word has at least one
   adjective/adverb WordNet sense, has no verb sense, and is already in base form (WordNet
   `morphy(word, pos)` returns the word unchanged). Capitalized words in the middle of a sentence
   are skipped. The word is replaced with a uniformly random single-word lemma, different from
   the word, of its first (most frequent) adjective/adverb WordNet synset. If there is no such
   lemma, the word is kept.
2. **Keyboard typo (probability 0.3).** One letter, chosen uniformly among all positions except
   the first, is replaced with a uniformly random neighbouring key on a QWERTY keyboard
   (e.g. `o` → one of `i, p, k, l`; `a` → one of `q, w, s, z`).

In both steps the original capitalization (lowercase, Capitalized, ALL-CAPS) is preserved.
WordNet is accessed via NLTK 3.8.1.

**Why it is reasonable.** User reviews are typed quickly, often on phones, and hitting an
adjacent key is the most common typo. Only one letter per word is changed and the first letter is
never touched, so each word stays readable. Different reviewers also express the same opinion
with different words ("boring" vs. "tedious"). Restricting synonyms to base-form
adjectives/adverbs and their most frequent sense keeps sentences grammatical and the meaning
unchanged. Negations are never modified, so a human would assign the transformed review the same
sentiment label as the original.

---

## Q3. Data augmentation

| Model | Original test set | Transformed test set |
|---|---|---|
| Trained on original data (Q1/Q2) | 93.07% | 88.09% |
| Trained on augmented data (Q3) | 92.74% | 89.76% |

**(1) Did performance on the transformed test data improve?** Yes, from 88.09% to 89.76%
(+1.67 points).

**(2) How did augmentation affect performance on the original test data?** It decreased slightly,
from 93.07% to 92.74% (−0.33 points), which is essentially unchanged.

**Intuitive explanation.** The augmented model sees reviews with keyboard typos and rarer
synonyms paired with their unchanged labels during training. A misspelled word such as
*atrocuous* is split by WordPiece into unfamiliar sub-word pieces. The original model never learned
that these pieces carry the same sentiment as *atrocious*, so it makes more mistakes on them.
Training on transformed examples teaches the model to map these noisy forms to the same
sentiment, which improves accuracy on the transformed test set. Only 5,000 of the 30,000 training
examples are transformed, so the model still learns mostly from clean text. Its accuracy on the
original test set therefore barely changes. The small drop reflects part of the training budget
going to a slightly different input distribution.

**Limitation.** The augmentation uses exactly the same transformation that produces the
transformed test set, so the model is trained on the specific shift it is evaluated on. In
practice the out-of-distribution shift at test time is unknown: it could be other typo patterns,
slang, sarcasm, paraphrases or a different domain. Augmenting with one hand-designed
transformation does not make the model robust to those shifts.

---

## Q4. Data statistics and processing

Table 1: Data statistics before any pre-processing (T5 tokenizer).

| Statistics Name | Train | Dev |
|---|---|---|
| Number of examples | 4225 | 466 |
| Mean sentence length | 17.10 | 17.07 |
| Mean SQL query length | 216.37 | 210.05 |
| Vocabulary size (natural language) | 791 | 465 |
| Vocabulary size (SQL) | 555 | 395 |

Table 2: Data statistics after pre-processing (T5 tokenizer).

| Statistics Name | Train | Dev |
|---|---|---|
| Model name | `google-t5/t5-small` | `google-t5/t5-small` |
| Mean sentence length | 23.10 | 23.07 |
| Mean SQL query length | 217.49 | 211.17 |
| Vocabulary size (natural language) | 796 | 470 |
| Vocabulary size (SQL) | 557 | 397 |

---

## Q5. T5 fine-tuning details

Table 3: Details of the best-performing T5 model configuration (fine-tuned).

| Design choice | Description |
|---|---|
| **Data processing** | (1) Strip and collapse whitespace in both the natural-language (NL) input and the SQL target. (2) Prepend the task prefix `"translate English to SQL: "` to every NL input. (3) Replace every `<` in the SQL targets with the placeholder `less_than`, because the T5 vocabulary has no token for `<`. After generation, `less_than` is mapped back to `<` before the query is executed. |
| **Tokenization** | Default `google-t5/t5-small` tokenizer (`T5TokenizerFast`) for both encoder and decoder, truncation at 512 tokens. Encoder input: tokenized prefixed NL + `</s>`. Decoder target: tokenized SQL + `</s>`. Decoder input (teacher forcing): `<extra_id_0>` as the beginning-of-sequence token, followed by the target shifted right by one. Batches are padded with pad id 0; the encoder attention mask is `input_ids != 0`, and padded target positions are excluded from the loss. Generated ids are decoded with `skip_special_tokens=True, clean_up_tokenization_spaces=False`. |
| **Architecture** | `T5ForConditionalGeneration` initialized from the pretrained `google-t5/t5-small` checkpoint. No layers or adapters are added. **Frozen:** the shared token-embedding matrix (`model.shared`, 32,128 × 512), which serves as the encoder and decoder input embedding and, through weight tying, the output projection. **Fine-tuned:** all other parameters, i.e. all self-attention, cross-attention and feed-forward weights of the 6 encoder and 6 decoder blocks, all layer norms, and the relative-position attention biases. |
| **Hyperparameters** | Loss: token-level cross-entropy on non-pad target tokens. Optimizer: AdamW (β = 0.9, 0.999, ε = 1e-8), learning rate 5e-4, weight decay 0.01 (not applied to biases and LayerNorm weights). Schedule: cosine decay with 1 epoch of linear warm-up over 20 epochs. Batch size 16 (train), 32 (dev/test). Stopping criteria: evaluate dev Record F1 after every epoch (greedy decoding, max 512 new tokens), keep the checkpoint with the best dev Record F1, early stopping with patience 5, max 20 epochs. The best checkpoint (epoch 17) is used for dev and test predictions with greedy decoding. |

---

## Q6. Results and error analysis

Table 4: Development and test results.

| System | Query EM | F1 score |
|---|---|---|
| **Dev Results** | | |
| T5 fine-tuned — Full model | 72.53 | 87.79 |
| **Test Results** | | |
| T5 fine-tuning | XX.XX | XX.XX |

Table 5: Qualitative error analysis on the dev set.

| Error Type | Example of Error | Error Description | Statistics |
|---|---|---|---|
| **Syntax error: unbalanced parentheses** | NL: *"list all arrivals from any airport to baltimore on thursday morning arriving before 9am"*. Prediction ends with `... AND( flight_1.arrival_time < flight_1.departure_time AND 1 = 1 ) ) ) ) )` → `near ")": syntax error` | In long queries with deeply nested `AND( ... OR( ... ) )` blocks, the model loses track of the nesting depth and emits too many or too few closing parentheses. The decoder has no explicit notion of bracket balance, and such deeply nested conditions are rare in training. | 15/466 |
| **Undeclared / ambiguous table alias** | NL: *"what aircraft is co 1209"*. Prediction: `SELECT DISTINCT aircraft_1.aircraft_code FROM aircraft aircraft_1 WHERE aircraft_1.aircraft_code = 'CO' AND flight_1.flight_number = 1209` → `no such column: flight_1.flight_number` | The WHERE clause references an alias (`flight_1`, `fare_basis_1`, `airport_service_2`, …) that was never declared in the FROM clause, or an alias is declared twice. The model generates FROM and WHERE token by token without seeing the database schema, so the two parts can become inconsistent, especially for rare question types needing long join chains. | 11/466 |
| **Wrong constant values** | NL: *"please list nonstop flights from las vegas to new york on america west"*. Gold: `flight_1.airline_code = 'HP'`; prediction: `flight_1.airline_code = 'AA'`. NL: *"us air flights departing from charlotte around 1pm"*. Gold: `departure_time >= 1230 AND departure_time <= 1330`; prediction: `>= 1130 AND <= 1330`. | The query structure is correct but a literal is wrong. Typical cases are airline/airport codes not lexically related to the name (America West → `HP`), ground-transport types ("train" → `'RENTAL CAR'` instead of `'RAPID TRANSIT'`), and time values for vague expressions ("around 4 o'clock", "after 5", "early afternoon"), whose annotation conventions must be memorized from few examples. | 19/466 |
| **Missing / extra tables and joins** | NL: *"show me the flights to and from love field"*. Gold: `... airport_1.airport_code = 'DAL' OR flight_1.from_airport = airport_2.airport_code AND airport_2.airport_code = 'DAL'`; prediction keeps only the "to" direction and drops `airport airport_2`. | The model omits part of a compositional request (one of two directions, one of two airlines, the aircraft/equipment join) or adds an unnecessary join. These questions combine several constraints with "and/or", and the model falls back to a more frequent, simpler template. | 12/466 |
| **Wrong SELECT columns** | NL: *"define airline us"*. Gold: `SELECT DISTINCT airline_1.airline_code , airline_1.airline_name , airline_1.note FROM airline ...`; prediction: `SELECT DISTINCT airline_1.airline_code FROM airline ...` | The conditions are mostly correct, but the projected columns or the queried entity differ: missing columns such as `airline_name` or `ground_fare`, or confusing airport and airline codes for short abbreviations ("what does iah mean"). What to return for vague requests ("define", "show me about") is annotated inconsistently, so the model picks the most common projection. | 11/466 |
