# DS-GA 1011 — Assignment 4 Write-up

- **GitHub repository (code for Part 1 and Part 2):** https://github.com/Abhinav0710rajput/nyu-dsga1011-a4
- **Google Drive (T5 checkpoint used for the Q7 outputs):** https://drive.google.com/file/d/1coS0vjK2hXTE-UgW5FEUat2EFrTi9fYn/view?usp=sharing

| Question | Result |
|---|---|
| Q1 — BERT, original test set | 93.07% accuracy |
| Q2b — BERT, transformed test set | 88.09% accuracy (−4.98 points) |
| Q3 — augmented BERT, original / transformed test set | 92.74% / 89.76% |
| Q6 — T5 fine-tuned, dev (Query EM / Record F1) | 72.53 / 87.79 |
| Q7 — T5 fine-tuned, test (Query EM / Record F1) | `XX.XX` / `XX.XX` (Gradescope) |

---

# Part 1: Fine-tuning BERT for Sentiment Classification

## Q1. Fine-tune BERT

The training loop in `do_train` (`part1/main.py`) was completed. For each batch, it moves the
inputs to the GPU, computes the loss returned by `BertForSequenceClassification`, then runs
`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` and `lr_scheduler.step()`.
Default hyperparameters were not modified.

- Initial test (`--debug_train`): **89.2%** accuracy (> 88% required).
- Full training and evaluation on the IMDB test set: **93.07%** accuracy (> 91% required).
  Output: `out_original.txt`.

## Q2a. Data transformation design

### Transformation: WordNet synonym replacement + QWERTY keyboard typos

The transformation combines two edits that real users make when writing reviews: substituting
words with synonyms and making keyboard typos. It is implemented in `custom_transform` in
`part1/utils.py`.

**Word selection.** The review is processed word by word using the regex `[A-Za-z]+` over the
raw text, so punctuation, whitespace and HTML tags (e.g. `<br />`) are left exactly as they were.
A word is **never modified** if:

- it has fewer than 3 letters;
- it is a negation or function word (*not, no, nor, never, none, nothing, neither, nobody,
  nowhere, without, hardly, barely, scarcely, cannot, but, the, and, for, are, was, were, this,
  that, with, you, his, her, its, they, them, she, him, has, had, have, been, who, what, which,
  than, then, there, their, from*), or a preposition/particle (*about, above, across, after,
  against, along, among, around, before, behind, below, beneath, beside, between, beyond, down,
  during, into, near, off, onto, out, over, past, since, through, throughout, till, toward,
  towards, under, until, upon, within*);
- it is part of a contraction, i.e. directly adjacent to an apostrophe (*didn't*, *movie's*).

Every other word is processed independently, in this order:

1. **Synonym replacement (probability 0.5).** Applied only if the word has at least one
   adjective/adverb WordNet sense, has **no verb sense** (this excludes participles such as
   *said*, *thinking*, *fighting*, which WordNet also lists as adjectives), and is already in base
   form (WordNet's `morphy(word, pos)` returns the word unchanged). Take
   the word's **first (most frequent) adjective/adverb WordNet synset** and replace the word with
   a uniformly random single-word lemma of that synset that differs from the word. If there is
   no such lemma, the word is kept. Capitalized words in the middle of a sentence (likely proper
   nouns) are not replaced.
2. **Keyboard typo (probability 0.3).** Choose one letter uniformly among all positions except
   the first, and replace it with a uniformly random neighbouring key on a QWERTY keyboard
   (e.g. `o` → one of `i, p, k, l`; `a` → one of `q, w, s, z`).

In both steps the original capitalization is preserved (lowercase, Capitalized or ALL-CAPS).

**Parameters:** synonym probability 0.5, typo probability 0.3, minimum word length 3,
WordNet via NLTK 3.8.1.

**Example.**

> Original: It was like watching "America's Funniest Home Videos". The damn parents. [...] What
> crappy dancing. Horrible and beautiful at once.
>
> Transformed: It was likw watching "America's Funniest Home Vidwos". The damn parents. [...] What
> stinky dancing. Atrocuous and beautiful at once.

### Why this is a reasonable transformation

- **Typos are common in real reviews.** User reviews are typed quickly, often on phones, and
  hitting an adjacent key is the most frequent kind of typo. Only one letter per word is changed
  and the first letter is never touched, so every word stays readable.
- **Reviewers word the same opinion differently.** One person writes "boring", another
  "tedious". Restricting synonyms to base-form adjectives/adverbs and to their most common sense
  keeps the sentence grammatical and avoids wrong-sense substitutions. An earlier version allowed
  any word with an adjective sense and produced meaning-changing swaps such as *thinking* →
  *intelligent* and *boring* → *drilling*; excluding words with verb senses and prepositions
  removes these.
- **The label is preserved.** Negations are never modified and synonyms come from the same
  sense, so a human would assign the original and transformed review the same sentiment.
- **It is non-trivial for the model.** Misspelled words are split by BERT's WordPiece tokenizer
  into unfamiliar sub-word pieces, and the substituted synonyms are usually rarer than the
  original words, so the model sees an input distribution different from training even though
  the meaning is unchanged.

## Q2b. Implement the transformation

The transformation from Q2a is implemented in `custom_transform` in `part1/utils.py`.
Accuracy of the Q1 model on the transformed test set is **88.09%**, compared with **93.07%** on
the original test set: a drop of **4.98 points** (> 4 points). Output: `out_transformed.txt`.

## Q3. Data augmentation

`create_augmented_dataloader` (`part1/main.py`) samples 5,000 random training examples
(`shuffle(seed=42).select(range(5000))`), applies `custom_transform` to them, and concatenates
them with the full original training set (25,000 + 5,000 = 30,000 examples). A new model was
trained on this data with the same hyperparameters as in Q1.

| Model | Original test set | Transformed test set |
|---|---|---|
| Q1 model (original training data) | 93.07% | 88.09% |
| Q3 model (augmented training data) | 92.74% | 89.76% |
| Change | −0.33 | **+1.67** |

Outputs: `out_augmented_original.txt`, `out_augmented_transformed.txt`.

**(1) Did performance on the transformed test data improve?** Yes. Accuracy on the
transformed test set increased from 88.09% to 89.76% (+1.67 points). The gap between
original and transformed accuracy shrank from 4.98 to 2.98 points.

**(2) How did augmentation affect performance on the original test data?** It stayed almost
the same, dropping slightly from 93.07% to 92.74% (−0.33 points). That change is about the size
of the run-to-run variation from random initialization of the classification head and data
order, so augmentation neither clearly helped nor clearly hurt the original data.

**Intuitive explanation.** During training, the augmented model sees reviews containing keyboard
typos and rarer adjective/adverb synonyms, together with their unchanged labels. A misspelled
word such as *atrocuous* is split by WordPiece into unfamiliar sub-word pieces. The Q1 model never
learned that these pieces carry the sentiment of *atrocious*, so it relies on the remaining
context and makes more errors. Augmentation teaches the model to map these noisy or rarer forms
to the same sentiment, and to rely less on single exact surface words. Performance on the
transformed test set improves as a result. Only 1/6 of the augmented training data is
transformed, and the transformed reviews keep their meaning, so the model still mostly learns
from clean text. Its accuracy on the original test set therefore barely changes. The small drop
is consistent with part of the training budget going to a slightly different input distribution.

**Limitation.** The augmentation uses exactly the same transformation (same kind of typos, same
synonym rule and probabilities) that produces the transformed test set. The model is therefore
trained on the specific shift it is later tested on. In practice, the out-of-distribution shift
at test time is unknown. It may be different typo patterns, slang, sarcasm, another domain
(e.g. product reviews), or paraphrases that change sentence structure. Augmenting with one
hand-designed transformation does not make the model robust to those shifts, and the
improvement measured here overestimates its robustness to real OOD data. Even on this matched
shift, the gain is only partial (a 2.98-point gap remains).

---

# Part 2: Fine-tuning T5 for Text-to-SQL

## Q4. Data statistics and processing

All statistics use the `google-t5/t5-small` tokenizer (`T5TokenizerFast`, 32,100-token vocabulary).
Lengths are numbers of T5 tokens; "vocabulary size" is the number of distinct T5 token ids that
occur in that split. Reproduce with `python data_stats.py` (from `part2/`).

### Table 1: before pre-processing

Raw lines of `{train,dev}.nl` / `{train,dev}.sql`, tokenized without special tokens.

| Statistics Name | Train | Dev |
|---|---|---|
| Number of examples | 4225 | 466 |
| Mean sentence length | 17.10 | 17.07 |
| Max sentence length | 59 | 43 |
| Mean SQL query length | 216.37 | 210.05 |
| Max SQL query length | 510 | 502 |
| Vocabulary size (natural language) | 791 | 465 |
| Vocabulary size (SQL) | 555 | 395 |

### Table 2: after pre-processing

Model name: `google-t5/t5-small`. Text exactly as fed to the model: task prefix added to the
input, `<` replaced in the SQL, and the `</s>` EOS token appended by the tokenizer.

| Statistics Name | Train | Dev |
|---|---|---|
| Mean sentence length | 23.10 | 23.07 |
| Max sentence length | 65 | 49 |
| Mean SQL query length | 217.49 | 211.17 |
| Max SQL query length | 511 | 503 |
| Vocabulary size (natural language) | 796 | 470 |
| Vocabulary size (SQL) | 557 | 397 |

### Pre-processing steps (`part2/preprocess.py`)

1. **Whitespace normalization** (NL and SQL): strip and collapse repeated whitespace.
2. **Task prefix** (NL): prepend `"translate English to SQL: "`, following T5's pre-training
   format. This adds 6 tokens to every input (17.10 → 23.10 incl. `</s>`).
3. **Replace `<` in SQL** with the placeholder `less_than`: the T5 vocabulary has no token for `<`,
   so 393 train (9.3%) and 45 dev (9.7%) queries contained `<unk>` and a model could never generate
   a correct `<` / `<=` comparison. `less_than` never occurs in the data; it is mapped back to `<`
   after generation. Decoding with `clean_up_tokenization_spaces=False` makes
   encode → decode → post-process reproduce all 4,691 train/dev queries exactly.

### Observations

- Inputs are short (~17 tokens) but SQL targets are long (~215 tokens, max 511 incl. `</s>`), so
  the decoder max length / generation length must be at least 512 tokens or long queries get cut off.
- The SQL vocabulary is small (~560 distinct tokens) and highly repetitive (table aliases such as
  `flight_1`, `airport_service_1`, join conditions), so most of the difficulty is in producing long,
  structured outputs rather than in vocabulary coverage.
- Dev statistics closely match train, so the dev set is a good proxy for model selection.

## Q5. T5 fine-tuning details

Table 3: details of the best-performing T5 model configuration (fine-tuned).
Code: `part2/preprocess.py`, `part2/load_data.py`, `part2/t5_utils.py`, `part2/train_t5.py`;
command: `slurm/t5_train.sbatch`.

| Design choice | Description |
|---|---|
| **Data processing** | (1) Strip and collapse whitespace in both the natural-language (NL) input and the SQL target. (2) Prepend the task prefix `"translate English to SQL: "` to every NL input, matching T5's pre-training format. (3) Replace every `<` in the SQL targets with the placeholder `less_than` (which never occurs in the data), because the T5 vocabulary has no token for `<` and 9.3% of train / 9.7% of dev queries would otherwise contain `<unk>`; after generation `less_than` is mapped back to `<` before the query is executed. No other changes (no schema in the input, no lower-casing of SQL). |
| **Tokenization** | Default `google-t5/t5-small` tokenizer (`T5TokenizerFast`, SentencePiece, 32,100 tokens) for both encoder and decoder, truncation at 512 tokens (longest target is 511 tokens incl. `</s>`, so nothing is cut). Encoder input: tokenized prefixed NL + `</s>`. Decoder target: tokenized SQL + `</s>`. Decoder input (teacher forcing): `<extra_id_0>` as beginning-of-sequence token followed by the target shifted right by one (`[<extra_id_0>] + target[:-1]`). Batches are padded dynamically with the pad id 0; the encoder attention mask is `input_ids != 0`, and padded target positions are excluded from the loss. At inference the generated ids are decoded with `skip_special_tokens=True, clean_up_tokenization_spaces=False` (the default clean-up turns `" , "` into `", "` and breaks SQL exact match), then `less_than` → `<`. |
| **Architecture** | `T5ForConditionalGeneration` initialized from the pretrained `google-t5/t5-small` checkpoint (60.5M parameters: 6-layer encoder, 6-layer decoder, d_model 512, 8 heads, feed-forward size 2048). No layers are added and no adapters/LoRA are used. **Frozen:** the shared token-embedding matrix (`model.shared`, 32,128 × 512 = 16.4M parameters, 27.2% of the model), by setting `requires_grad = False`. In T5-small this one matrix is used as the encoder input embedding, the decoder input embedding, and (through weight tying, `tie_word_embeddings=True`) the output projection / LM head, so all three stay at their pretrained values. **Fine-tuned:** everything else (44.1M parameters): all self-attention, cross-attention and feed-forward weights of the 6 encoder and 6 decoder blocks, their layer norms, the final encoder/decoder layer norms, and the relative-position attention biases. The optimizer only receives parameters with `requires_grad = True`. Rationale: the SQL and question vocabulary is small (~560 / ~800 distinct tokens, Q4) and all of it is already in the pretrained vocabulary, so the pretrained token representations can be kept; freezing them reduces the number of trained parameters by about a quarter and limits overfitting on 4,225 training examples, while the transformer layers adapt to the text-to-SQL mapping. |
| **Hyperparameters** | Loss: token-level cross-entropy on non-pad target tokens. Optimizer: AdamW (β = 0.9, 0.999, ε = 1e-8), learning rate 5e-4, weight decay 0.01 (not applied to biases and LayerNorm weights). Schedule: cosine decay with 1 epoch of linear warm-up, over 20 epochs. Batch size 16 (train), 32 (dev/test). Stopping: evaluate on the dev set after every epoch (greedy decoding, max 512 new tokens, record F1 on the executed queries); keep the checkpoint with the best dev record F1; early stopping with patience 5 epochs, max 20 epochs. The best checkpoint is used for the final dev and test predictions (greedy decoding). In the final run dev F1 was still improving slowly, so early stopping never triggered. The job was terminated during the dev evaluation of epoch 18 (0-indexed). The best checkpoint is from epoch 17 (dev Record F1 87.79), and the dev/test outputs were produced from it with `part2/eval_t5.py`. Hardware: 1× NVIDIA A100 40GB. |

## Q6. Results and error analysis

### Quantitative results

Table 4: Development and test results (×100). "Full model" is the configuration in Q5 (fine-tuned
`google-t5/t5-small`, frozen shared embeddings, greedy decoding); the best checkpoint by dev
Record F1 was reached after epoch 17 (0-indexed) of 20.

| System | Query EM | F1 score |
|---|---|---|
| **Dev Results** | | |
| T5 fine-tuned — Full model | 72.53 | 87.79 |
| **Test Results** | | |
| T5 fine-tuning | XX.XX | XX.XX |

Dev learning curve of the full model (checkpoint selection by Record F1):

| Epoch | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | **17** |
|---|---|---|---|---|---|---|---|---|---|---|
| Record F1 | 30.80 | 63.53 | 71.79 | 77.47 | 77.59 | 83.18 | 84.89 | 86.10 | 87.56 | **87.79** |
| SQL EM | 13.52 | 40.77 | 54.51 | 61.80 | 63.95 | 68.88 | 69.74 | 70.82 | 71.67 | **72.53** |
| SQL error rate (%) | 41.63 | 16.09 | 15.88 | 12.66 | 11.37 | 10.94 | 8.15 | 6.22 | 5.36 | **5.79** |

### Qualitative error analysis

Produced with `part2/error_analysis.py` on the dev predictions of the full model
(`results/t5_ft_experiment_dev.sql`). Of the 466 dev queries, 338 are exact SQL matches,
72 return records different from the ground truth (or fail to execute), and 56 differ from the
gold SQL but return identical records. Each of the 72 incorrect queries is assigned to the first
matching class in the order below (execution error → SELECT clause → set of tables → set of
constants → remaining condition structure).

Table 5: Error analysis on the dev set.

| Error type | Example of error | Error description | Statistics |
|---|---|---|---|
| **Syntax error: unbalanced parentheses** | NL: *"list all arrivals from any airport to baltimore on thursday morning arriving before 9am"*. Prediction ends with `... AND( flight_1.arrival_time < flight_1.departure_time AND 1 = 1 ) ) ) ) )` → `OperationalError: near ")": syntax error` | The model generates a long query with deeply nested `AND( ... OR( ... ) )` blocks and loses track of the nesting depth, emitting one or more closing parentheses too many (or too few). 14 of the 15 cases have an unequal number of `(` and `)`. These are the longest queries in the data (60–130 SQL words); the decoder has no explicit notion of bracket balance and the nested time/day conditions are rare in training. | 15/466 |
| **Undeclared / ambiguous table alias (schema linking)** | NL: *"what aircraft is co 1209"*. Gold joins `aircraft`, `equipment_sequence`, `flight`, … Prediction: `SELECT DISTINCT aircraft_1.aircraft_code FROM aircraft aircraft_1 WHERE aircraft_1.aircraft_code = 'CO' AND flight_1.flight_number = 1209` → `no such column: flight_1.flight_number` | The WHERE clause refers to an alias (`flight_1`, `fare_basis_1`, `airport_service_2`, …) that was never declared in the FROM clause, or declares the same alias twice (ambiguous column). The model generates the FROM list and the WHERE conditions independently token by token and does not see the database schema, so the two parts can become inconsistent, especially for rare question types that need long join chains. | 11/466 |
| **Wrong constant values** | NL: *"please list nonstop flights from las vegas to new york on america west"*. Gold: `flight_1.airline_code = 'HP'`; prediction: `flight_1.airline_code = 'AA'`. NL: *"us air flights departing from charlotte around 1pm"*. Gold: `departure_time >= 1230 AND departure_time <= 1330`; prediction: `>= 1130 AND <= 1330`. | The query structure is correct, but a literal is wrong: airline codes whose names are not lexically related to the code (America West → `HP`, guessed `AA` from "America"), airport codes (`'O'` instead of `'MCO'` for "orlando international"), ground-transport types ("train" → `'RENTAL CAR'` instead of `'RAPID TRANSIT'`), and especially time values for vague expressions ("around 4 o'clock", "after 5", "early afternoon", "around 9 o'clock" at night), where the ATIS annotation conventions (±30 min windows, 12- vs 24-hour time) must be memorized from few examples. | 19/466 |
| **Missing / extra tables and joins** | NL: *"show me the flights to and from love field"*. Gold: `... airport_1.airport_code = 'DAL' OR flight_1.from_airport = airport_2.airport_code AND airport_2.airport_code = 'DAL'`; prediction only keeps the "to" direction and drops `airport airport_2`. NL: *"flights from dallas to baltimore on american and delta"*: prediction keeps only `'DL'`, drops `'AA'` and the Fort Worth airport join. | The model omits part of a compositional request (one of two directions, one of two airlines, the return leg, the aircraft/equipment join) or adds an unnecessary join (e.g. a `flight_stop` table). These questions combine several constraints with "and/or", and the model tends to fall back to the most frequent simpler template seen in training. | 12/466 |
| **Wrong SELECT columns** | NL: *"define airline us"*. Gold: `SELECT DISTINCT airline_1.airline_code , airline_1.airline_name , airline_1.note FROM airline ...`; prediction: `SELECT DISTINCT airline_1.airline_code FROM airline ...`. NL: *"what does iah mean"*: prediction selects from `airline` instead of `airport`. | The WHERE part is mostly right, but the projected columns (or the entity the question asks about) differ: missing extra columns such as `ground_fare` or `airline_name`, selecting `aircraft_code` instead of `flight_id`, or confusing airport vs. airline codes for short abbreviations. The annotation of what to return for vague requests ("show me about", "define", "what does X mean") is inconsistent in ATIS, so the model picks the most common projection. | 11/466 |
| Other: wrong condition structure / timeout | NL: *"i need to reverse the flight from pittsburgh to denver please"*: prediction keeps Pittsburgh → Denver instead of Denver → Pittsburgh. | Same tables and constants as the gold query, but conditions are attached to the wrong column (origin/destination swapped, wrong comparison), plus one generated query for a round-trip question (*"flights from boston to atlanta and the return flights"*) that exceeds the 120 s execution limit. | 4/466 |

**Remarks.**
- 56/466 predictions are not exact SQL matches but return exactly the same records, e.g. `departure_time <= 1000` instead of `( departure_time >= 0 AND departure_time <= 1000 )`, or reordered AND-conditions. This is why SQL EM (72.53) is much lower than Record EM (86.48).
- Some errors stem from label noise in the data: 5 dev and 29 train gold queries are a placeholder (`SELECT state_1.state_code FROM state state_1 WHERE state_1.state_code = 'ASD';`). On 2 of these dev examples the model produced a reasonable flight query and is counted as wrong; conversely, it sometimes generates this placeholder for unusual questions (e.g. *"what is ord"*).

## Q7. Performance evaluation

Submitted files: `t5_ft_experiment_test.sql` (432 generated SQL queries) and `t5_ft_experiment_test.pkl` (their database records), produced from the epoch 17 checkpoint (dev Record F1 87.79) with greedy decoding. Checkpoint: see the Google Drive link at the top.
