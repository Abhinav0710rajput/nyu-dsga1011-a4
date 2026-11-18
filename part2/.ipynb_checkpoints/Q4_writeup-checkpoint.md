# Q4. Data statistics and processing

All statistics use the `google-t5/t5-small` tokenizer (`T5TokenizerFast`, 32,100-token vocabulary).
Lengths are numbers of T5 tokens; "vocabulary size" is the number of distinct T5 token ids that
occur in that split. Reproduce with `python data_stats.py` (from `part2/`).

## Table 1: before pre-processing

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

## Table 2: after pre-processing

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

## Pre-processing steps (`part2/preprocess.py`)

1. **Whitespace normalization** (NL and SQL): strip and collapse repeated whitespace.
2. **Task prefix** (NL): prepend `"translate English to SQL: "`, following T5's pre-training
   format. This adds 6 tokens to every input (17.10 → 23.10 incl. `</s>`).
3. **Replace `<` in SQL** with the placeholder `less_than`: the T5 vocabulary has no token for `<`,
   so 393 train (9.3%) and 45 dev (9.7%) queries contained `<unk>` and a model could never generate
   a correct `<` / `<=` comparison. `less_than` never occurs in the data; it is mapped back to `<`
   after generation. Decoding with `clean_up_tokenization_spaces=False` makes
   encode → decode → post-process reproduce all 4,691 train/dev queries exactly.

## Observations

- Inputs are short (~17 tokens) but SQL targets are long (~215 tokens, max 511 incl. `</s>`), so
  the decoder max length / generation length must be at least 512 tokens or long queries get cut off.
- The SQL vocabulary is small (~560 distinct tokens) and highly repetitive (table aliases such as
  `flight_1`, `airport_service_1`, join conditions), so most of the difficulty is in producing long,
  structured outputs rather than in vocabulary coverage.
- Dev statistics closely match train, so the dev set is a good proxy for model selection.
