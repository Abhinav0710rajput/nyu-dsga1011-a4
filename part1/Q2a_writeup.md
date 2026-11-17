# Q2a. Data transformation design

## Transformation: WordNet synonym replacement + QWERTY keyboard typos

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

## Why this is a reasonable transformation

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
