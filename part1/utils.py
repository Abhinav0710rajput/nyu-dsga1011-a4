import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import re
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


### Transformation: WordNet synonym replacement + QWERTY keyboard typos
# The review text is edited in place (punctuation, spacing and HTML tags are left untouched).
# Each eligible word (alphabetic, >= 3 letters, not a negation/function word, not part of a
# contraction like "didn't") is independently:
#   1. if it is an adjective/adverb in base form with no verb sense, replaced with probability SYNONYM_PROB by a
#      synonym from its most common adjective/adverb WordNet sense, then
#   2. given one typo with probability TYPO_PROB: one letter (never the first) is swapped for a
#      neighbouring key on a QWERTY keyboard.
# Capitalization of the original word is preserved. Negations are never touched so the label is kept.

SYNONYM_PROB = 0.5
TYPO_PROB = 0.30
MIN_WORD_LEN = 3

QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj",
    "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb",
    "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

# Words that carry negation or are pure function words: never transformed.
PROTECTED_WORDS = {
    "not", "no", "nor", "never", "none", "nothing", "neither", "nobody", "nowhere", "without",
    "hardly", "barely", "scarcely", "cannot", "but", "the", "and", "for", "are", "was", "were",
    "this", "that", "with", "you", "his", "her", "its", "they", "them", "she", "him", "has",
    "had", "have", "been", "who", "what", "which", "than", "then", "there", "their", "from",
    # prepositions / particles (WordNet lists many of them as adjectives or adverbs)
    "about", "above", "across", "after", "against", "along", "among", "around", "before",
    "behind", "below", "beneath", "beside", "between", "beyond", "down", "during", "into",
    "near", "off", "onto", "out", "over", "past", "since", "through", "throughout", "till",
    "toward", "towards", "under", "until", "upon", "within",
}

WORD_RE = re.compile(r"[A-Za-z]+")
SYNONYM_POS = ("a", "s", "r")  # adjective, adjective satellite, adverb

_wordnet_cache = {}


def _match_case(new_word, original):
    if original.isupper():
        return new_word.upper()
    if original[0].isupper():
        return new_word[0].upper() + new_word[1:]
    return new_word


def _get_synonyms(word):
    # Only words that are purely adjectives/adverbs (no verb sense, so participles like "said",
    # "thinking" are skipped) and already in base form, and only the lemmas of their most frequent
    # adjective/adverb sense (avoids rare senses that change meaning).
    key = word.lower()
    if key not in _wordnet_cache:
        candidates = []
        all_synsets = wordnet.synsets(key)
        synsets = [s for s in all_synsets if s.pos() in SYNONYM_POS]
        has_verb_sense = any(s.pos() == "v" for s in all_synsets)
        if synsets and not has_verb_sense and wordnet.morphy(key, synsets[0].pos()) == key:
            for lemma in synsets[0].lemmas():
                name = lemma.name().lower()
                if name != key and name.isalpha():
                    candidates.append(name)
        _wordnet_cache[key] = sorted(set(candidates))
    return _wordnet_cache[key]


def _add_typo(word):
    # Replace one letter (not the first) with a neighbouring QWERTY key.
    positions = [i for i in range(1, len(word)) if word[i].lower() in QWERTY_NEIGHBORS]
    if not positions:
        return word
    i = random.choice(positions)
    replacement = random.choice(QWERTY_NEIGHBORS[word[i].lower()])
    if word[i].isupper():
        replacement = replacement.upper()
    return word[:i] + replacement + word[i + 1:]


def _transform_word(match):
    word = match.group(0)
    text, start, end = match.string, match.start(), match.end()
    in_contraction = (end < len(text) and text[end] == "'") or (start > 0 and text[start - 1] == "'")
    if len(word) < MIN_WORD_LEN or word.lower() in PROTECTED_WORDS or in_contraction:
        return word

    # Synonyms only for lowercase/sentence-initial words, not ALL-CAPS or mid-sentence proper nouns.
    is_proper_noun = word[0].isupper() and start > 0 and text[:start].rstrip()[-1:] not in ".!?"
    if not is_proper_noun and random.random() < SYNONYM_PROB:
        synonyms = _get_synonyms(word)
        if synonyms:
            word = _match_case(random.choice(synonyms), word)
    if random.random() < TYPO_PROB:
        word = _add_typo(word)
    return word


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    example["text"] = WORD_RE.sub(_transform_word, example["text"])

    ##### YOUR CODE ENDS HERE ######

    return example
