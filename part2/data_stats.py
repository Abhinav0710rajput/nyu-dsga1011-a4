"""Q4: data statistics before (Table 1) and after (Table 2) pre-processing, using the T5 tokenizer.

Usage (from part2/):   python data_stats.py
"""
import os

from transformers import T5TokenizerFast

from preprocess import preprocess_nl, preprocess_sql

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f]


def stats(tokenizer, texts, add_special_tokens):
    ids = [tokenizer(t, add_special_tokens=add_special_tokens)["input_ids"] for t in texts]
    lengths = [len(x) for x in ids]
    vocab = {tok for x in ids for tok in x}
    return {
        "mean": sum(lengths) / len(lengths),
        "max": max(lengths),
        "vocab": len(vocab),
        "unk": sum(tokenizer.unk_token_id in x for x in ids),
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'Statistic':<38}{'Train':>12}{'Dev':>12}")
    for name, train_val, dev_val in rows:
        fmt = (lambda v: f"{v:>12.2f}") if isinstance(train_val, float) else (lambda v: f"{v:>12}")
        print(f"{name:<38}{fmt(train_val)}{fmt(dev_val)}")


def main():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    data = {split: (load_lines(f"{DATA_DIR}/{split}.nl"), load_lines(f"{DATA_DIR}/{split}.sql"))
            for split in ("train", "dev")}

    # Table 1: raw text, tokenized with the T5 tokenizer, no special tokens
    raw = {s: (stats(tokenizer, nl, False), stats(tokenizer, sql, False)) for s, (nl, sql) in data.items()}
    print_table("Table 1: before pre-processing (T5 tokens, no special tokens)", [
        ("Number of examples", len(data["train"][0]), len(data["dev"][0])),
        ("Mean sentence length", raw["train"][0]["mean"], raw["dev"][0]["mean"]),
        ("Max sentence length", raw["train"][0]["max"], raw["dev"][0]["max"]),
        ("Mean SQL query length", raw["train"][1]["mean"], raw["dev"][1]["mean"]),
        ("Max SQL query length", raw["train"][1]["max"], raw["dev"][1]["max"]),
        ("Vocabulary size (natural language)", raw["train"][0]["vocab"], raw["dev"][0]["vocab"]),
        ("Vocabulary size (SQL)", raw["train"][1]["vocab"], raw["dev"][1]["vocab"]),
        ("SQL queries containing <unk>", raw["train"][1]["unk"], raw["dev"][1]["unk"]),
    ])

    # Table 2: pre-processed text exactly as fed to the model (task prefix, whitespace
    # normalization, and the </s> EOS token the tokenizer appends)
    proc = {s: (stats(tokenizer, [preprocess_nl(x) for x in nl], True),
                stats(tokenizer, [preprocess_sql(x) for x in sql], True))
            for s, (nl, sql) in data.items()}
    print_table("Table 2: after pre-processing (as fed to T5, incl. task prefix and </s>)", [
        ("Mean sentence length", proc["train"][0]["mean"], proc["dev"][0]["mean"]),
        ("Max sentence length", proc["train"][0]["max"], proc["dev"][0]["max"]),
        ("Mean SQL query length", proc["train"][1]["mean"], proc["dev"][1]["mean"]),
        ("Max SQL query length", proc["train"][1]["max"], proc["dev"][1]["max"]),
        ("Vocabulary size (natural language)", proc["train"][0]["vocab"], proc["dev"][0]["vocab"]),
        ("Vocabulary size (SQL)", proc["train"][1]["vocab"], proc["dev"][1]["vocab"]),
        ("SQL queries containing <unk>", proc["train"][1]["unk"], proc["dev"][1]["unk"]),
    ])
    print(f"\nTokenizer: google-t5/t5-small (full vocabulary {len(tokenizer)} tokens)")


if __name__ == "__main__":
    main()
