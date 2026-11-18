import argparse
import pickle
import re
from collections import Counter, defaultdict

CATEGORIES = [
    "Execution error: syntax",
    "Execution error: undeclared/ambiguous alias",
    "Execution error: timeout",
    "Wrong SELECT columns",
    "Wrong tables / joins",
    "Wrong constant values",
    "Wrong condition structure",
]


def read_lines(path):
    with open(path) as f:
        return [line.strip() for line in f]


def select_clause(q):
    m = re.match(r"\s*SELECT\s+(.*?)\s+FROM\s", q)
    return m.group(1) if m else q


def tables(q):
    return Counter(re.findall(r"\b(\w+) \1_\d+\b", q))


def literals(q):
    return Counter(re.findall(r"'[^']*'|\b\d+\b", q))


def categorize(gold, pred, err):
    if err:
        if "timed out" in err:
            return CATEGORIES[2]
        if "no such column" in err or "ambiguous column" in err:
            return CATEGORIES[1]
        return CATEGORIES[0]
    if select_clause(gold) != select_clause(pred):
        return CATEGORIES[3]
    if tables(gold) != tables(pred):
        return CATEGORIES[4]
    if literals(gold) != literals(pred):
        return CATEGORIES[5]
    return CATEGORIES[6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_sql", default="results/t5_ft_experiment_dev.sql")
    parser.add_argument("--pred_records", default="records/t5_ft_experiment_dev.pkl")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    nl = read_lines("data/dev.nl")
    gold = read_lines("data/dev.sql")
    pred = read_lines(args.pred_sql)
    with open("records/ground_truth_dev.pkl", "rb") as f:
        gold_recs, _ = pickle.load(f)
    with open(args.pred_records, "rb") as f:
        pred_recs, pred_errs = pickle.load(f)

    total = len(gold)
    wrong = [i for i in range(total) if pred_errs[i] or set(gold_recs[i]) != set(pred_recs[i])]
    sql_diff_same_records = [i for i in range(total)
                             if gold[i] != pred[i] and not pred_errs[i] and set(gold_recs[i]) == set(pred_recs[i])]

    groups = defaultdict(list)
    for i in wrong:
        groups[categorize(gold[i], pred[i], pred_errs[i])].append(i)

    print(f"Total dev queries: {total}")
    print(f"Record mismatch (incl. errors): {len(wrong)}/{total}")
    print(f"SQL differs but records identical: {len(sql_diff_same_records)}/{total}\n")
    for cat in CATEGORIES:
        idx = groups.get(cat, [])
        print(f"{cat}: {len(idx)}/{total}  ({len(idx)}/{len(wrong)} of wrong queries)")
    for cat in CATEGORIES:
        idx = groups.get(cat, [])
        if not idx:
            continue
        print(f"\n===== {cat} =====")
        for i in idx[:args.examples]:
            print(f"[{i}] NL:   {nl[i]}")
            print(f"     GOLD: {gold[i]}")
            print(f"     PRED: {pred[i]}")
            if pred_errs[i]:
                print(f"     ERR:  {pred_errs[i]}")
    print("\n===== SQL differs but records identical =====")
    for i in sql_diff_same_records[:args.examples]:
        print(f"[{i}] NL:   {nl[i]}")
        print(f"     GOLD: {gold[i]}")
        print(f"     PRED: {pred[i]}")


if __name__ == "__main__":
    main()
