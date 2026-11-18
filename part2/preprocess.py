"""Text pre-processing for T5 text-to-SQL. Shared by data_stats.py (Q4) and the data loader (Q5)."""
import re

# Task prefix prepended to every natural-language input (T5 was pretrained with task prefixes).
TASK_PREFIX = "translate English to SQL: "


def preprocess_nl(text):
    """Natural-language input fed to the T5 encoder."""
    text = re.sub(r"\s+", " ", text.strip())
    return TASK_PREFIX + text


# The T5 vocabulary has no token for "<" (it becomes <unk>), but ~9% of the SQL queries use
# "<" / "<=" comparisons. Replace it by a placeholder the tokenizer can represent, and map it
# back after generation. "less_than" never occurs in the data.
LT_PLACEHOLDER = "less_than"


def preprocess_sql(sql):
    """Target SQL fed to the T5 decoder: collapse repeated whitespace and replace "<"."""
    sql = re.sub(r"\s+", " ", sql.strip())
    return sql.replace("<", LT_PLACEHOLDER)


def postprocess_sql(generated):
    """Turn decoded model output back into executable SQL (inverse of preprocess_sql).

    Decode with tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
    the default clean-up turns " , " into ", ", which breaks exact match with the gold SQL. With it
    off, decode(encode(preprocess_sql(q))) -> postprocess_sql gives back every train/dev query exactly.
    """
    return generated.strip().replace(LT_PLACEHOLDER, "<")
