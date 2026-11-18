import re

TASK_PREFIX = "translate English to SQL: "


def preprocess_nl(text):
    text = re.sub(r"\s+", " ", text.strip())
    return TASK_PREFIX + text


LT_PLACEHOLDER = "less_than"


def preprocess_sql(sql):
    sql = re.sub(r"\s+", " ", sql.strip())
    return sql.replace("<", LT_PLACEHOLDER)


def postprocess_sql(generated):
    return generated.strip().replace(LT_PLACEHOLDER, "<")
