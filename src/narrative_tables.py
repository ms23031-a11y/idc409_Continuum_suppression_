def md_table(df):
    cols = list(df.columns); s = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for _, r in df.iterrows(): s += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return s
