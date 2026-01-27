import pandas as pd
import re

df = pd.read_csv('Dataset/train.csv', encoding="utf-8")
de = pd.read_csv('Dataset/test.csv', encoding="utf-8")


print("Initial Rows (Train): ", len(df))
print("Initial Rows (Test): ", len(de))

pd.set_option('display.max_colwidth', None)

RE_hyperlinks = re.compile(
    r'\b[A-Za-z][A-Za-z0-9_]*:'
    r'[A-Za-z0-9_()]+'
    r'(?:'
    r'#[A-Za-z0-9_().-]+'
    r'|'
    r'\.[A-Za-z0-9]{2,5}'
    r')'
    r'\b'
)

df["comment_text"] = df["comment_text"].astype(str).str.replace(RE_hyperlinks, "<HyperLink>", regex=True)

df.to_csv("Dataset/train.csv", index=False)