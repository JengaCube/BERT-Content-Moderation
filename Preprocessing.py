import pandas as pd
import re
import os

os.mkdir("Outputs")

df = pd.read_csv('archive/train.csv', encoding="utf-8")
de = pd.read_csv('archive/test.csv', encoding="utf-8")


print("Initial Rows (Train): ", len(df))
print("Initial Rows (Test): ", len(de))

pd.set_option('display.max_colwidth', None)

#Format: ABC:ABC#ABC or ABC:ABC.ABC

RE_hyperlinks = re.compile(
    r'\b'
    r'[A-Za-z][A-Za-z_]*:'
    r'[A-Za-z0-9_()]+'
    r'(?:'
    r'#[A-Za-z0-9_().-]+'
    r'|'
    r'\.[A-Za-z0-9]{2,5}'
    r'|'
    r''
    r')'
    r'\b'
)

#Format: NNN.NNN.NNN.NNN

RE_IP_addresses = re.compile(
    r'\b'
    r'[0-9]{1,3}'
    r'\.'
    r'[0-9]{1,3}'
    r'\.'
    r'[0-9]{1,3}'
    r'\.'
    r'[0-9]{1,3}'
    r'\b'
)

print("1", df['comment_text'].str.findall(RE_hyperlinks).str.len().sum())
print("2", df['comment_text'].str.findall(RE_IP_addresses).str.len().sum())


df["comment_text"] = df["comment_text"].astype(str).str.replace(RE_hyperlinks, "<HyperLink>", regex=True)
df["comment_text"] = df["comment_text"].astype(str).str.replace(RE_IP_addresses, "<IP Address>", regex=True)


df.to_csv("archive/train.csv", index=False)

de["comment_text"] = de["comment_text"].astype(str).str.replace(RE_hyperlinks, "<HyperLink>", regex=True)
de["comment_text"] = de["comment_text"].astype(str).str.replace(RE_IP_addresses, "<IP Address>", regex=True)

de.to_csv("archive/test.csv", index=False)