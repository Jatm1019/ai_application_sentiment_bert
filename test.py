import pandas as pd

from sudachipy import tokenizer
from sudachipy import dictionary

# 表示設定（これ大事！）
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)
df = pd.read_csv('wrime-ver2.tsv', sep='\t')
print(df.columns)
print(df.head())

# 欠損値チェックやラベルの分布確認
print(df.isnull().sum())

# 辞書とトークナイザーを初期化
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C  # 分割モード：A（細かい）〜C（大きい）

# 解析したい文章
text = "私は昨日、東京大学に行きました。"

# 形態素解析を実行
tokens = tokenizer_obj.tokenize(text, mode)

# 結果を表示
for token in tokens:
    print(f"{token.surface()}\t{token.part_of_speech()}")