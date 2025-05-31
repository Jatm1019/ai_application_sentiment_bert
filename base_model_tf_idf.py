import pandas as pd
import numpy as np
import seaborn as sns

from sudachipy import tokenizer
from sudachipy import dictionary
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def tokenize(text):
    return [token.surface() for token in tokenizer_obj.tokenize(text, mode)]

df = pd.read_csv('wrime-ver2.tsv', sep='\t')
tokenizer_obj = dictionary.Dictionary().create()
tfid_obj = TfidfVectorizer(min_df=0.01, max_df=0.5, sublinear_tf=True)
mode = tokenizer.Tokenizer.SplitMode.B
plt.rcParams['font.family'] = 'Meiryo'



# 形態素解析を実行
df['wakati'] = df['Sentence'].apply(lambda x: " ".join(tokenize(x)))
X = tfid_obj.fit_transform(df["wakati"])

# 結果表示
print("ベクトル形状:", X.shape)
print("特徴語:", tfid_obj.get_feature_names_out())

df_label = df[[ "Avg. Readers_Joy","Avg. Readers_Sadness","Avg. Readers_Anticipation","Avg. Readers_Surprise","Avg. Readers_Anger","Avg. Readers_Fear","Avg. Readers_Disgust","Avg. Readers_Trust"]].dropna()
df["label"] = df_label.values.argmax(axis=1)
df = df[["Sentence", "label"]]


y = df["label"]
print(df['label'].value_counts())
# 学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

Y_pred = model.predict(X_test)
print(X_test)
print(Y_pred)

# オッズ
feature_names = tfid_obj.get_feature_names_out()
num_top_words = 20  # 表示する上位/下位の特徴語数
num_classes = len(model.classes_)  # クラス数を自動取得
fig, axes = plt.subplots(nrows=num_classes, ncols=1, figsize=(10, 3 * num_classes))
# クラス数が1の場合、axesが配列でなくなるので対応
if num_classes == 1:
    axes = [axes]


jp_font = 'MS Gothic'
for i in range(len(model.classes_)):
    coef = model.coef_[i]
    odds_ratio = np.exp(coef)
    top_indices = np.argsort(odds_ratio)[-30:]  # 上位30特徴語
    top_features = tfid_obj.get_feature_names_out()[top_indices]
    top_odds = odds_ratio[top_indices]

    plt.figure(figsize=(6, 10))
    sns.barplot(x=top_odds, y=top_features, palette="coolwarm", orient="h")
    plt.title(f"オッズ比（Class {model.classes_[i]}）", fontproperties=jp_font, fontsize=14)
    plt.xlabel("オッズ比", fontproperties=jp_font)
    plt.ylabel("特徴語", fontproperties=jp_font)
    plt.tight_layout()
    plt.show()

# 正解率グラフ
# 学習時とテスト時の精度を表示
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# 正解率グラフ（棒グラフ）
plt.bar(['Train', 'Test'], [train_acc, test_acc], color=['skyblue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy')
plt.ylim(0, 1)
plt.show()
# 混同行列の表示
cm = confusion_matrix(y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()





print(classification_report(y_true=y_test, y_pred=Y_pred))