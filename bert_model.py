import pandas as pd
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import evaluate
from torch import nn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
import torch

# トークナイザー、モデルの準備
model_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 分類用のモデルを取得
num_labels = 8 # 分類する種類の数

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device) # 学習用

def tokenize(batch):
    return tokenizer(batch["Sentence"], padding=True, truncation=True, max_length=128)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted")
    }
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # 出力を取得（logits）
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # カスタム損失関数：重み付きクロスエントロピー
        loss_fct = CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
df = pd.read_csv('wrime-ver2.tsv', sep='\t')

# 必要なカラムのみ抽出
df_label = df[[ "Avg. Readers_Joy","Avg. Readers_Sadness","Avg. Readers_Anticipation","Avg. Readers_Surprise","Avg. Readers_Anger","Avg. Readers_Fear","Avg. Readers_Disgust","Avg. Readers_Trust"]].dropna()
df["label"] = df_label.values.argmax(axis=1)
df = df[["Sentence", "label"]]
# ラベルごとの重みを計算
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df["label"]),
    y=df["label"]
)
# PyTorchテンソルに変換し、GPUに送る
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Datasetに変換
dataset = Dataset.from_pandas(df)

data_trans = dataset.map(tokenize, batched=True, batch_size=len(dataset))
# フォーマットをPyTorchに変換
data_trans.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
print(data_trans[:10])


# train/test 分割
dataset_dict = data_trans.train_test_split(test_size=0.1,shuffle=True)

dataset_dict_train=dataset_dict["train"]
dataset_dict_test=dataset_dict["test"]


# ハイパーパラメータ設定
training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_train_batch_size=8,
    num_train_epochs=3.0,
    save_strategy="epoch",
    eval_steps=20     
    )
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 学習実行する
# 今まで設定下物をTrainerクラスに渡して、学習する
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict_train,
    eval_dataset=dataset_dict_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

def predict(text):
    inputs = tokenizer(text, add_special_tokens=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    ps = nn.Softmax(1)(outputs.logits)

    max_p = torch.max(ps)
  #  result = torch.argmax(ps).item() if max_p > 0.8 else -1
    result = torch.argmax(ps).item()

    return result

result = predict('明日の天気は晴れでしょう')

print(result)
