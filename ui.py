import streamlit as st
import pickle
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

t = Tokenizer()

filename = "model.pkl"

def tokenize(text):
    return [token.surface for token in t.tokenize(text)]

def text_pred(text,model,tokenizer, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt",padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k,v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        return pred

st.title("感情分析アプリ")

text=st.text_area("文章を入力してください")

if st.button("分析する"):
    # 保存したモデルの読み込み
    model_path = "./my_model" 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    result = text_pred(text,model,tokenizer,device)
    result_text = ""
    if result == 0:
        result_text="喜び"
    elif result == 1:
        result_text="悲しみ"
    elif result == 2:
        result_text="期待"
    elif result == 3:
        result_text="驚き"
    elif result == 4:
        result_text="怒り"
    elif result == 5:
        result_text="恐れ"
    elif result == 6:
        result_text="嫌悪"
    elif result == 7:
        result_text="信頼"

    st.write("結果："+result_text)