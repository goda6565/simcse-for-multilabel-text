from transformers import AutoTokenizer

# モデル読み込み
base_model_name = "princeton-nlp/unsup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
