import numpy as np
from tqdm import tqdm
import torch


def embed_texts(
    texts: list[str], tokenizer, encoder, batch_size: int = 32
) -> np.ndarray:
    """SimCSEのモデルを用いてテキストの埋め込みを計算"""
    embeddings = []

    # バッチごとにテキストを処理
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Embedding texts", ncols=100
    ):  # tqdmで進捗表示
        batch_texts = texts[i : i + batch_size]

        # テキストにトークナイザを適用
        tokenized_texts = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to("cuda:0")

        # トークナイズされたテキストをベクトルに変換
        with torch.inference_mode():
            with torch.amp.autocast("cuda"):
                encoded_texts = encoder(**tokenized_texts).last_hidden_state[:, 0]

        # ベクトルをNumPyのarrayに変換
        emb = encoded_texts.cpu().numpy().astype(np.float32)
        # ベクトルのノルムが1になるように正規化
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        embeddings.append(emb)

    # バッチ処理した埋め込みを結合
    return np.concatenate(embeddings, axis=0)
