import numpy as np
import random


def label_count(example):
    """ラベルの個数をカウント"""
    # ラベルの個数をカウントするための辞書
    label_counts = {}
    # 各ラベルセットに対してループ
    for data in example:
        labels = data["labels"]
        indices = np.nonzero(labels)[0]
        for index in indices:
            if index not in label_counts:
                label_counts[index] = 1
            label_counts[index] += 1

    # 辞書を値でソート
    sorted_label_counts = dict(
        sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
    )

    # ソートされたラベルインデックス
    sorted_label_keys = list(sorted_label_counts.keys())

    return sorted_label_keys


def freq_labeling(example, sorted_label_key):
    """データが持つラベルの中で最も頻度が高いものをそのデータのラベルとする"""
    sorted_label_key

    label_indices = np.nonzero(example["labels"])[0]

    most_freq_label = 0  # ラベルのインデックス
    most_freq_label_rank = 100  # ラベルの順位

    for label in label_indices:
        if most_freq_label_rank >= sorted_label_key.index(label):  # 順位
            most_freq_label = label
            most_freq_label_rank = sorted_label_key.index(label)

    # 新しい配列を作成して、選ばれたインデックスの位置だけ1にする
    new_arr = [0] * len(example["labels"])
    new_arr[most_freq_label] = 1
    example["labels"] = new_arr

    return example


def drop_unique_label(example, unique_label_dict):
    label = example["labels"]
    if unique_label_dict[str(label)] == 1:
        return False
    else:
        return True  # ユニークなラベルのサンプルは削除


def get_same_label(examples):
    same_label_dict = {}
    for example in examples:
        label = example["labels"]
        if label not in list(same_label_dict.keys()):
            same_label_dict[str(label)] = []
        same_label_dict[str(label)].append(example["text"])
    return same_label_dict


def set_same_label_text(example, same_label_dict):
    label = example["labels"]
    example["same_label_text"] = random.choice(same_label_dict[str(label)])
    return example


def create_same_label_datasets(examples):
    """各データにランダムで同じラベルを持つテキストを付与"""
    # ラベル一覧の取得
    unique_label_dict = {}
    for example in examples:
        label = example["labels"]
        unique_label_dict[str(label)] = unique_label_dict.get(str(label), 0) + 1

    # ラベルがユニークなものを削除
    filtered_examples = examples.filter(
        lambda example: drop_unique_label(example, unique_label_dict)
    )

    same_label_dict = get_same_label(filtered_examples)

    annoteted_sample = filtered_examples.map(
        lambda example: set_same_label_text(example, same_label_dict)
    )

    return annoteted_sample
