from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import set_seed

from bert.utils import label_count, freq_labeling, create_same_label_datasets
from train_configs.scl import training_args
from model_configs.princeton import base_model_name, tokenizer
from bert.metrics.knn import compute_metrics
from bert.collates.eval import eval_collate_fn
from bert.collates.scl import sup_train_collate_fn
from bert.models.scl import SimCSEModel

import wandb

wandb.init(project="SimCSE-for-multilabel", name="scl")

# 乱数のシードを設定する
set_seed(42)

# データ読み込み
train_dataset = load_dataset("Harutiin/eurlex-for-bert", split="train")
valid_dataset = load_dataset("Harutiin/eurlex-for-bert", split="validation")
test_dataset = load_dataset("Harutiin/eurlex-for-bert", split="test")

# peように訓練データを再構成(positive_ensured)
pe_train_dataset = create_same_label_datasets(train_dataset)

# 頻度によってリラベリング
label_count_list = label_count(valid_dataset)
valid_dataset = valid_dataset.map(
    lambda example: freq_labeling(example, label_count_list)
)

# 教師なしSimCSEのモデルを初期化する
sup_model = SimCSEModel(base_model_name, mlp_only_train=True)

# 訓練設定


class SimCSETrainer(Trainer):
    """SimCSEの訓練に使用するTrainer"""

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        検証・テストセットのDataLoaderでeval_collate_fnを使うように
        Trainerのget_eval_dataloaderをオーバーライド
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=64,
            collate_fn=eval_collate_fn,
            pin_memory=True,
        )


# Trainerを初期化する
trainer = SimCSETrainer(
    model=sup_model,
    args=training_args,
    data_collator=sup_train_collate_fn,
    train_dataset=pe_train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# パラメータを連続にする
for param in sup_model.parameters():
    param.data = param.data.contiguous()

# 教師なしSimCSEの訓練を行う
trainer.train()

# エンコーダを保存
encoder_path = "outputs/scl/encoder"
sup_model.encoder.save_pretrained(encoder_path)
tokenizer.save_pretrained(encoder_path)
