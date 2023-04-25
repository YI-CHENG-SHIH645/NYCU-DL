import os
import json
from os import path
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import argparse
from DataPrep import MNLIPrep
from transformers import BertForSequenceClassification, BertConfig

DATASET = 'mnli'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples",
                        default=200,
                        type=int,
                        help="The number of dev examples to compute the model prediction.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare the data
    data_prep = MNLIPrep(f'{DATASET}_data')
    bert_input = data_prep.create_bert_input_dict('dev_matched', slice(0, args.num_examples))
    data = TensorDataset(*bert_input.values())
    data_loader = DataLoader(data, sampler=SequentialSampler(data), batch_size=1)

    # Load a fine-tuned model
    state_dict = torch.load(path.join('models', f'model.{DATASET}.bin'), map_location=device)
    config = BertConfig(vocab_size=28996, num_labels=len(data_prep.labels()))
    model = BertForSequenceClassification\
        .from_pretrained('bert-base-cased',
                         state_dict=state_dict,
                         config=config).to(device)
    model.eval()

    num_head, num_layer = 12, 12
    attr_every_head_record = [0] * num_head * num_layer

    # input_ids, attention_mask, token_type_ids, gold_labels
    for batch in data_loader:
        batch = [b.to(device) for b in batch]
        input_len = int(batch[2][0].sum())
        # bert input: dict keys: input_ids, attention_mask, token_type_ids, gold_labels
        # only one instance
        for tgt_layer in range(num_layer):
            # att: (12, 512, 512) / logits: (batch_size, num_labels)
            att, _ = model(*batch, tgt_layer=tgt_layer)

            # (64, 12, 512, 512) / (12, 512, 512)
            scaled_atts, step = scale_att(att.data, args.batch_size, args.num_batch)
            # 加底線是 inplace 的意思
            scaled_atts.requires_grad_(True)

            attr_all = None
            for i in range(args.num_batch):
                chunk = slice(i * args.batch_size, (i + 1) * args.batch_size)
                batch_atts = scaled_atts[chunk]
                # (1, ) / (chunk, 12, 512, 512)
                tgt_prob, grad = (
                    model(**bert_input, tgt_layer=tgt_layer,
                          tmp_score=batch_atts, pred_label=batch[3][0])
                )
                grad = grad.sum(dim=0)
                attr_all = grad if attr_all is None else torch.add(attr_all, grad)
            attr_all = attr_all[:, 0:input_len, 0:input_len] * step[:, 0:input_len, 0:input_len]
            for i in range(0, num_head):
                attr_every_head_record[tgt_layer * num_head + i] += float(attr_all[i].max())

    with open(os.path.join(args.output_dir, "head_importance_attr.json"), "w") as f_out:
        f_out.write(json.dumps(attr_every_head_record, indent=2, sort_keys=True))
        f_out.write('\n')


# 此處 batch_size 和 一般認知 訓練的 batch_size 不同
def scale_att(att, batch_size, num_batch, baseline=None):
    if baseline is None:
        baseline = torch.zeros_like(att)

    num_points = batch_size * num_batch
    scale = 1.0 / num_points

    # 這裡預設 att 就是一筆 (12, 512, 512)
    # 切 64 份，一份一個 step
    step = (att.unsqueeze(0) - baseline.unsqueeze(0)) * scale

    # res: (64, 12, 512, 512), 從 1/64 att_score ~ 1.0 att_score
    res = torch.cat([torch.add(baseline.unsqueeze(0), step*i) for i in range(num_points)], dim=0)

    # 這邊 step[0]: (12, 512, 512)，是 (att_score - 0) * 1/64
    return res, step[0]


if __name__ == '__main__':
    main()
