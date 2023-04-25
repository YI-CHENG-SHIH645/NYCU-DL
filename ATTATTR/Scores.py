from os import path
import json
import random
import numpy as np
import torch
import argparse
from DataPrep import MNLIPrep
from transformers import BertConfig
from Bert import BertForSequenceClassification

DATASET = 'mnli'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_index",
                        default=16,
                        type=int,
                        help="Get attr output of the target example.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=4,
                        type=int,
                        help="Num batch of an example.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare the data
    data_prep = MNLIPrep(f'{DATASET}_data')
    bert_input = data_prep.create_bert_input_dict(
        'dev_matched', slice(args.example_index, args.example_index+1))
    input_len = int(bert_input['attention_mask'][0].sum())

    # Load a fine-tuned model
    state_dict = torch.load(path.join('models', f'model.{DATASET}.bin'), map_location=device)
    config = BertConfig(vocab_size=28996, num_labels=len(data_prep.labels()))

    # input_ids: (batch_size, 512), word -> integer
    # token_type_ids: (batch_size, 512), word 是第一句的: 0, 是第二句的 1, 是 padding 0
    # attention_mask: (batch_size, 512), 是要算 att 的， 還是是 padding 所以不用計算 att
    # label_ids: (1, ), 0: contradiction, 1: neutral, 2: entailment

    # 這邊 model 打算自己寫
    model = BertForSequenceClassification\
        .from_pretrained('bert-base-cased',
                         state_dict=state_dict,
                         config=config).to(device)
    model.eval()

    res_attr = []
    att_all = []
    num_head, num_layer = 12, 12

    # bert input: dict keys: input_ids, attention_mask, token_type_ids, gold_labels
    # only one instance
    for tgt_layer in range(num_layer):
        # att: (12, 512, 512) / logits: (batch_size, num_labels)
        att, logits = model(**bert_input, tgt_layer=tgt_layer)
        pred_label = int(torch.argmax(logits))
        att_all.append(att)

        baseline = None
        # (64, 12, 512, 512) / (12, 512, 512)
        scaled_atts, step = scale_att(att.data, args.batch_size, args.num_batch, baseline)
        # 加底線是 inplace 的意思
        scaled_atts.requires_grad_(True)

        attr_all = None
        for i in range(args.num_batch):
            chunk = slice(i*args.batch_size, (i+1)*args.batch_size)
            batch_atts = scaled_atts[chunk]
            # (1, ) / (chunk, 12, 512, 512)
            tgt_prob, grad = (
                model(**bert_input, tgt_layer=tgt_layer,
                      tmp_score=batch_atts, pred_label=pred_label)
            )
            grad = grad.sum(dim=0)
            attr_all = grad if attr_all is None else torch.add(attr_all, grad)
        attr_all = attr_all[:, 0:input_len, 0:input_len] * step[:, 0:input_len, 0:input_len]
        # num_layers 個 (12, input_len, input_len)
        res_attr.append(attr_all.data)

    file_name = "attr_zero_base_exp{0}.json"
    with open(path.join(args.output_dir, file_name.format(args.example_index)), "w") as f_out:
        for grad in res_attr:
            res_grad = grad.tolist()
            output = json.dumps(res_grad)
            f_out.write(output + '\n')

    file_name = "att_zero_base_exp{0}.json"
    with open(path.join(args.output_dir, file_name.format(args.example_index)), "w") as f_out:
        for att in att_all:
            att = att[:, 0:input_len, 0:input_len]
            output = json.dumps(att.tolist())
            f_out.write(output + '\n')


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
