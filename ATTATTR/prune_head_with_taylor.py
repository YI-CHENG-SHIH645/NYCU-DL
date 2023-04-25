import os
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
    data_loader = DataLoader(data, sampler=SequentialSampler(data), batch_size=32)

    # Load a fine-tuned model
    state_dict = torch.load(path.join('models', f'model.{DATASET}.bin'), map_location=device)
    config = BertConfig(vocab_size=28996, num_labels=len(data_prep.labels()))
    model = BertForSequenceClassification\
        .from_pretrained('bert-base-cased',
                         state_dict=state_dict,
                         config=config).to(device)
    model.eval()

    num_head, num_layer = 12, 12
    head_importance = torch.zeros(num_layer, num_head).to(device)

    # input_ids, attention_mask, token_type_ids, gold_labels
    for batch in data_loader:
        print(batch)


if __name__ == '__main__':
    main()
