# ATTATTR

原版：https://github.com/YRdddream/attattr

modify to runnable

```commandline
    sudo docker run -it \
                    --rm \
                    --runtime=nvidia \
                    -v /home/ubuntu/NCTU_DL/attattr/:/attattr \
                    --ipc=host \
                    --privileged \
                    attattr bash
```

```commandline
python examples/generate_attrscore.py \
        --task_name mnli \
        --data_dir model_and_data/mnli_data/ \
        --bert_model bert-base-cased \
        --batch_size 16 \
        --num_batch 4 \
        --model_file model_and_data/model.mnli.bin \
        --get_att_score --get_att_attr \
        --output_dir model_and_data/mnli_scores \
        --example_index 16
```

```commandline
python examples/get_tokens_and_pred.py \
        --output_dir model_and_data/mnli_scores \
        --data_dir model_and_data/mnli_data \
        --bert_model bert-base-cased \
        --task_name mnli \
        --model_file model_and_data/model.mnli.bin
```

```commandline
python examples/attribution_tree.py \
        --attr_file model_and_data/mnli_scores/attr_zero_base_exp16.json \
        --tokens_file model_and_data/mnli_scores/tokens_and_pred_100.json \
        --task_name mnli --example_index 16
```

```commandline
python examples/prune_head_with_attr.py \
        --task_name mnli \
        --data_dir model_and_data/mnli_data/ \
        --bert_model bert-base-cased \
        --model_file model_and_data/model.mnli.bin \
        --output_dir model_and_data/mnli_pruning/ \
        --eval_batch_size 30
```

```commandline
python examples/run_adver_connection.py \
        --task_name mnli \
        --data_dir model_and_data/mnli_data/ \
        --bert_model bert-base-cased \
        --batch_size 16 \
        --num_batch 4 \
        --zero_baseline \
        --model_file model_and_data/model.mnli.bin \
        --output_dir model_and_data/adver_conn/ \
        --num_exp 10
```

References
* https://huggingface.co/transformers/_modules/transformers/models/bert/configuration_bert.html
* https://huggingface.co/transformers/_modules/transformers/models/bert/tokenization_bert.html
* https://huggingface.co/transformers/_modules/transformers/models/bert/tokenization_bert_fast.html
* https://huggingface.co/transformers/model_doc/bert.html#overview
* https://github.com/microsoft/unilm/blob/master/s2s-ft/s2s_ft/modeling_decoding.py
* https://github.com/shehzaadzd/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py
* https://github.com/lonePatient/Bert-Multi-Label-Text-Classification/issues/2
* https://github.com/huggingface/transformers/issues/420