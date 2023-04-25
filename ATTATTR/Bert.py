import torch
from torch import nn
import math
from transformers.models.bert.modeling_bert import \
    BertPreTrainedModel, BertPooler, BertEmbeddings, \
    BertSelfAttention, BertSelfOutput, BertIntermediate, \
    BertOutput


class BertEM(BertEmbeddings):
    def __init__(self, config):
        super(BertEM, self).__init__(config)

    def forward(
            self,
            input_ids=None,  # (batch_size, 512)
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
            tgt_layer=None,
    ):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # (batch_size, 512) -> (batch_size, 512, 768)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if tgt_layer >= 0:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAtt(BertSelfAttention):
    def __init__(self, config):
        super(BertSelfAtt, self).__init__(config)

    def forward(
            self,
            hidden_states,  # (batch_size, 512, 768) 含有 sentence / token_type / possibly position 資訊
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            tmp_score=None
    ):
        # 12 heads, 1 head: 768/12 = 64, all head size: 768
        mixed_query_layer = self.query(hidden_states)  # 768 -> 768 (Linear)
        mixed_key_layer = self.key(hidden_states)  # 768 -> 768 (Linear)
        mixed_value_layer = self.value(hidden_states)  # 768 -> 768 (Linear)
        # (batch_size, 512, 768)

        # view (batch_size, 512, 12, 64)
        # permute (batch_size, 12, 512, 64)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # (batch_size, 12, 512, 512)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # score = score / root(dk), where dk = 64
        attention_scores /= math.sqrt(self.attention_head_size)

        # 這麼做是因為現在的 attention_mask: 0.0 if we want else -10000.0 (丟 softmax 形同 0.0)
        attention_scores += attention_mask

        # (batch_size, 12, 512, 512)
        attention_prob = nn.Softmax(dim=-1)(attention_scores)
        if tmp_score is not None:
            attention_prob = tmp_score

        # (batch_size, 12, 512, 64)
        context_layer = torch.matmul(attention_prob, value_layer)

        # (batch_size, 512, 12, 64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # flatten as (batch_size, 512, 768)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if tmp_score is None:
            return context_layer, attention_prob
        else:
            return context_layer, None


# 繼承 nn.Module 的原因是因為我們需要使用自己的 SelfAtt
class BertAtt(nn.Module):
    def __init__(self, config):
        super(BertAtt, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask, tmp_score=None):
        # (batch_size, 512, 768) / (batch_size, 12, 512, 512)
        self_output, att_score = self.self(hidden_states, attention_mask, tmp_score=tmp_score)
        # 相加後 LayerNorm -> (batch_size, 512, 768)
        att_output = self.output(self_output, hidden_states)

        return att_output, att_score


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAtt(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, tmp_score=None):
        att_output, att_score = self.attention(hidden_states, attention_mask, tmp_score)
        # (batch_size, 512, 768) -> (batch_size, 512, 3072) (through gelu)
        intermediate_output = self.intermediate(att_output)
        layer_output = self.output(intermediate_output, hidden_states)

        return layer_output, att_score


class BertEN(nn.Module):
    def __init__(self, config):
        super(BertEN, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask,
            output_all_encoded_layers=True,
            tgt_layer=None,
            tmp_score=None
    ):

        all_encoder_layers, att_score = [], None
        for i, layer_module in enumerate(self.layer):
            if i == tgt_layer:
                hidden_states, att_score = layer_module(hidden_states, attention_mask, tmp_score)
            else:
                hidden_states, _ = layer_module(hidden_states, attention_mask, tmp_score)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, att_score


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEN(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            output_all_encoded_layers=True,
            tgt_layer=None,
            tmp_score=None
    ):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # (batch_size, 512, 768)
        embedding_output = self.embeddings(input_ids, token_type_ids, tgt_layer)
        # [num_layers 個 (batch_size, 512, 768)] / (batch_size, 12, 512, 512)
        encoded_layers, att_score = (
            self.encoder(
                embedding_output,
                extended_attention_mask,  # (batch_size, 1, 1, 512)
                output_all_encoded_layers=output_all_encoded_layers,
                tgt_layer=tgt_layer,
                tmp_score=tmp_score
            )
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, att_score


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            labels_ids=None,
            tgt_layer=None,
            tmp_score=None,
            pred_label=None
    ):
        _, pooled_output, att_score = (
            self.bert(input_ids, token_type_ids, attention_mask,
                      output_all_encoded_layers=False, tgt_layer=tgt_layer, tmp_score=tmp_score)
        )
        # first token: (batch_size, 768)  -1 ~ 1 (Tanh)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        prob = torch.softmax(logits, dim=-1)

        # gold label 被模型預測的機率
        tgt_prob = prob[:, labels_ids[0]]

        if tmp_score is None:
            return att_score[0], logits
        else:
            # gradient shape 同 tmp_score (chunk, 12, 512, 512)
            gradient = torch.autograd.grad(torch.unbind(prob[:, pred_label]), tmp_score)
            return tgt_prob, gradient[0]
