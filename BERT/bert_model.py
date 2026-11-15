import torch
import torch.nn as nn
from config import Config
from transformers import BertModel
from transformers import BertConfig

config = Config()

class BertClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert_model = BertModel.from_pretrained(config.bert_path)

        in_features = BertConfig.from_pretrained(config.bert_path).hidden_size
        self.linear = nn.Linear(in_features = in_features,out_features = config.classname_len)

    def forward(self,input_ids,attention_mask):
        bert_output = self.bert_model(input_ids = input_ids,attention_mask = attention_mask)

        return self.linear(bert_output.last_hidden_state[:,0])
