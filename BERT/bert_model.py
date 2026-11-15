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
        # 注意：不要在 no_grad 作用域内调用 linear，否则分类头也没有梯度
        bert_output = self.bert_model(input_ids = input_ids,attention_mask = attention_mask)

        return self.linear(bert_output.last_hidden_state[:,0])