import torch
from tqdm import tqdm
from config import Config
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig


config = Config()

bert_tokenizer = BertTokenizer.from_pretrained(config.bert_path)
bert_model = BertModel.from_pretrained(config.bert_path)
bert_config = BertConfig.from_pretrained(config.bert_path)

def load_raw_file(datapath):
    with open(datapath,mode='r',encoding="Utf-8") as f:
        lines = f.readlines()

    result_list = []
    for line in tqdm(lines, desc="处理文件中"):
        line = line.strip()
        if line == "":
            continue

        title,label = line.split("\t")

        result_list.append((title,int(label)))

    return result_list

class NewDataset(Dataset):
    def __init__(self,data_list):
        self.data_list = data_list
        self.sample_len = len(self.data_list)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        index = min(max(index,0),self.sample_len-1)
        title,label = self.data_list[index]

        return title,label

# 对每个批次进行处理
def collate_fn(batch_data):
    titles,labels = zip(*batch_data)

    title_tensor = bert_tokenizer.batch_encode_plus(
        titles,
        padding = "max_length",
        truncation = True,
        max_length = config.max_length
    )

    input_ids = title_tensor.input_ids
    attention_mask = title_tensor.attention_mask

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    return input_ids,attention_mask,labels

def build_dataloader(datapath,shuffle = False):
    data_list = load_raw_file(datapath)

    dataset = NewDataset(data_list)

    dataloader = DataLoader(
        dataset = dataset,
        shuffle = shuffle,
        batch_size = config.batch_size,
        collate_fn = collate_fn
    )

    return dataloader

if __name__ == "__main__":
    for input_ids, attention_mask, labels in build_dataloader(config.dev_datapath, False):
        print("input_ids-->", input_ids)
        print("attention_mask-->", attention_mask)
        print("labels-->", labels)
        break
