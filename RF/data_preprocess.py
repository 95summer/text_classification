import jieba
import pandas as pd
from config import Config

config = Config()


def data_preprocessing(task_type, datapath):
    # 读取文件
    df = pd.read_csv(datapath, sep="\t", encoding="UTF-8", names=["text_classification", "label"])
    # 分词 只取前30个
    df["words"] = df["text_classification"].apply(lambda line: " ".join(jieba.lcut(line)[:30]))
    # 存储文件
    if task_type in datapath:
        df.to_csv(config.process_datapath.get(task_type), sep=",", header=True, index=False, mode="w", encoding="UTF-8")


if __name__ == "__main__":
    for task_type, datapath in config.original_datapath.items():
        data_preprocessing(task_type, datapath)
