import pandas as pd
from text_classification.RF.config import Config
from collections import Counter
import matplotlib.pyplot as plt

config = Config()
datapath = config.train_datapath
# datapath = config.test_datapath

if __name__ == '__main__':
    # 读取文件
    df = pd.read_csv(datapath,sep="\t",encoding="UTF-8",names=["text","label"])
    print(df.head())

    # 统计每种目标值的样本条数
    label_counts = Counter(df["label"])
    print(label_counts)

    # 统计每条句子的长度
    df["length"] = df["text"].str.len()
    print(df.head())

    # 最长、最短、均值、中位数等的统计
    print("句子最大长度：",df["length"].max())
    print("句子最短长度：",df["length"].min())
    print("句子均值长度：",df["length"].mean())
    print("句子中位数长度：",df["length"].median())

    # 绘制直方图
    df["length"].hist()
    plt.show()