import torch

class Config:
    def __init__(self):
        # 有 GPU 就用 GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 处理后数据
        self.train_datapath = "../data/processed/train.txt"
        self.dev_datapath = "../data/processed/dev.txt"
        self.test_datapath = "../data/processed/test.txt"
        # 类别名在原始数据目录
        self.class_datapath = "../data/raw/class.txt"

        # 数据加载器参数
        self.batch_size = 64
        self.max_length = 64

        # Bert预训练模型的路径
        self.bert_path = r"E:\PretrainedModel\bert-base-chinese"

        # 目标值文件解析
        self.classname_list = [line.strip() for line in open(self.class_datapath,mode="r",encoding="UTF-8")]
        self.classname_len = len(self.classname_list)

        # 日志文件存储目录
        self.log_path = "./"

        # 训练好模型的保存路径
        self.save_model = "save_model/bert.pkl"

if __name__ == '__main__':
    config = Config()
