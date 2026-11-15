class Config:
    def __init__(self):
        # 原始数据
        self.class_datapath = "../data/raw/class.txt"
        self.stopwords_datapath = "../data/raw/stopwords.txt"
        self.all_datapath = "../data/raw/intent_all.tsv"

        # 处理后数据
        self.train_datapath = "../data/processed/train.txt"
        self.test_datapath = "../data/processed/test.txt"
        self.dev_datapath = "../data/processed/dev.txt"

        # 原始文件路径
        self.original_datapath = {
            "train": self.train_datapath,
            "test": self.test_datapath,
            "dev": self.dev_datapath
        }

        # 分词预处理后文件存放路径
        self.process_train_datapath = "../data/processed/data_preprocess_result/train.csv"
        self.process_test_datapath = "../data/processed/data_preprocess_result/test.csv"
        self.process_dev_datapath = "../data/processed/data_preprocess_result/dev.csv"

        self.process_datapath = {
            "train": self.process_train_datapath,
            "test": self.process_test_datapath,
            "dev": self.process_dev_datapath
        }

        # 训练好的模型保存路径
        self.tfidf_model_path = "model/tfidf.pkl"
        self.rf_model_path = "model/rf.pkl"
