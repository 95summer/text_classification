class Config:
    def __init__(self):
        # 训练集
        self.train_datapath = "../data/train.txt"
        # 测试集
        self.test_datapath = "../data/class.txt"
        # 验证集
        self.dev_datapath = "../data/dev.txt"
        # 类别
        self.class_datapath = "../data/class.txt"
        # 训练集
        self.stopwords_datapath = "../data/stopwords.txt"

        # 预处理后训练集
        self.process_train_datapath = "./data_preprocess_result/train.csv"
        # 预处理后测试集
        self.process_test_datapath = "./data_preprocess_result/test.csv"
        # 预处理后验证集
        self.process_dev_datapath = "./data_preprocess_result/dev.csv"

        # 原始文件路径
        self.original_datapath = {
            "train": self.train_datapath,
            "test": self.test_datapath,
            "dev": self.dev_datapath
        }

        # 分词预处理后文件存放路径
        self.process_datapath = {
            "train": self.process_train_datapath,
            "test": self.process_test_datapath,
            "dev": self.process_dev_datapath
        }
        # 训练好的模型保存路径
        self.tfidf_model_path = "model/tfidf.pkl"
        self.rf_model_path = "model/rf.pkl"

        # 预测结果存储路径
        self.predict_result_path = "text_classification/RF/data_preprocess_result/dev.csv"
