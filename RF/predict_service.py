import jieba
import pickle
from config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

config = Config()

# 1-加载模型
with open(config.rf_model_path, mode='rb') as f:
    model: RandomForestClassifier = pickle.load(f)

with open(config.tfidf_model_path, mode='rb') as f:
    tf_obj: TfidfVectorizer = pickle.load(f)


# 2-对单条标题预测
def predict(new_data):
    # 分词取词
    words = " ".join(jieba.lcut(new_data["title"])[:30])
    # 向量化
    x = tf_obj.transform([words])
    # 预测
    y_pred_index = model.predict(x)
    y_pred_index = y_pred_index[0]
    # 取出类别
    id_class_dict = {i:class_name.strip() for i,class_name in enumerate(open(config.class_datapath,mode='r',encoding='UTF-8'))}
    # 通过ID获得类别名称
    pred_class_name = id_class_dict[y_pred_index]
    new_data["pred_class"] = pred_class_name

    return new_data

if __name__ == "__main__":
    new_data = {"title":"我画地为牢啊,厉娜!"}
    result = predict(new_data)
    print(result)