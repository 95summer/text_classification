import pandas as pd
from config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score
import pickle

config = Config()

if __name__ == "__main__":
    # 1-读取数据
    df = pd.read_csv(config.process_train_datapath)

    # 2-获取特征数据目标值
    words = df['words']
    label = df['label']

    # 3-用TF-IDF表示词向量
    # 加载停用词
    with open(config.stopwords_datapath,encoding = "UTF-8") as f:
        stop_words_list = f.read().split()
    # stop_words_list = open(config.stopwords_datapath,encoding = "UTF-8").read().split()

    # 向量化
    tf_obj = TfidfVectorizer(stop_words = stop_words_list)
    x = tf_obj.fit_transform(words)

    # 4-训练集测试集划分
    x_train,x_test,y_train,y_test = train_test_split(x,label,test_size=0.2,random_state=1110,shuffle = True)

    # 5-模型训练
    """
            参数解释：
                n_estimators：弱学习器的个数
                n_jobs：指定多少个线程数
    """
    model = RandomForestClassifier(n_estimators=100,n_jobs=10)
    model.fit(x_train,y_train)

    # 6-模型评估
    """
            注意：我们现在是多分类问题，因此average的值不能是binary。
                macro：宏平均。适合多分类的场景，如果样本分布均衡推荐使用
                micro：微观平均。适合多分类的场景，如果样本分布不均衡推荐使用
    """
    y_pred = model.predict(x_test)
    print("精确率:",precision_score(y_test,y_pred,average='macro'))
    print("召回率:",recall_score(y_test, y_pred, average='macro'))
    print("F1值:",f1_score(y_test, y_pred, average='macro'))

    # 7-保存模型
    with open(config.rf_model_path,mode="wb") as f:
        pickle.dump(model,f)

    with open(config.tfidf_model_path,mode = "wb") as f:
        pickle.dump(tf_obj,f)