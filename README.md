# 旅游场景意图分类

基于 [CrossWOZ](https://github.com/thu-coai/CrossWOZ) 公开对话数据集的中文意图分类实验：给定一句用户对旅游客服说的话，判断它属于 **酒店 / 餐馆 / 景点 / 地铁 / 出租** 中的哪一个领域。

对比两种方案：传统机器学习基线（jieba + TF-IDF + 随机森林）与预训练模型微调（BERT）。

---

## 结果速览

| 模型 | 配置 | dev macro-F1 | test macro-F1 | test acc |
|---|---|---|---|---|
| 随机森林 | jieba + TF-IDF | 0.9255 | — | — |
| **BERT** | bert-base-chinese，max_len 64 / batch 64 / lr 5e-5 / 3 epoch | 0.9716 | **0.9719** | 0.9572 |

**分领域明细（BERT on test，3642 条）**

| 领域 | 精确率 | 召回率 | F1 | 样本数 |
|---|---|---|---|---|
| 酒店 | 0.954 | 0.949 | 0.952 | 1081 |
| 餐馆 | 0.979 | 0.945 | 0.962 | 1246 |
| 景点 | 0.932 | 0.972 | 0.952 | 1155 |
| 地铁 | 1.000 | 0.988 | **0.994** | 84 |
| 出租 | 1.000 | 1.000 | **1.000** | 76 |

---

## 数据

**来源**：[CrossWOZ](https://github.com/thu-coai/CrossWOZ)（清华大学 CoAI，ACL 2020），6012 组真人对话、约 8.4 万轮，每条对话带对话行为（dialog act）标注。

**本仓库不包含数据文件**（原始 json 共 445 MB，且属第三方数据集）。下载方式：

```bash
# GitHub 直连不可用时用 jsDelivr CDN
curl -L -o train.json.zip https://gcore.jsdelivr.net/gh/thu-coai/CrossWOZ@master/data/crosswoz/train.json.zip
curl -L -o val.json.zip   https://gcore.jsdelivr.net/gh/thu-coai/CrossWOZ@master/data/crosswoz/val.json.zip
curl -L -o test.json.zip  https://gcore.jsdelivr.net/gh/thu-coai/CrossWOZ@master/data/crosswoz/test.json.zip

# 解压到 data/raw/crosswoz/
```

**标签来源**：从每句话自带的 `dialog_act` 四元组 `[意图, 领域, 槽位, 值]` 里取第 2 位（领域）。例如：

```
"你好，麻烦帮我推荐一个门票免费的景点"
dialog_act = [["General","greet","none","none"],
              ["Inform","景点","门票","免费"],
              ["Request","景点","名称",""]]
                                    ↑
                                 取这个 "景点" 作为标签
```

**样本量**：36,285 / 3,615 / 3,642（train / dev / test），沿用 CrossWOZ 官方划分。

**清洗规则**：两句过滤掉不用的样本（共 6,061 条）——

1. `dialog_act` 第 2 位不是 5 个领域之一的（如 `thank`、`greet` 这类寒暄动作）
2. 一句话同时涉及多个领域的（标签不唯一）

---

## 目录结构

```
.
├── data/
│   ├── raw/
│   │   ├── crosswoz/            # 原始数据集（需自行下载，不入库）
│   │   ├── class.txt            # 5 个领域标签，行号即 label
│   │   └── stopwords.txt
│   ├── processed/
│   │   ├── train.txt            # 用户原话<TAB>领域编号
│   │   ├── dev.txt
│   │   ├── test.txt
│   │   └── data_preprocess_result/   # jieba 分词后的 csv（给随机森林用）
│   ├── parse_crosswoz.py        # 原始 json → 分类样本
│   ├── data_preprocess.py       # jieba 分词 → csv
│   └── data_exploration.py      # 类别分布与句长探查
│
├── BERT/
│   ├── config.py                # 路径与超参
│   ├── preprocessing.py         # Dataset / DataLoader / Tokenizer
│   ├── bert_model.py            # BERT + Linear 分类头
│   └── train_eval.py            # 每轮在 dev 上验证，保存最优权重
│
├── RF/
│   ├── config.py
│   ├── train.py                 # TF-IDF + 随机森林
│   ├── predict.py               # dev 集评估
│   └── predict_service.py       # 单条推理
│
└── LLM/
    └── predict.py               # 大模型零样本分类（对比实验）
```

---

## 环境依赖

```
Python 3.10
torch, transformers          # BERT 微调
scikit-learn, jieba          # 随机森林基线
pandas, numpy
tqdm
```

预训练权重路径在 `BERT/config.py` 的 `bert_path` 中配置（默认 `E:\PretrainedModel\bert-base-chinese`）。

---

## 快速开始

```bash
# 1. 解析原始数据 → data/processed/train|dev|test.txt
cd data
python parse_crosswoz.py --crosswoz raw/crosswoz --out .

# 2. jieba 分词 → csv（随机森林用）
python data_preprocess.py

# 3. 训练随机森林基线
cd ../RF
python train.py
python predict.py

# 4. BERT 微调
cd ../BERT
python train_eval.py
```

---

## 关键发现

**1. 少样本类别的 F1 反而最高**

地铁、出租各只占 2% 的样本（test 集分别 84 / 76 条），F1 却达到 0.994 / 1.000。原因是「地铁站」「车牌号」「打车」这类词汇的领域专属性极强，判别特征明显。

**2. 真正的难点是类别间的词汇重叠，不是样本量**

混淆全部集中在酒店 / 餐馆 / 景点三类。它们共享「评分」「附近」「推荐」「人均」这些高频词，仅靠词面特征无法区分。

**3. 跨领域样本必须丢弃**

CrossWOZ 中有相当比例的句子同时涉及两个领域（如「我要去这家酒店，能帮我叫辆车吗」），单标签分类无法表达，只能丢弃。这是数据规模从 4 万降到 3.6 万的主因。

---

## 已知限制

- 数据是公开学术数据集，标注由对话行为规则生成，非纯人工标注，存在一定噪声
- 类别不均衡（地铁/出租各占 2%），未做重采样或类别加权
- 测试集 3642 条，单条误差约 0.03 个百分点，指标建议按量级理解
- 未做交叉验证，只用了单次官方划分

---

## License

代码 MIT。数据集版权归 CrossWOZ 作者所有，请遵循其原始许可（详见其仓库）。
